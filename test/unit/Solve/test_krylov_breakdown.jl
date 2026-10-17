using XCALibre
using Accessors
using LinearAlgebra
using SparseArrays

# BiCGStab and CGS break down (0/0) when the residual becomes orthogonal to their shadow vector.
# A unit lower-triangular matrix (one-way coupling, as upwind convection) with b nonzero in one cell
# triggers it after one step; every serial solve path must still reach the exact solution.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "finer_mesh_laplace.unv"))
hardware = Hardware(backend=CPU(), workgroup=1024)

model = Physics(time=Steady(), solid=Solid{Uniform}(k=1.0), energy=Energy{Conduction}(), domain=mesh)
BCs = assign(region=mesh, (T = [
    Dirichlet(:left_wall, 50.0), Zerogradient(:right_wall), Dirichlet(:bottom_wall, 10.0), Zerogradient(:upper_wall)],))
schemes = (T = Schemes(laplacian = Linear),)

function breakdown_solve(solver, precision; warm=false)
    solvers = (T = SolverSetup(
        solver=solver, preconditioner=Jacobi(), convergence=1e-8, relax=1.0, rtol=0.0, atol=1e-12,
        itmax=1000, precision=precision),)
    config = Configuration(solvers=solvers, schemes=schemes,
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1), hardware=hardware, boundaries=BCs)
    T = model.energy.T
    T_eqn = (
        - Laplacian{schemes.T.laplacian}(model.solid.rDf, T) == - Source(ScalarField(mesh))
    ) → ScalarEquation(T, config.boundaries.T)
    discretise!(T_eqn, T, config)
    apply_boundary_conditions!(T_eqn, config.boundaries.T, nothing, 0.0, config)

    # unit diagonal, couplings to lower-numbered cells only and b - Ax₀ = e₁, exact in every
    # precision; the warm start x₀ = 1 keeps the shadow M⁻¹(b - Ax₀) localised as well
    A = XCALibre.ModelFramework._A(T_eqn)
    rowptr, colval, nzval = _rowptr(A), _colval(A), _nzval(A)
    for i ∈ eachindex(rowptr)[1:end-1], nzi ∈ rowptr[i]:(rowptr[i+1] - 1)
        j = colval[nzi]
        nzval[nzi] = j == i ? 1.0 : (j < i ? -0.25 : 0.0)
    end
    n = length(rowptr) - 1
    Acsc = sparse(transpose(SparseMatrixCSC(n, n, Int.(rowptr), Int.(colval), copy(nzval))))
    b = XCALibre.ModelFramework._b(T_eqn)
    b .= warm ? Acsc*ones(n) : 0.0
    b[1] += 1.0

    @reset T_eqn.preconditioner = set_preconditioner(solvers.T.preconditioner, T_eqn)
    @reset T_eqn.solver = _workspace(solvers.T, T_eqn)
    update_preconditioner!(T_eqn.preconditioner, mesh, config)
    initialise!(T, warm ? 1.0 : 0.0)
    solve_system!(T_eqn, solvers.T, T, nothing, config)
    copy(T.values), Acsc \ b
end

@testset "$(nameof(typeof(s))) $(nameof(typeof(p))) survives a shadow-vector breakdown" for (s, p, tol) ∈ (
        (Bicgstab(), FullPrecision(), 1e-10), (Cgs(), FullPrecision(), 1e-10),
        (Bicgstab(), MixedF32(), 1e-5), (Cgs(), MixedF32(), 1e-5), (Bicgstab(), MixedF16(), 1e-2))
    x, xs = breakdown_solve(s, p)
    @test all(isfinite, x)
    @test norm(x - xs)/norm(xs) < tol
end

@testset "Bicgstab FullPrecision warm start survives a shadow-vector breakdown" begin
    x, xs = breakdown_solve(Bicgstab(), FullPrecision(); warm=true)
    @test all(isfinite, x)
    @test norm(x - xs)/norm(xs) < 1e-10
end

# coarsest-level Krylov solve of the AMG cycle (used on GPU backends), on a unit lower-bidiagonal matrix
function coarse_breakdown_solve(solver, backend=CPU(); n=10, move=identity)
    S = XCALibre.Solve
    rowptr = move(Int32[1; 2:2:2n])
    colval = move(Int32[1; reduce(vcat, ([i - 1, i] for i ∈ 2:n))])
    nzval = move([1.0; repeat([-0.5, 1.0], n - 1)])
    op = S._AMGCoarseOp(rowptr, colval, nzval, n, n, backend, 32)
    b = move([1.0; zeros(n - 1)])
    ws = S._workspace(solver, b)
    c = ws isa S._ShadowWorkspace ? similar(b) : nothing
    kb = S.AMGKrylovCoarse(ws, op, Diagonal(move(ones(n))), 0.0, 1e-12, 1000, nzval, 0, 0, c)
    x = similar(b)
    S._apply_coarse_direct!(x, kb, b)
    Array(x), [0.5^(i - 1) for i ∈ 1:n]
end

@testset "AMG coarse $(nameof(typeof(s))) survives a shadow-vector breakdown" for s ∈ (Bicgstab(), Cgs())
    x, xs = coarse_breakdown_solve(s)
    @test all(isfinite, x)
    @test norm(x - xs)/norm(xs) < 1e-10
end
