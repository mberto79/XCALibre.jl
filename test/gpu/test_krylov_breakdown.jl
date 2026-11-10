# Shadow-vector breakdown retry on the GPU (not in a runtests.jl tier: needs a GPU).
#   julia --project=<env with XCALibre and CUDA> test/gpu/test_krylov_breakdown.jl
using XCALibre, Test, CUDA, Accessors
using LinearAlgebra, SparseArrays

include(joinpath(@__DIR__, "..", "unit", "Solve", "test_krylov_breakdown.jl"))

# the solve paths of unit/Solve/test_krylov_breakdown.jl, with the matrix built on the device
function gpu_breakdown_solve(solver, precision; warm=false)
    backend = CUDABackend()
    dmesh = adapt(backend, mesh)
    dmodel = Physics(time=Steady(), solid=Solid{Uniform}(k=1.0), energy=Energy{Conduction}(), domain=dmesh)
    dBCs = assign(region=dmesh, (T = [Dirichlet(:left_wall, 50.0), Zerogradient(:right_wall),
        Dirichlet(:bottom_wall, 10.0), Zerogradient(:upper_wall)],))
    solvers = (T = SolverSetup(
        solver=solver, preconditioner=Jacobi(), convergence=1e-8, relax=1.0, rtol=0.0, atol=1e-12,
        itmax=1000, precision=precision),)
    config = Configuration(solvers=solvers, schemes=schemes,
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1),
        hardware=Hardware(backend=backend, workgroup=32), boundaries=dBCs)
    T = dmodel.energy.T
    T_eqn = (
        - Laplacian{schemes.T.laplacian}(dmodel.solid.rDf, T) == - Source(ScalarField(dmesh))
    ) → ScalarEquation(T, config.boundaries.T)
    discretise!(T_eqn, T, config)
    apply_boundary_conditions!(T_eqn, config.boundaries.T, nothing, 0.0, config)

    A = XCALibre.ModelFramework._A(T_eqn)
    rowptr, colval = Array(_rowptr(A)), Array(_colval(A))
    nzval = Array(_nzval(A))
    for i ∈ eachindex(rowptr)[1:end-1], nzi ∈ rowptr[i]:(rowptr[i+1] - 1)
        j = colval[nzi]
        nzval[nzi] = j == i ? 1.0 : (j < i ? -0.25 : 0.0)
    end
    copyto!(_nzval(A), nzval)
    n = length(rowptr) - 1
    Acsc = sparse(transpose(SparseMatrixCSC(n, n, Int.(rowptr), Int.(colval), nzval)))
    b = warm ? Acsc*ones(n) : zeros(n)
    b[1] += 1.0
    copyto!(XCALibre.ModelFramework._b(T_eqn), b)

    @reset T_eqn.preconditioner = set_preconditioner(solvers.T.preconditioner, T_eqn)
    @reset T_eqn.solver = _workspace(solvers.T, T_eqn)
    update_preconditioner!(T_eqn.preconditioner, dmesh, config)
    initialise!(T, warm ? 1.0 : 0.0)
    solve_system!(T_eqn, solvers.T, T, nothing, config)
    Array(T.values), Acsc \ b
end

@testset "GPU $(nameof(typeof(s))) $(nameof(typeof(p))) survives a shadow-vector breakdown" for (s, p, tol, warm) ∈ (
        (Bicgstab(), FullPrecision(), 1e-10, false), (Bicgstab(), FullPrecision(), 1e-10, true),
        (Cgs(), FullPrecision(), 1e-10, false), (Bicgstab(), MixedF32(), 1e-5, false))
    x, xs = gpu_breakdown_solve(s, p; warm=warm)
    @test all(isfinite, x)
    @test norm(x - xs)/norm(xs) < tol
end

@testset "GPU AMG coarse $(nameof(typeof(s))) survives a shadow-vector breakdown" for s ∈ (Bicgstab(), Cgs())
    x, xs = coarse_breakdown_solve(s, CUDABackend(); move=CuArray)
    @test all(isfinite, x)
    @test norm(x - xs)/norm(xs) < 1e-10
end
