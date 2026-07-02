# Phase 3 PETSc assembly/SpMV/KSP tests; run per rank under mpiexec (see runtests_mpi.jl)
using XCALibre, PETSc, MPI, Test
using LinearAlgebra, SparseArrays
using PETSc: LibPETSc
using XCALibre.ModelFramework: _A, _b, _nzval

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# same setup as unit_test_laplace.jl, on either the global or a distributed mesh
function assemble_T(mesh)
    hardware = Hardware(backend=CPU(), workgroup=64)
    model = Physics(
        time = Steady(),
        solid = Solid{Uniform}(k=1.0),
        energy = Energy{Conduction}(),
        domain = mesh
    )
    BCs = assign(
        region = mesh,
        (
            T = [
                Dirichlet(:left_wall, 50.0),
                Zerogradient(:right_wall),
                Dirichlet(:bottom_wall, 10.0),
                Zerogradient(:upper_wall)
            ],
        )
    )
    solvers = SolverSetup(
        solver=Cg(), preconditioner=Jacobi(),
        convergence=1e-8, relax=1.0, rtol=1e-12, atol=1e-14, itmax=500)
    schemes = Schemes(laplacian=Linear)
    runtime = Runtime(iterations=1, write_interval=-1, time_step=1)
    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)
    initialise!(model.energy.T, 15)
    T_eqn = (
        Time{schemes.time}(model.solid.rhocp, model.energy.T)
        - Laplacian{schemes.laplacian}(model.solid.rDf, model.energy.T)
        ==
        - Source(ScalarField(mesh))
    ) → ScalarEquation(model.energy.T, config.boundaries.T)
    discretise!(T_eqn, model.energy.T, config)
    apply_boundary_conditions!(T_eqn, config.boundaries.T, nothing, 0.0, config)
    T_eqn, model, config
end

gmesh = rank == 0 ?
    UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "laplace_unit_5by5.unv")) :
    nothing

# serial reference on rank 0: SpMV, row sums, direct solution (by original cell id)
ref = if rank == 0
    T_eqn, _, _ = assemble_T(gmesh)
    Acsr = T_eqn.equation.A.parent
    nglobal = size(Acsr, 1)
    xg = [0.1i + sin(i) for i ∈ 1:nglobal]
    (nglobal, xg, Acsr * xg, Acsr * ones(nglobal), Matrix(Acsr) \ Vector(T_eqn.equation.b))
else
    nothing
end
nglobal, xg, yref, rsref, xsol = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
part = dm.partition
n_owned = part.n_owned
orig = dm.orig_cells

T_eqn, model, config = assemble_T(dm)
s = PETScSolver(T_eqn, dm, config.solvers)
passemble!(s, T_eqn, part)

_, y = LibPETSc.MatCreateVecs(s.petsclib, s.A)
owned_vals(v) = PETSc.withlocalarray!(copy, v; read=true, write=false)

@testset "Phase 3 assembly (rank $rank)" begin
    # owned block is contiguous in the global row numbering
    @test all(part.local_to_global[i] == part.row_start + i - 1 for i ∈ 1:n_owned)

    # MatMult matches serial SpMV
    PETSc.withlocalarray!(s.x; read=false, write=true) do arr
        for i ∈ 1:n_owned
            arr[i] = xg[orig[i]]
        end
    end
    LinearAlgebra.mul!(y, s.A, s.x)
    @test maximum(abs.(owned_vals(y) .- yref[orig[1:n_owned]]); init=0.0) <= 1e-12

    # transpose MatMult identical (operator is symmetric)
    LibPETSc.MatMultTranspose(s.petsclib, s.A, s.x, y)
    @test maximum(abs.(owned_vals(y) .- yref[orig[1:n_owned]]); init=0.0) <= 1e-12

    # global row sums match serial
    PETSc.withlocalarray!(a -> fill!(a, 1.0), s.x; read=false, write=true)
    LinearAlgebra.mul!(y, s.A, s.x)
    @test maximum(abs.(owned_vals(y) .- rsref[orig[1:n_owned]]); init=0.0) <= 1e-12

    # KSP CG solve matches serial direct solution
    Tvals = model.energy.T.values
    psolve!(s, Tvals)
    @test maximum(abs.(Tvals[1:n_owned] .- xsol[orig[1:n_owned]]); init=0.0) <= 1e-8

    # adjoint solve (symmetric system => same solution)
    Tt = fill(15.0, length(Tvals))
    psolve_transpose!(s, Tt)
    @test maximum(abs.(Tt[1:n_owned] .- xsol[orig[1:n_owned]]); init=0.0) <= 1e-8

    # values-only re-assembly: scaled coefficients give scaled MatMult
    _nzval(_A(T_eqn)) .*= 2
    passemble!(s, T_eqn, part)
    PETSc.withlocalarray!(s.x; read=false, write=true) do arr
        for i ∈ 1:n_owned
            arr[i] = xg[orig[i]]
        end
    end
    LinearAlgebra.mul!(y, s.A, s.x)
    @test maximum(abs.(owned_vals(y) .- 2 .* yref[orig[1:n_owned]]); init=0.0) <= 1e-11
end
