# Phase 8E gate: distributed KOmega (RANS) vs serial simple!, per rank under mpiexec.
# Mirrors test_psimple.jl — the point is that the turbulence transported-scalar eqns route
# through wrap_eqn/sync so a non-Laminar model runs distributed and matches serial per cell.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

bfs_mesh() = UNV2D_mesh(
    joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "backwardFacingStep_10mm.unv"), scale=0.001)

k_in = 0.01
w_in = 100.0

bfs_komega_bcs(mesh) = assign(region=mesh, (
    U = [Dirichlet(:inlet, [0.5, 0.0, 0.0]), Extrapolated(:outlet),
         Wall(:wall, [0.0, 0.0, 0.0]), Symmetry(:top)],
    p = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0),
         Extrapolated(:wall), Symmetry(:top)],
    k = [Dirichlet(:inlet, k_in), Zerogradient(:outlet),
         Dirichlet(:wall, 1e-12), Symmetry(:top)],
    omega = [Dirichlet(:inlet, w_in), Zerogradient(:outlet),
         Dirichlet(:wall, w_in), Symmetry(:top)],
    nut = [Dirichlet(:inlet, k_in/w_in), Extrapolated(:outlet),
         Dirichlet(:wall, 0.0), Symmetry(:top)]))

function komega_case(mesh, bcs; iterations)
    model = Physics(
        time = Steady(),
        fluid = Fluid{Incompressible}(nu=1e-3),
        turbulence = RANS{KOmega}(),
        energy = Energy{Isothermal}(),
        domain = mesh)
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
            convergence=1e-15, relax=0.7, rtol=1e-8, atol=1e-12, itmax=2000),
        p = SolverSetup(solver=Cg(), preconditioner=Jacobi(),
            convergence=1e-15, relax=0.3, rtol=1e-8, atol=1e-12, itmax=2000),
        k = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
            convergence=1e-15, relax=0.6, rtol=1e-8, atol=1e-12, itmax=2000),
        omega = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
            convergence=1e-15, relax=0.6, rtol=1e-8, atol=1e-12, itmax=2000))
    schemes = (U=Schemes(divergence=Upwind), p=Schemes(),
               k=Schemes(divergence=Upwind), omega=Schemes(divergence=Upwind))
    runtime = Runtime(iterations=iterations, time_step=1, write_interval=-1)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
        hardware=Hardware(backend=CPU(), workgroup=64), boundaries=bcs(mesh))
    initialise!(model.momentum.U, [0.0, 0.0, 0.0])
    initialise!(model.momentum.p, 0.0)
    initialise!(model.turbulence.k, k_in)
    initialise!(model.turbulence.omega, w_in)
    initialise!(model.turbulence.nut, k_in/w_in)
    model, config
end

iterations = 100

gmesh = rank == 0 ? bfs_mesh() : nothing
ref = if rank == 0
    ms, cs = komega_case(gmesh, bfs_komega_bcs; iterations)
    simple!(ms, cs)
    (collect(ms.momentum.U.x.values), collect(ms.momentum.U.y.values),
     collect(ms.momentum.p.values), collect(ms.turbulence.k.values),
     collect(ms.turbulence.omega.values), collect(ms.turbulence.nut.values))
else
    nothing
end
Us_x, Us_y, ps, ks, ws, nuts = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
model, config = komega_case(dm, bfs_komega_bcs; iterations)
run!(model, config)

n = dm.partition.n_owned
nloc = n + dm.partition.n_ghost
orig = dm.orig_cells
owned_err(loc, ser) = maximum(abs.(Array(loc)[1:n] .- ser[orig[1:n]]); init=0.0)

dux = owned_err(model.momentum.U.x.values, Us_x)
duy = owned_err(model.momentum.U.y.values, Us_y)
dp = owned_err(model.momentum.p.values, ps)
dk = owned_err(model.turbulence.k.values, ks)
# omega is O(1e2+) at walls (wall function) — compare relative, not absolute
dw = owned_err(model.turbulence.omega.values, ws) / maximum(abs.(ws))
dnut = owned_err(model.turbulence.nut.values, nuts)

@testset "KOmega distributed (rank $rank)" begin
    @test dux < 1e-6
    @test duy < 1e-6
    @test dp < 1e-6
    @test dk < 1e-6
    @test dw < 1e-6
    @test dnut < 1e-6
    # ghosts synced to the converged solution
    kv = model.turbulence.k.values
    @test all(abs(kv[i] - ks[orig[i]]) < 1e-6 for i ∈ n+1:nloc)
end
rank == 0 && println("KOMEGA bfs n=$(MPI.Comm_size(comm)) dux=$dux dp=$dp dk=$dk dw=$dw dnut=$dnut")
