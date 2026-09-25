# Restart: a run resumed from the results written at iteration 50 must retrace the straight run's last 50
# iterations (SIMPLE laminar, PISO laminar, SIMPLE k-omega SST) under Jacobi.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

mesh_path = joinpath(pkgdir(XCALibre), "examples/0_GRIDS/3d_box_1000x1000x1000mm_5.unv")
walls = (:y_min, :y_max, :z_min, :z_max)
k_in, w_in = 0.01, 100.0

function restart_case(mesh, kind; iterations, write_interval)
    sst = kind == :sst
    model = Physics(time=kind == :piso ? Transient() : Steady(), fluid=Fluid{Incompressible}(nu=1e-3),
        turbulence=sst ? RANS{KOmegaSST}(walls=walls) : RANS{Laminar}(), energy=Energy{Isothermal}(), domain=mesh)
    wall(v) = [Dirichlet(w, v) for w ∈ walls]
    bcs = (
        U = [Dirichlet(:x_min, [0.5, 0.0, 0.0]), Extrapolated(:x_max), [Wall(w, [0.0, 0.0, 0.0]) for w ∈ walls]...],
        p = [Extrapolated(:x_min), Dirichlet(:x_max, 0.0), [Extrapolated(w) for w ∈ walls]...])
    sst && (bcs = (; bcs...,
        k = [Dirichlet(:x_min, k_in), Zerogradient(:x_max), wall(1e-12)...],
        omega = [Dirichlet(:x_min, w_in), Zerogradient(:x_max), wall(w_in)...],
        nut = [Dirichlet(:x_min, k_in / w_in), Extrapolated(:x_max), wall(0.0)...]))
    tight(s, relax) = SolverSetup(solver=s, preconditioner=Jacobi(), convergence=1e-15, relax=relax,
        rtol=1e-8, atol=1e-12, itmax=2000)
    solvers = (U=tight(Bicgstab(), kind == :piso ? 1.0 : 0.7), p=tight(Cg(), kind == :piso ? 1.0 : 0.3))
    sst && (solvers = (; solvers..., k=tight(Bicgstab(), 0.6), omega=tight(Bicgstab(), 0.6), y=tight(Cg(), 1.0)))
    schemes = (U=Schemes(divergence=Upwind, time=kind == :piso ? Euler : SteadyState), p=Schemes())
    sst && (schemes = (; schemes..., k=Schemes(divergence=Upwind), omega=Schemes(divergence=Upwind), y=Schemes()))
    runtime = Runtime(iterations=iterations, time_step=kind == :piso ? 0.1 : 1, write_interval=write_interval)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
        hardware=Hardware(backend=CPU(), workgroup=64), boundaries=assign(region=mesh, bcs))
    initialise!(model.momentum.U, [0.0, 0.0, 0.0]); initialise!(model.momentum.p, 0.0)
    if sst
        initialise!(model.turbulence.k, k_in); initialise!(model.turbulence.omega, w_in)
        initialise!(model.turbulence.nut, k_in / w_in)
    end
    model, config
end

gmesh = rank == 0 ? UNV3D_mesh(mesh_path, scale=0.001) : nothing
dm = distribute(gmesh; comm)
N, K = 100, 50
for kind ∈ (:simple, :piso, :sst)
    tmp = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm; root=0)
    cd(tmp)
    m1, c1 = restart_case(dm, kind; iterations=N, write_interval=K)
    r1 = run!(m1, c1; output=OpenFOAM())
    m2, c2 = restart_case(dm, kind; iterations=N, write_interval=-1)
    r2 = run!(m2, c2; restart=kind == :piso ? K * 0.1 : K)
    n = dm.partition.n_owned
    rel(a, b) = maximum(abs.(a .- b); init=0.0) / max(maximum(abs.(a); init=0.0), eps())
    dres = rel(r1.p[K+1:N], r2.p[K+1:N])
    dU = rel(m1.momentum.U.x.values[1:n], m2.momentum.U.x.values[1:n])
    dp = rel(m1.momentum.p.values[1:n], m2.momentum.p.values[1:n])
    dmax = MPI.Allreduce(max(dres, dU, dp), max, comm)
    @testset "restart $kind (rank $rank)" begin
        @test dmax <= 1e-10
        # the resumed run never executed iterations 1:K: their residual slots keep the initial fill
        @test all(==(r2.p[1]), r2.p[1:K]) && r2.p[1] ∈ (0, 1) && r1.p[1:K] != r2.p[1:K]
        kind == :sst && @test MPI.Allreduce(rel(m1.turbulence.nut.values[1:n], m2.turbulence.nut.values[1:n]), max, comm) <= 1e-10
    end
    rank == 0 && println("RESTART $kind n=$(MPI.Comm_size(comm)) residual/field max rel diff = $dmax")
    MPI.Barrier(comm)
    rank == 0 && rm(tmp; recursive=true)
end
