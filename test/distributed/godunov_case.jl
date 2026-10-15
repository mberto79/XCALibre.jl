# shared density-based (godunov!) case for the distributed tests: Mach 1.2 flow past a 2D
# cylinder, laminar or KOmegaSST with Dirichlet walls (no wall functions) so the serial and
# distributed runs match to round-off.

cylinder_mesh() = UNV2D_mesh(
    joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "cylinder_d10mm_5mm.unv"), scale=0.001)

const g_gamma = 1.4
const g_cp = 1005.0
const g_T = 300.0
const g_p = 101325.0
const g_U = 1.2 * sqrt(g_gamma * g_cp * (1.0 - 1.0/g_gamma) * g_T)
const g_k = 1.5 * (0.01 * g_U)^2
const g_w = 1e5

function godunov_bcs(mesh, sst)
    U = [Dirichlet(:inlet, [g_U, 0.0, 0.0]), Zerogradient(:outlet),
         Wall(:cylinder, [0.0, 0.0, 0.0]), Slip(:top), Slip(:bottom)]
    p = [Dirichlet(:inlet, g_p), Zerogradient(:outlet), Zerogradient(:cylinder),
         Zerogradient(:top), Zerogradient(:bottom)]
    T = [Dirichlet(:inlet, g_T), Zerogradient(:outlet), Zerogradient(:cylinder),
         Zerogradient(:top), Zerogradient(:bottom)]
    sst || return assign(region=mesh, (; U, p, T))
    k = [Dirichlet(:inlet, g_k), Zerogradient(:outlet), Dirichlet(:cylinder, 0.0),
         Zerogradient(:top), Zerogradient(:bottom)]
    omega = [Dirichlet(:inlet, g_w), Zerogradient(:outlet), Dirichlet(:cylinder, 1e6),
         Zerogradient(:top), Zerogradient(:bottom)]
    nut = [Dirichlet(:inlet, g_k/g_w), Zerogradient(:outlet), Dirichlet(:cylinder, 0.0),
         Zerogradient(:top), Zerogradient(:bottom)]
    assign(region=mesh, (; U, p, T, k, omega, nut))
end

function godunov_case(mesh; sst=false, iterations=30, flux=HLLC(), reconstruction=MUSCL{VanLeer}(),
        time_stepping=RK2(), adaptive=nothing, gradient=Gauss, backend=CPU())
    model = Physics(
        time = Transient(),
        fluid = Fluid{SupersonicFlow}(nu=1e-5, cp=g_cp, gamma=g_gamma, Pr=0.7),
        turbulence = sst ? RANS{KOmegaSST}(walls=(:cylinder,)) : RANS{Laminar}(),
        energy = Energy{SensibleEnthalpy}(Tref=0.0),
        domain = mesh)
    tight() = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
        convergence=1e-15, relax=1.0, rtol=1e-13, atol=1e-16, itmax=2000)
    solvers = sst ? (rho=(convergence=1e-15,), k=tight(), omega=tight()) : (rho=(convergence=1e-15,),)
    schemes = (U=Schemes(gradient=gradient), p=Schemes(gradient=gradient),
        T=Schemes(gradient=gradient), k=Schemes(), omega=Schemes(),
        flux=flux, reconstruction=reconstruction, time_stepping=time_stepping)
    runtime = adaptive === nothing ?
        Runtime(iterations=iterations, write_interval=-1, time_step=1e-7) :
        Runtime(iterations=iterations, write_interval=-1, time_step=1e-7, adaptive=adaptive)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
        hardware=Hardware(backend=backend, workgroup=64), boundaries=godunov_bcs(mesh, sst))
    initialise!(model.momentum.U, [g_U, 0.0, 0.0])
    initialise!(model.momentum.p, g_p)
    initialise!(model.energy.T, g_T)
    if sst
        initialise!(model.turbulence.k, g_k)
        initialise!(model.turbulence.omega, g_w)
        initialise!(model.turbulence.nut, g_k/g_w)
    end
    model, config
end

# cell fields compared between the serial and distributed runs
godunov_fields(model) = begin
    f = [model.momentum.U.x, model.momentum.U.y, model.momentum.p, model.energy.T, model.fluid.rho]
    model.turbulence isa KOmegaSST &&
        append!(f, [model.turbulence.k, model.turbulence.omega, model.turbulence.nut])
    f
end

# 3D: Mach 1.2 stream through a box, no-slip floor (wall distance, SST), slip elsewhere
box_mesh() = UNV3D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "3d_box_1000x1000x1000mm_5.unv"), scale=0.001)

function box_bcs(mesh)
    slip = (:y_max, :z_min, :z_max)
    zg(names...) = [Zerogradient(s) for s ∈ names]
    assign(region=mesh, (
        U = [Dirichlet(:x_min, [g_U, 0.0, 0.0]), Zerogradient(:x_max),
             Wall(:y_min, [0.0, 0.0, 0.0]), [Slip(s) for s ∈ slip]...],
        p = [Dirichlet(:x_min, g_p), zg(:x_max, :y_min, slip...)...],
        T = [Dirichlet(:x_min, g_T), zg(:x_max, :y_min, slip...)...],
        k = [Dirichlet(:x_min, g_k), Zerogradient(:x_max), Dirichlet(:y_min, 0.0), zg(slip...)...],
        omega = [Dirichlet(:x_min, g_w), Zerogradient(:x_max), Dirichlet(:y_min, 1e6), zg(slip...)...],
        nut = [Dirichlet(:x_min, g_k/g_w), Zerogradient(:x_max), Dirichlet(:y_min, 0.0), zg(slip...)...]))
end

function box_case(mesh; iterations=10, write_interval=-1)
    model = Physics(
        time = Transient(),
        fluid = Fluid{SupersonicFlow}(nu=1e-5, cp=g_cp, gamma=g_gamma, Pr=0.7),
        turbulence = RANS{KOmegaSST}(walls=(:y_min,)),
        energy = Energy{SensibleEnthalpy}(Tref=0.0),
        domain = mesh)
    tight() = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
        convergence=1e-15, relax=1.0, rtol=1e-13, atol=1e-16, itmax=2000)
    schemes = (U=Schemes(), p=Schemes(), T=Schemes(), k=Schemes(), omega=Schemes(),
        flux=Rusanov(), reconstruction=MUSCL{MinMod}(), time_stepping=RK2())
    runtime = Runtime(iterations=iterations, write_interval=write_interval, time_step=1e-7,
        adaptive=LocalTimeStepping(maxCo=0.5))
    config = Configuration(solvers=(rho=(convergence=1e-15,), k=tight(), omega=tight()),
        schemes=schemes, runtime=runtime, hardware=Hardware(backend=CPU(), workgroup=64),
        boundaries=box_bcs(mesh))
    initialise!(model.momentum.U, [g_U, 0.0, 0.0])
    initialise!(model.momentum.p, g_p)
    initialise!(model.energy.T, g_T)
    initialise!(model.turbulence.k, g_k)
    initialise!(model.turbulence.omega, g_w)
    initialise!(model.turbulence.nut, g_k/g_w)
    model, config
end

# difference relative to the largest value of the whole serial field (or history)
relerr(a, b, ref=b) = maximum(abs.(a .- b); init=0.0) / max(maximum(abs, ref; init=0.0), eps())

# run `godunov_case(; kw...)` serially on rank 0 (global mesh `gmesh`) and on the distributed
# mesh `dm`, and test owned/ghost fields, residual and force histories against each other
function compare_godunov(name, kw, gmesh, dm, comm; tol=get(kw, :sst, false) ? 1e-7 : 1e-9)
    rank = MPI.Comm_rank(comm)
    forces = (patches=[:cylinder], interval=5)
    n = dm.partition.n_owned
    nloc = n + dm.partition.n_ghost
    orig = dm.orig_cells
    ref = if rank == 0
        ms, cs = godunov_case(gmesh; kw...)
        rs = run!(ms, cs; progress=false, forces)
        (map(f -> collect(f.values), godunov_fields(ms)), rs)
    else
        nothing
    end
    fields_s, res_s = MPI.bcast(ref, comm; root=0)

    model, config = godunov_case(dm; kw...)
    res = run!(model, config; progress=false, forces)
    fields_d = map(f -> Array(f.values), godunov_fields(model))

    owned = [relerr(fd[1:n], fs[orig[1:n]], fs) for (fd, fs) ∈ zip(fields_d, fields_s)]
    ghost = [relerr(fd[n+1:nloc], fs[orig[n+1:nloc]], fs) for (fd, fs) ∈ zip(fields_d, fields_s)]
    F_s = res_s.forces.cylinder
    F_d = res.forces.cylinder
    dF = maximum(relerr(F_d.pressure[i], F_s.pressure[i]) for i ∈ eachindex(F_s.pressure))

    @testset "$name (rank $rank)" begin
        @test all(isfinite, res.rho)
        @test maximum(owned) < tol
        @test maximum(ghost) < tol
        @test relerr(res.rho, res_s.rho) < tol
        @test relerr(res.rhoU, res_s.rhoU) < tol
        @test relerr(res.rhoE, res_s.rhoE) < tol
        @test F_d.iteration == F_s.iteration
        @test dF < tol
    end
    rank == 0 && println("godunov $name n=$(MPI.Comm_size(comm)): owned=$(maximum(owned)) ",
        "ghost=$(maximum(ghost)) rho_res=$(relerr(res.rho, res_s.rho)) force=$dF")
end
