# Smoke run writing <out>.res (residuals, field hashes) and <out>.time; usage in dev/scripts/INDEX.md.
const T0 = time()
using XCALibre
const T_LOAD = time() - T0
Base.cumulative_compile_timing(true)
mode, iterations, out = ARGS[1], parse(Int, ARGS[2]), ARGS[3]
mode == "gpu" && @eval using CUDA
mode == "mpi" && @eval using PETSc, MPI
mode in ("cpu", "2d", "mpi") && @eval using ThreadPinning
const POLYMESH = expanduser("~/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS/OpenFOAM/constant/polyMesh")
if mode == "part"
    partition_mesh(FOAM3D_mesh(POLYMESH, scale=1, integer_type=Int32), iterations; dir=out)
    exit()
end

function motorbike_case(mesh_dev, hardware)
    velocity = [20.0, 0.0, 0.0]; noSlip = [0.0, 0.0, 0.0]
    k_in = 0.24; w_in = 1.78; nut_in = k_in/w_in
    model = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=1.5e-5),
        turbulence=RANS{KOmega}(), energy=Energy{Isothermal}(), domain=mesh_dev)
    side = (Slip(:upperWall), Slip(:frontAndBack))
    BCs = assign(region=mesh_dev, (
        U = [Dirichlet(:inlet, velocity), Zerogradient(:outlet), Wall(:lowerWall, velocity), Wall(:motorBike, noSlip), side...],
        p = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0), Wall(:lowerWall), Wall(:motorBike), side...],
        k = [Dirichlet(:inlet, k_in), Zerogradient(:outlet), KWallFunction(:lowerWall), KWallFunction(:motorBike), side...],
        omega = [Dirichlet(:inlet, w_in), Zerogradient(:outlet), OmegaWallFunction(:lowerWall), OmegaWallFunction(:motorBike), side...],
        nut = [Dirichlet(:inlet, nut_in), Zerogradient(:outlet), NutWallFunction(:lowerWall), NutWallFunction(:motorBike), side...]))
    ss(s, r) = SolverSetup(solver=s, preconditioner=Jacobi(), convergence=1e-14, relax=r, rtol=0.1, itmax=1000)
    # PSOLVER=amg selects the benchmark's AMG pressure solver
    ps = get(ENV, "PSOLVER", "cg") == "amg" ?
        AMG(mode=Cg(), coarsening=Geometric(), smoother=AMGJacobi(), fuse_levels=0) : Cg()
    solvers = (U=ss(Bicgstab(), 0.7), p=ss(ps, 0.3), k=ss(Bicgstab(), 0.3), omega=ss(Bicgstab(), 0.3))
    schemes = (U=Schemes(time=SteadyState, divergence=LUST, gradient=Gauss), p=Schemes(time=SteadyState, gradient=Gauss),
        k=Schemes(time=SteadyState, divergence=Upwind, gradient=Gauss), omega=Schemes(time=SteadyState, divergence=Upwind, gradient=Gauss))
    init!() = (initialise!(model.momentum.U, velocity); initialise!(model.momentum.p, 0.0);
        initialise!(model.turbulence.k, k_in); initialise!(model.turbulence.omega, w_in); initialise!(model.turbulence.nut, nut_in))
    config(n) = Configuration(solvers=solvers, schemes=schemes, hardware=hardware, boundaries=BCs,
        runtime=Runtime(iterations=n, write_interval=-1, time_step=1))
    model, init!, config, (m, c) -> potential_flow!(m, c; ncorrectors=10)
end

function bfs2d_case(mesh_dev, hardware)
    velocity = [69.4, 0.0, 0.0]; k_in = 1.0; w_in = 1000.0
    model = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=1e-3),
        turbulence=RANS{KOmegaSST}(walls=(:wall, :top)), energy=Energy{Isothermal}(), domain=mesh_dev)
    BCs = assign(region=mesh_dev, (
        U = [Dirichlet(:inlet, velocity), Zerogradient(:outlet), Wall(:wall, [0.0, 0.0, 0.0]), Wall(:top, [0.0, 0.0, 0.0])],
        p = [Neumann(:inlet, 0.0), Dirichlet(:outlet, 0.0), Wall(:wall), Wall(:top)],
        k = [Dirichlet(:inlet, k_in), Zerogradient(:outlet), KWallFunction(:wall), KWallFunction(:top)],
        omega = [Dirichlet(:inlet, w_in), Zerogradient(:outlet), OmegaWallFunction(:wall), OmegaWallFunction(:top)],
        nut = [Dirichlet(:inlet, k_in/w_in), Zerogradient(:outlet), NutWallFunction(:wall), NutWallFunction(:top)]))
    ss(s, r) = SolverSetup(solver=s, preconditioner=Jacobi(), convergence=1e-12, relax=r, rtol=1e-2, atol=1e-10)
    solvers = (U=ss(Bicgstab(), 0.7), p=ss(Cg(), 0.3), k=ss(Bicgstab(), 0.7), omega=ss(Bicgstab(), 0.7), y=ss(Cg(), 0.9))
    schemes = (U=Schemes(divergence=LUST, gradient=Gauss), p=Schemes(), k=Schemes(divergence=Upwind),
        omega=Schemes(divergence=Upwind), y=Schemes(gradient=Midpoint))
    init!() = (initialise!(model.momentum.U, velocity); initialise!(model.momentum.p, 0.0);
        initialise!(model.turbulence.k, k_in); initialise!(model.turbulence.omega, w_in); initialise!(model.turbulence.nut, k_in/w_in))
    config(n) = Configuration(solvers=solvers, schemes=schemes, hardware=hardware, boundaries=BCs,
        runtime=Runtime(iterations=n, write_interval=-1, time_step=1))
    model, init!, config, (m, c) -> nothing
end

t_mesh = @elapsed mesh = if mode == "2d"
    UNV2D_mesh(pkgdir(XCALibre, "examples/0_GRIDS/backwardFacingStep_10mm.unv"), scale=0.001, integer_type=Int64)
elseif mode == "mpi"
    MPI.Initialized() || MPI.Init()
    distribute(ARGS[4])
else
    FOAM3D_mesh(POLYMESH, scale=1, integer_type=Int32)
end

backend = mode == "gpu" ? CUDABackend() : CPU(static=true)
workgroup = mode == "gpu" ? 32 : AutoTune()
mode == "gpu" || activate_multithread(backend)
mode in ("cpu", "2d") && pinthreads(:cores)
mode == "mpi" && mpi_pinthreads(:cores)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = mode == "gpu" ? adapt(backend, mesh) : mesh
model, init!, config, pre! = (mode == "2d" ? bfs2d_case : motorbike_case)(mesh_dev, hardware)

c0 = Base.cumulative_compile_time_ns()[1]
t_pre = @elapsed (init!(); pre!(model, config(1)))
t_first = @elapsed run!(model, config(1))
comp_s = (Base.cumulative_compile_time_ns()[1] - c0)/1e9
GC.gc(true)
init!(); pre!(model, config(iterations))
t_run = @elapsed residuals = run!(model, config(iterations); progress=get(ENV, "PROGRESS", "true") == "true",
    (mode == "mpi" ? (; petsc_options=get(ENV, "PETSC_OPTS", "")) : (;))...)

if mode != "mpi" || is_root()
    open(out * ".res", "w") do io
        for k in sort(collect(keys(residuals)))
            println(io, k, " ", join(repr.(residuals[k]), " "))
        end
        U = model.momentum.U; tu = model.turbulence
        for (n, f) in (("U.x", U.x), ("U.y", U.y), ("U.z", U.z), ("p", model.momentum.p), ("k", tu.k), ("omega", tu.omega), ("nut", tu.nut))
            println(io, "hash ", n, " ", repr(hash(Array(f.values))))
        end
    end
    open(out * ".time", "w") do io
        println(io, "mode=$mode threads=$(Threads.nthreads()) iterations=$iterations mesh_s=$t_mesh first_run_s=$t_first run_s=$t_run load_s=$T_LOAD pre_s=$t_pre compile_s=$comp_s faces=$(nameof(typeof(XCALibre.Solvers._base_mesh(mesh).faces)))")
    end
end
