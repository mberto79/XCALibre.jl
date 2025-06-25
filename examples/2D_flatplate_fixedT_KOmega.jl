using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "flatplate_2D_laminar.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

velocity = [10, 0.0, 0.0]
nu = 1e-4
Re = velocity[1]*1/nu
cp = 1005.0
gamma = 1.4
Pr = 0.7

k_inlet = 0.375
ω_inlet = 1000

model = Physics(
    time = Steady(),
    fluid = Fluid{WeaklyCompressible}(
        nu = nu,
        cp = cp,
        gamma = gamma,
        Pr = Pr
        ),
    turbulence = RANS{KOmega}(),
    energy = Energy{SensibleEnthalpy}(Tref=288.15),
    domain = mesh_dev
    )

BCs = assign(
    region = mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Extrapolated(:top)
            # Symmetry(:top, 0.0)
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 100000.0),
            Wall(:wall),
            Extrapolated(:top)
        ],
        h = [
            FixedTemperature(:inlet, T=300.0, Enthalpy(cp=cp, Tref=288.15)),
            Zerogradient(:outlet),
            FixedTemperature(:wall, T=310.0, Enthalpy(cp=cp, Tref=288.15)),
            Extrapolated(:top)
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Zerogradient(:outlet),
            Dirichlet(:wall, 0.0),
            Extrapolated(:top)
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Zerogradient(:outlet),
            OmegaWallFunction(:wall),
            Extrapolated(:top)
        ],
        nut = [
            Dirichlet(:inlet, k_inlet/ω_inlet),
            Extrapolated(:outlet),
            Dirichlet(:wall, 0.0), 
            Extrapolated(:top)
        ]
    )
)

schemes = (
    U = Schemes(divergence=Linear),
    p = Schemes(divergence=Linear),
    h = Schemes(divergence=Linear),
    k = Schemes(divergence=Upwind),
    omega = Schemes(divergence=Upwind)
)


solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        atol = 1e-1
    ),
    p = SolverSetup(
        solver      = Gmres(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        atol = 1e-2
    ),
    h = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-2,
        atol = 1e-1
    ),
    k = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.6,
        atol = 1e-1
    ),
    omega = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.6,
        atol = 1e-1
    )
)

runtime = Runtime(iterations=1000, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 100000.0)
initialise!(model.energy.T, 300.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config) # 9.39k allocs

using Plots
plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
plot!(1:length(Re), Re, yscale=:log10, label="h")