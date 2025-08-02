using XCALibre
# using CUDA

# backwardFacingStep_2mm, 5mm or 10mm
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_10mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

nu = 1e-3
# u_mag = 3.5 # 2mm mesh
u_mag = 1.5 # 5mm mesh
velocity = [u_mag, 0.0, 0.0]
k_inlet = 1
ω_inlet = 1000
ω_wall = ω_inlet
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region = mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Extrapolated(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Wall(:top, [0.0, 0.0, 0.0])
        ],
        p = [
            Neumann(:inlet, 0.0),
            Dirichlet(:outlet, 0.0),
            Neumann(:wall, 0.0),
            Neumann(:top, 0.0)
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Neumann(:outlet, 0.0),
            KWallFunction(:wall),
            KWallFunction(:top)
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Neumann(:outlet, 0.0),
            OmegaWallFunction(:wall),
            OmegaWallFunction(:top)
        ],
        nut = [
            Dirichlet(:inlet, k_inlet/ω_inlet),
            Neumann(:outlet, 0.0),
            Dirichlet(:wall, 0.0), 
            Dirichlet(:top, 0.0)
        ]
    )
)

schemes = (
    U = Schemes(divergence=LUST, gradient=Midpoint),
    p = Schemes(gradient=Midpoint),
    k = Schemes(gradient=Midpoint),
    omega = Schemes(gradient=Midpoint)
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-2,
        atol = 1e-10
    ),
    p = SolverSetup(
        solver      = Cg(), #Gmres(), #Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-3,
        atol = 1e-10
    ),
    k = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-2,
        atol = 1e-10
    ),
    omega = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-2,
        atol = 1e-10
    )
)

runtime = Runtime(iterations=2000, write_interval=1000, time_step=1)
# runtime = Runtime(iterations=1, write_interval=-1, time_step=1)

configure!(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)
configure!(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model) # 36.90k allocs

# # Reff = stress_tensor(model.momentum.U, nu, model.turbulence.nut, config)
# Fp = pressure_force(:wall, model.momentum.p, 1.25)
# Fv = viscous_force(:wall, model.momentum.U, 1.25, nu, model.turbulence.nut)
# ave = boundary_average(:inlet, model.momentum.U, config)
# ave = boundary_average(:outlet, model.momentum.U, config)

# using Plots
# plot(; ylims=(1e-8,1), xlims=(1,500))
# plot!(1:length(residuals.Ux), residuals.Ux, yscale=:log10, label="Ux")
# plot!(1:length(residuals.Uy), residuals.Uy, yscale=:log10, label="Uy")
# plot!(1:length(residuals.p), residuals.p, yscale=:log10, label="p")