using XCALibre
# using CUDA

# backwardFacingStep_2mm, 5mm or 10mm
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = set_hardware(backend=backend, workgroup=workgroup)
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

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0),
    Dirichlet(:top, 0.0)
)

@assign! model turbulence omega (
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    OmegaWallFunction(:wall),
    OmegaWallFunction(:top)
)

@assign! model turbulence nut (
    Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, 0.0), 
    Dirichlet(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence=LUST, gradient=Midpoint),
    p = set_schemes(gradient=Midpoint),
    k = set_schemes(gradient=Midpoint),
    omega = set_schemes(gradient=Midpoint)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-2,
        atol = 1e-10
    ),
    p = set_solver(
        model.momentum.p;
        solver      = Cg(), #Gmres(), #Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-3,
        atol = 1e-10
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-2,
        atol = 1e-10
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-2,
        atol = 1e-10
    )
)

runtime = set_runtime(iterations=2000, write_interval=1000, time_step=1)
# runtime = set_runtime(iterations=1, write_interval=-1, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)


GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config) # 36.90k allocs

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