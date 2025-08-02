using Plots
using XCALibre
using CUDA

# backwardFacingStep_2mm, backwardFacingStep_10mm
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

nu = 1e-3
u_mag = 1.5
velocity = [u_mag, 0.0, 0.0]
k_inlet = 1
ω_inlet = 1000
ω_wall = ω_inlet
Re = velocity[1]*0.1/nu

model = Physics(
    time = Transient(),
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
            Extrapolated(:outlet),
            Dirichlet(:outlet, 0.0),
            Wall(:wall),
            Wall(:top)
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Extrapolated(:outlet),
            KWallFunction(:wall),
            KWallFunction(:top)
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Extrapolated(:outlet),
            OmegaWallFunction(:wall),
            OmegaWallFunction(:top)
        ],
        nut = [
            Dirichlet(:inlet, k_inlet/ω_inlet),
            Extrapolated(:outlet),
            Dirichlet(:wall, 0.0), 
            Dirichlet(:top, 0.0)
        ]
    )
)

schemes = (
    U = Schemes(gradient=Midpoint, time=Euler),
    p = Schemes(gradient=Midpoint),
    k = Schemes(gradient=Midpoint, time=Euler),
    omega = Schemes(gradient=Midpoint, time=Euler)
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 1.0,
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 1.0,
    ),
    k = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 1.0,
    ),
    omega = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
    )
)

runtime = Runtime(
    iterations=1000, write_interval=10, time_step=0.01)

configure!(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config) # 36.90k allocs

Reff = stress_tensor(model.momentum.U, nu, model.turbulence.nut)
Fp = pressure_force(:wall, model.momentum.p, 1.25)
Fv = viscous_force(:wall, model.momentum.U, 1.25, nu, model.turbulence.nut)


# plot(; xlims=(0,494))
# plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
# plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
# plot!(1:length(Rp), Rp, yscale=:log10, label="p")