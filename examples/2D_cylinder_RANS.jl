using Plots
using XCALibre
# using CUDA


grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# INLET CONDITIONS 

Umag = 2.5
velocity = [Umag, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-5
νR = 20
Tu = 0.05
k_inlet = 3/2*(Tu*Umag)^2
ω_inlet = k_inlet/(νR*nu)
Re = (0.2*velocity[1])/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    # turbulence = RANS{KOmega}(β⁺=0.09),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region = mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Extrapolated(:top),
            Extrapolated(:bottom),
            Wall(:cylinder, noSlip)
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Zerogradient(:top),
            Zerogradient(:bottom),
            Wall(:cylinder)
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Extrapolated(:outlet),
            Extrapolated(:top),
            Extrapolated(:bottom),
            KWallFunction(:cylinder)
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Extrapolated(:outlet),
            Extrapolated(:top),
            Extrapolated(:bottom),
            OmegaWallFunction(:cylinder) # need constructor to force keywords
        ],
        nut = [
            Dirichlet(:inlet, k_inlet/ω_inlet),
            Extrapolated(:outlet),
            Extrapolated(:top),
            Extrapolated(:bottom), 
            NutWallFunction(:cylinder)
        ]
    )
)

limiter = MFaceBased(model.domain)
schemes = (
    U = Schemes(divergence=Upwind, gradient=Gauss, limiter=limiter),
    p = Schemes(gradient=Gauss),
    k = Schemes(divergence=Upwind, gradient=Gauss),
    omega = Schemes(divergence=Upwind, gradient=Gauss)
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        atol = 0.1
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.15,
        atol = 0.01
    ),
    k = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.6,
        atol = 0.1
    ),
    omega = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres(), Cg()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.6,
        atol = 0.1
    )
)

runtime = Runtime(iterations=1000, write_interval=100, time_step=1)

configure!(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config, ncorrectors=1); #, pref=0.0)

Reff = stress_tensor(model.momentum.U, nu, model.turbulence.nut)
Fp = pressure_force(:cylinder, model.momentum.p, 1.25)
Fv = viscous_force(:cylinder, model.momentum.U, 1.25, nu, model.turbulence.nut)

# plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
# plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
# plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
# plot!(1:length(Rp), Rp, yscale=:log10, label="p")