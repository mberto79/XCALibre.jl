using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

# backwardFacingStep_2mm, backwardFacingStep_10mm

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "flatplate_2D_lowRe.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

workgroup = workgroupsize(mesh)
backend = CPU()
mesh_dev = adapt(backend, mesh)

Umag = 10
L = 1
velocity = [Umag, 0.0, 0.0]
nu = 1e-5
Re = velocity[1]*1/nu
k_inlet = 0.375
ω_inlet = 1000

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu=nu),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev # mesh_dev  # use mesh_dev for GPU backend
    )

BCs = assign(
    region=mesh,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Extrapolated(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Extrapolated(:top)            
        ],
        p = [
            Extrapolated(:inlet),
            Dirichlet(:outlet, 0.0),
            Extrapolated(:wall),
            Extrapolated(:top)
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Extrapolated(:outlet),
            Dirichlet(:wall, 0.0),
            Extrapolated(:top)
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Extrapolated(:outlet),
            OmegaWallFunction(:wall),
            Extrapolated(:top)
        ],
        nut = [
            Extrapolated(:inlet),
            Extrapolated(:outlet),
            Dirichlet(:wall, 0.0), 
            Extrapolated(:top)
        ]
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes(divergence=Upwind),
    k = Schemes(divergence=Upwind),
    omega = Schemes(divergence=Upwind)
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-1
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-2
    ),
    k = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-1
    ),
    omega = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-1
    )
)

runtime = Runtime(iterations=100, write_interval=100, time_step=1)

hardware = Hardware(backend=backend, workgroup=workgroup)
# hardware = Hardware(backend=CUDABackend(), workgroup=32)
# hardware = Hardware(backend=ROCBackend(), workgroup=32)

configure!(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

@test initialise!(model.momentum.U, velocity) === nothing
@test initialise!(model.momentum.p, 0.0) === nothing
@test initialise!(model.turbulence.k, k_inlet) === nothing
@test initialise!(model.turbulence.omega, ω_inlet) === nothing
@test initialise!(model.turbulence.nut, k_inlet/ω_inlet) === nothing

residuals = run!(model, config)

# test that we are close enough to drag coefficient
ReL = Umag*L/nu
Cd = 0.074/ReL^(1/5)
Cd_sim = viscous_force(:wall, model.momentum.U, 1, nu, model.turbulence.nut, BCs.U)/(0.5*Umag^2)

@test Cd ≈ Cd_sim[1] atol=0.065