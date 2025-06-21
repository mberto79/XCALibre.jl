using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

# backwardFacingStep_2mm, backwardFacingStep_10mm
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "flatplate_2D_highRe.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

workgroup = workgroupsize(mesh)
backend = CPU()
mesh_dev = adapt(backend, mesh)

# Inlet conditions
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
            Neumann(:outlet, 0.0),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Neumann(:top, 0.0)
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
            Neumann(:top, 0.0)
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Neumann(:outlet, 0.0),
            OmegaWallFunction(:wall),
            Neumann(:top, 0.0)
        ],
        nut = [
            Neumann(:inlet, 0.0),
            Neumann(:outlet, 0.0),
            NutWallFunction(:wall), 
            Neumann(:top, 0.0)
        ]
    )
)

solvers = (
    U = set_solver(
        region=mesh_dev,
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-1
    ),
    p = set_solver(
        region=mesh_dev,
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-2
    ),
    k = set_solver(
        region=mesh_dev,
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-1
    ),
    omega = set_solver(
        region=mesh_dev,
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-1
    )
)

runtime = set_runtime(iterations=100, write_interval=100, time_step=1)

hardware = set_hardware(backend=backend, workgroup=workgroup)
# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=ROCBackend(), workgroup=32)

for grad_limiter ∈ [nothing, FaceBased(model.domain), MFaceBased(model.domain)]
    
    local schemes = (
        U = set_schemes(divergence=Upwind, limiter=grad_limiter),
        p = set_schemes(divergence=Upwind, limiter=grad_limiter),
        k = set_schemes(divergence=Upwind),
        omega = set_schemes(divergence=Upwind)
    )

    local config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

    GC.gc()

    @test initialise!(model.momentum.U, velocity) === nothing
    @test initialise!(model.momentum.p, 0.0) === nothing
    @test initialise!(model.turbulence.k, k_inlet) === nothing
    @test initialise!(model.turbulence.omega, ω_inlet) === nothing
    @test initialise!(model.turbulence.nut, k_inlet/ω_inlet) === nothing

    local residuals = run!(model, config)

    local outlet = boundary_average(:outlet, model.momentum.U, BCs.U, config)

    @test Umag ≈ outlet[1] atol =0.1*Umag
end