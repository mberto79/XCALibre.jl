using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "flatplate_2D_laminar.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

workgroup = 1023 #workgroupsize(mesh)
backend = CPU()
mesh_dev = adapt(backend, mesh)

# Inlet conditions
Umag = 10
velocity = [Umag, 0.0, 0.0]
nu = 1e-4
Re = velocity[1]*1/nu
cp = 1005.0
gamma = 1.4
Pr = 0.7

k_inlet = 0.375
ω_inlet = 1000

model = Physics(
    time = Steady(),
    fluid = Fluid{WeaklyCompressible}(nu=nu, cp=cp, gamma=gamma, Pr=Pr),
    turbulence = RANS{KOmega}(),
    energy = Energy{SensibleEnthalpy}(Tref=288.15),
    domain = mesh_dev # mesh_dev  # use mesh_dev for GPU backend
    )

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            # Neumann(:outlet, 0.0),
            # Extrapolated(:outlet),
            Zerogradient(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            # Symmetry(:top)
            Extrapolated(:top)            
        ],
        p = [
            # Neumann(:inlet, 0.0),
            Zerogradient(:inlet),
            # Extrapolated(:inlet),
            Dirichlet(:outlet, 100000.0),
            # Neumann(:wall, 0.0),
            Wall(:wall),
            # Neumann(:top, 0.0)
            # Extrapolated(:top)
            # Zerogradient(:top)
            # Symmetry(:top)
            Extrapolated(:top)            
        ],
        h = [
            FixedTemperature(:inlet, T=300.0, model=model.energy),
            # Neumann(:outlet, 0.0),
            # Extrapolated(:outlet),
            Zerogradient(:outlet),
            FixedTemperature(:wall, T=310.0, model=model.energy),
            # Neumann(:top, 0.0)
            # Extrapolated(:top)
            # Symmetry(:top)
            Extrapolated(:top)            
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            # Neumann(:outlet, 0.0),
            Extrapolated(:outlet),
            Dirichlet(:wall, 0.0),
            # Neumann(:top, 0.0)
            # Extrapolated(:top)
            # Symmetry(:top)
            Extrapolated(:top)            
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            # Neumann(:outlet, 0.0),
            Extrapolated(:outlet),
            OmegaWallFunction(:wall),
            # Neumann(:top, 0.0)
            # Extrapolated(:top)
            # Symmetry(:top)
            Extrapolated(:top)            
        ],
        nut = [
            Dirichlet(:wall, 0.0), 
            # Neumann.([:inlet, :outlet, :top], Ref(0.0))...
            Extrapolated.([:inlet, :outlet, :top])...
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
    preconditioner = DILU(), #Jacobi(),
    convergence = 1e-7,
    relax       = 0.2,
    rtol = 1e-2
),
h = set_solver(
    region=mesh_dev,
    solver      = Bicgstab(), # Bicgstab(), Gmres()
    preconditioner = DILU(),
    convergence = 1e-7,
    relax       = 0.7,
    rtol = 1e-1,
),
k = set_solver(
    region=mesh_dev,
    solver      = Bicgstab(), # Bicgstab(), Gmres()
    preconditioner = Jacobi(), 
    convergence = 1e-7,
    relax       = 0.7,
    rtol = 1e-1
),
omega = set_solver(
    region=mesh_dev,
    solver      = Bicgstab(), # Bicgstab(), Gmres()
    preconditioner = Jacobi(),
    convergence = 1e-7,
    relax       = 0.7,
    rtol = 1e-1
)
)

runtime = set_runtime(iterations=100, write_interval=100, time_step=1)

hardware = set_hardware(backend=backend, workgroup=workgroup)

# for grad_limiter ∈ [nothing]  #FaceBased(model.domain), MFaceBased(model.domain)]
    grad_limiter = nothing
    schemes = (
        U = set_schemes(divergence=Upwind, limiter=grad_limiter),
        p = set_schemes(divergence=Upwind, limiter=grad_limiter),
        h = set_schemes(divergence=Upwind),
        k = set_schemes(divergence=Upwind),
        omega = set_schemes(divergence=Upwind)
    )
    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

    GC.gc()

    @test initialise!(model.momentum.U, velocity) === nothing
    @test initialise!(model.momentum.p, 100000.0) === nothing
    @test initialise!(model.energy.T, 300.0) === nothing
    @test initialise!(model.turbulence.k, k_inlet) === nothing
    @test initialise!(model.turbulence.omega, ω_inlet) === nothing
    @test initialise!(model.turbulence.nut, k_inlet/ω_inlet) === nothing

    residuals = run!(model, config)

    inlet = boundary_average(:inlet, model.momentum.U, BCs.U, config)
    outlet = boundary_average(:outlet, model.momentum.U, BCs.U, config)
    top = boundary_average(:top, model.momentum.U, BCs.U, config)

    BCs.U, @test Umag ≈ inlet[1]
    @test Umag ≈ outlet[1] atol = 0.95
    @test Umag ≈ top[1] atol = 0.15
# end