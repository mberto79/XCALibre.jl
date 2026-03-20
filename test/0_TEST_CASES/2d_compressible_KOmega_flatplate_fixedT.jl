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
            # Extrapolated(:outlet),
            Zerogradient(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            # Symmetry(:top)
            Extrapolated(:top)            
        ],
        p = [
            Zerogradient(:inlet),
            # Extrapolated(:inlet),
            Dirichlet(:outlet, 100000.0),
            Wall(:wall),
            # Zerogradient(:top)
            # Symmetry(:top)
            Extrapolated(:top)            
        ],
        h = [
            FixedTemperature(:inlet, T=300.0, Enthalpy(cp=cp, Tref=288.15)),
            # Extrapolated(:outlet),
            Zerogradient(:outlet),
            FixedTemperature(:wall, T=310.0, Enthalpy(cp=cp, Tref=288.15)),
            # Extrapolated(:top)
            # Symmetry(:top)
            Extrapolated(:top)            
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Extrapolated(:outlet),
            Dirichlet(:wall, 0.0),
            # Extrapolated(:top)
            # Symmetry(:top)
            Extrapolated(:top)            
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Extrapolated(:outlet),
            OmegaWallFunction(:wall),
            # Extrapolated(:top)
            # Symmetry(:top)
            Extrapolated(:top)            
        ],
        nut = [
            Dirichlet(:wall, 0.0), 
            Extrapolated.([:inlet, :outlet, :top])...
        ]
    )
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
    preconditioner = DILU(), #Jacobi(),
    convergence = 1e-7,
    relax       = 0.2,
    rtol = 1e-2
),
h = SolverSetup(
    solver      = Bicgstab(), # Bicgstab(), Gmres()
    preconditioner = DILU(),
    convergence = 1e-7,
    relax       = 0.7,
    rtol = 1e-1,
),
k = SolverSetup(
    solver      = Bicgstab(), # Bicgstab(), Gmres()
    preconditioner = Jacobi(), 
    convergence = 1e-7,
    relax       = 0.7,
    rtol = 1e-1
),
omega = SolverSetup(
    solver      = Bicgstab(), # Bicgstab(), Gmres()
    preconditioner = Jacobi(),
    convergence = 1e-7,
    relax       = 0.7,
    rtol = 1e-1
)
)

runtime = Runtime(iterations=100, write_interval=100, time_step=1)

hardware = Hardware(backend=backend, workgroup=workgroup)

# for grad_limiter ∈ [nothing]  #FaceBased(model.domain), MFaceBased(model.domain)]
    grad_limiter = nothing
    schemes = (
        U = Schemes(divergence=Upwind, limiter=grad_limiter),
        p = Schemes(divergence=Upwind, limiter=grad_limiter),
        h = Schemes(divergence=Upwind),
        k = Schemes(divergence=Upwind),
        omega = Schemes(divergence=Upwind)
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