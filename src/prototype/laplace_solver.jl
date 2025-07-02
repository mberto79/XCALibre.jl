using Revise
# using Adapt
using XCALibre
# using .ModelPhysics: Solid, Uniform

# include("../ModelPhysics/2_medium_models.jl")     # XCALibre core types  
# include("../ModelPhysics/0_type_definition.jl")     # XCALibre core types  
# include("../ModelPhysics/ModelPhysics.jl")     # XCALibre core types  

# using XCALibre: Solid, Uniform   # pull the names into Main




using CUDA

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
# grid = "backwardFacingStep_10mm.unv"
# grid = "summer_2d_5x10.unv"
grid = "summer_3d_extruded_pipe.unv"
mesh_file = joinpath(grids_dir, grid)

# mesh = UNV2D_mesh(mesh_file, scale=0.001)
mesh = UNV3D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)

mesh_dev = adapt(backend, mesh)


model = Physics(
    time = Steady(),
    medium = Solid{Uniform}(k = 16.2),
    # fluid = Fluid{Incompressible}(nu = nu), #medium = Solid{Uniform}(k = 1.0),
    # turbulence = RANS{Laminar}(), #No turbulence
    # turbulence = noTurbulence(),
    # turbulence = RANS{NoTurbulence}(), #NEED TO REPLACE WITH 'NoTurbulence,'
    # energy = Energy{Isothermal}(),
    energy = Energy{LaplaceEnergy}(),
    domain = mesh_dev
    )

BCs = assign(
    region = mesh_dev,
    (
        T = [     
            # FixedTemperature(:inlet; T = 300.0),                               
            # FixedTemperature(
            #     :inlet,
            #     Enthalpy(cp = 490, Tref = 288.15); 
            #     T = 300.0                         
            # ),
            Dirichlet(:inlet, 500),
            Zerogradient(:outlet),    
            Zerogradient(:walls)       
            
            # FixedTemperature(
            #     :walls,
            #     Enthalpy(cp = 490, Tref = 288.15); 
            #     T = 50.0                         
            # ),
        ],
    )
)


solvers = (
    T = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), # Jacobi(), #NormDiagonal(), # DILU()
        convergence = 1e-8,
        relax       = 0.8,
        rtol = 1e-4,
        atol = 1e-5
    )
)

schemes = (
    T = Schemes(laplacian = Linear)
)


runtime = Runtime(
    iterations=10, write_interval=1, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

# T_field = ScalarField(mesh_dev)
initialise!(model.energy.T, 10.0)

residuals = run!(model, config)