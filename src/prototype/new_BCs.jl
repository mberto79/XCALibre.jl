
# Step 0. Load libraries
using XCALibre
using Adapt
using CUDA
# using CUDA # Uncomment to run on NVIDIA GPUs
# using AMDGPU # Uncomment to run on AMD GPUs


# Step 1. Define mesh

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_10mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001) # convert mesh

# Step 2. Select backend and setup hardware
backend = CPU()
# backend = CUDABackend() # ru non NVIDIA GPUs
# backend = ROCBackend() # run on AMD GPUs

hardware = Hardware(backend=backend, workgroup=4)
# hardware = Hardware(backend=backend, workgroup=32) # use for GPU backends

mesh_dev = mesh # use this line to run on CPU
# mesh_dev = adapt(backend, mesh)  # Uncomment to run on GPU 

# Step 3. Flow conditions
velocity = [1.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

# Step 4. Define physics
model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

# Step 5. Define boundary conditions
@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Wall(:top, [0.0, 0.0, 0.0]),
)

abstract type AbstractBoundary end


struct FixedValue{S,V,I} <: AbstractBoundary
    name::S
    value::V
    ID::I 
    IDs_range::R
end
Adapt.Adapt.@adapt_structure FixedValue

struct FixedGradient{S,V,I} <: AbstractBoundary
    name::S
    value::V
    ID::I 
    IDs_range::R
end
Adapt.Adapt.@adapt_structure FixedGradient

(::Type{T})(name::Symbol, value) where T<:AbstractBoundary = T(name,value,0,0:0)
# T.name.wrapper(name, value, 0)
t0 = FixedValue(:inlet, 1.2)

t1 = FixedValue(:inlet, 1.2)
t2 = FixedGradient(:outlet, [1,2,3])

t1_gpu = adapt(CUDABackend(), t1)
t2_gpu = adapt(CUDABackend(), t2)

function match_ID(BC, mesh)
    (; boundaries) = mesh # needs to be a copy
    intType = _get_int(mesh)
    for (ID, boundary) ∈ enumerate(boundaries)
        if BC.name == boundary.name
            return intType(ID), boundary.IDs_range
        end
    end
    error(""""$(BC.name)" is not a recognised boundary name""")
end

using StaticArrays
using Accessors

adapt_value(value::Number, mesh) = _get_float(mesh)(value)
adapt_value(value::Vector, mesh) = begin
    F = _get_float(mesh)
    @assert length(value) == 3 "Vectors must have 3 components"
    SVector{3,F}(value)
end
adapt_value(value::Function, mesh) = value

function finalise2!(BCs, region)
    newBCs = []
    for (i, BC) ∈ enumerate(BCs)
        ID, IDs_range = match_ID(BC, region)
        value = adapt_value(BC.value, region)
        push!(newBCs, typeof(BC).name.wrapper(BC.name, value, ID, IDs_range))
    end
    # Tuple(newBCs)
    newBCs
end

function assign2(args; region)
    BCs = []
    names = propertynames(args)
    for arg ∈ args
        updatedBCs = finalise2!(arg, region)
        push!(BCs, updatedBCs)
    end
    assignedBCs = NamedTuple{names}(Tuple.(BCs))
    nboundaries = length(region.boundaries)
    for (name, assignedBC) ∈ zip(names, assignedBCs)
        @assert length(assignedBC) == nboundaries "Inconsistent number of boundaries assigned to field $name"
    end
    return assignedBCs
end

@time BCs = assign2(
    region=mesh_dev,
    (
        U = [
            FixedValue(:outlet, [1,12.2,50]),
            FixedGradient(:inlet, 0.0),
            FixedGradient(:wall, 0.0),
            FixedValue(:top, 0.0)
        ],
        p = [
            FixedValue(:outlet, sin),
            FixedGradient(:inlet, 0.0),
            FixedGradient(:wall, 0.0),
            FixedValue(:top, 0.0)
        ],
    )
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

# Step 6. Choose discretisation schemes
schemes = (
    U = Schemes(divergence = Linear),
    p = Schemes() # no input provided (will use defaults)
)

# Step 7. Set up linear solvers and preconditioners
solvers = (
    U = SolverSetup(
        model.momentum.U;
        solver      = BicgstabSolver, # Options: GmresSolver
        preconditioner = Jacobi(), # Options: NormDiagonal()
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    ),
    p = SolverSetup(
        model.momentum.p;
        solver      = CgSolver, # Options: CgSolver, BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), # Options: NormDiagonal()
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    )
)

# Step 8. Specify runtime requirements
runtime = Runtime(iterations=2000, time_step=1, write_interval=2000)
runtime = Runtime(iterations=1, time_step=1, write_interval=-1) # hide

# Step 9. Construct Configuration object
configure!(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

# Step 10. Initialise fields (initial guess)
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

# Step 11. Run simulation
residuals = run!(model);

# Step 12. Post-process
pwd() # find active directory where the file "iteration_002000.vtk" was saved