using Plots
using FVM_1D
using Krylov
using CUDA
using KernelAbstractions

const BACKEND = :CUDA 

## FUNCTION TO BE ALTERED

# @kernel function mandelbrot_AB(z, c, iteration)
#     i,j = @index(Global, NTuple)
    
#     for k in 1:iteration
#         @inbounds z[i,j] = z[i,j] * z[i,j] + c[i,j]

#         if real(z[i,j]) > 10e7 || imag(z[i,j]) > 10e7
#             @inbounds z[i,j] = 1
#             break
#         end
        
#     end

#     if real(z[i,j]) != 1
#         z[i,j] = 0mesh_gpu = cu(mesh)
#     end

# end

# const MyDevice = CUDADevice
# kernel = mandelbrot_AB(MyDevice())

# kernel(z, c, iteration, ndrange=size(z))



function boundary_index_mapping(
    boundaries::Vector{Boundary{Symbol, Vector{TI}}},
    symbol_mapping::Dict{Symbol,Int},
    name_index::Int
    ) where TI<:Integer
    bci = zero(TI)

    for i ∈ eachindex(boundaries)
        bci += one(TI)
        
        if get(symbol_mapping, boundaries[i].name, nothing) == name_index
            return bci 
        end

    end
end

function boundary_index(
    boundaries::Vector{Boundary{Symbol, Vector{TI}}}, name::Symbol
    ) where TI<:Integer
    bci = zero(TI)
    for i ∈ eachindex(boundaries)
        bci += one(TI)
        if boundaries[i].name == name
            return bci 
        end
    end
end

# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)
mesh_gpu = cu(mesh)
symbol_mapping = number_symbols(mesh)

## Dictionary definition
function number_symbols(mesh)
    symbol_mapping = Dict{Symbol, Int}()

    for (i, boundary) in enumerate(mesh.boundaries)
        if haskey(symbol_mapping, boundary.name)
            # Do nothing, the symbol is already mapped
        else
            new_number = length(symbol_mapping) + 1
            symbol_mapping[boundary.name] = new_number
        end
    end
    
    return symbol_mapping
end
symbol_mapping = number_symbols(mesh)

# Inlet conditions

velocity = [0.50, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

# Manual assignment of boundaries
BCs = ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

# CPU test case
for arg in BCs
    idx = boundary_index(mesh.boundaries,arg.ID)
    println("$idx")
end
    
# CPU index case
    
get(symbol_mapping,boundary1.ID,nothing)

for arg in BCs
    idx = boundary_index_mapping(mesh.boundaries,symbol_mapping,get(symbol_mapping,arg.ID,nothing))
    println("$idx")
end

# GPU test case
BCs_gpu = cu(( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
))

for arg in BCs_gpu
    mesh_boundary_index = get(symbol_mapping,arg.ID,nothing)
    println("$mesh_boundary_index")
end

mesh_boundary_index_gpu = cu(mesh_boundary_index)

const MyDevice = BACKEND
kernel = boundary_index_gpu(MyDevice())
bci = zero(Integer)
kernel(mesh_gpu.boudnaries, BCs_gpu[1].ID, bci)
bci

@kernel function boundary_index_mapping(
    boundaries::Vector{Boundary{Symbol, Vector{TI}}},
    name_index::Int,
    bci::Int,
    ) where TI<:Integer
    
    bci_temp = zero(TI)

    for i ∈ eachindex(boundaries)
        bci_temp += one(TI)
        
        if get(symbol_mapping, boundaries[i].name, nothing) == name_index
            bci = bci_temp
        end

    end
end
