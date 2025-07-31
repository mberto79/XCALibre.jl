export get_boundaries
export _get_float, _get_int, _get_backend, _convert_array!
export bounding_box
export boundary_info, boundary_map
export total_boundary_faces, boundary_index
export norm_static
# export x, y, z # access cell centres
# export xf, yf, zf # access face centres

_get_int(mesh) = eltype(mesh.get_int)
_get_float(mesh) = eltype(mesh.get_float)
_get_backend(mesh) = get_backend(mesh.cells)

function _convert_array!(arr, backend::CPU)
    return arr
end

# Function to prevent redundant CPU copy
function get_boundaries(boundaries::Array)
    return boundaries
end

# Function to copy from GPU to CPU
function get_boundaries(boundaries::AbstractGPUArray)
    # Copy boundaries to CPU
    boundaries_cpu = Array{eltype(boundaries)}(undef, length(boundaries))
    copyto!(boundaries_cpu, boundaries)
    return boundaries_cpu
end

function bounding_box(mesh::AbstractMesh)
    (; faces, face_nodes, nodes) = mesh
    nbfaces = total_boundary_faces(mesh)

    backend = get_backend(faces)
    F = _get_float(mesh)

    pmin = KernelAbstractions.zeros(backend, F, 3)
    pmax = KernelAbstractions.zeros(backend, F, 3)

    ndrange = nbfaces
    workgroup = typeof(backend) <: CPU ? cld(nbfaces, Threads.nthreads()) : 32
    kernel! = _bounding_box(backend, workgroup, ndrange)
    kernel!(pmin, pmax, faces, face_nodes, nodes)
    return pmin, pmax
end

@kernel function _bounding_box(pmin, pmax, faces, face_nodes, nodes)
    fID = @index(Global)
    @inbounds face = faces[fID]
    (; nodes_range) = face

    @inbounds for nID ∈ @view face_nodes[nodes_range]
        node = nodes[nID]
        coords = node.coords
        pmin[1] = min(pmin[1], coords[1])
        pmin[2] = min(pmin[2], coords[2])
        pmin[3] = min(pmin[3], coords[3])
        pmax[1] = max(pmax[1], coords[1])
        pmax[2] = max(pmax[2], coords[2])
        pmax[3] = max(pmax[3], coords[3])
    end
end


# function total_boundary_faces(mesh::Mesh2{I,F}) where {I,F}
function total_boundary_faces(mesh::AbstractMesh)
    nbfaces = zero(_get_int(mesh))
    @inbounds for boundary ∈ get_boundaries(mesh.boundaries)
        nbfaces += length(boundary.IDs_range)
    end
    nbfaces
end

# Extract bundary index based on set name 
struct boundary_info{I<:Integer, S<:Symbol}
    ID::I
    Name::S
end
Adapt.@adapt_structure boundary_info

# Create LUT to map boudnary names to indices
function boundary_map(mesh)
    I = _get_int(mesh); S = Symbol
    boundary_map = boundary_info{I,S}[]

    mesh_temp = adapt(CPU(), mesh) # WARNING: Temp solution 

    for (i, boundary) in enumerate(mesh_temp.boundaries)
        push!(boundary_map, boundary_info{I,S}(i, boundary.name))
    end

    return boundary_map
end

function boundary_index(
    boundaries::Vector{boundary_info{TI, S}}, name::S
    ) where{TI<:Integer,S<:Symbol}
    for index in eachindex(boundaries)
        if boundaries[index].Name == name
            return boundaries[index].ID
        end
    end
end

function boundary_index(boundaries::Vector{Boundary{S, UR}}, name::S) where {S<:Symbol,UR}
    # bci = zero(TI)
    for index ∈ eachindex(boundaries)
        # bci += 1
        if boundaries[index].name == name
            return index 
        end
    end
end

# function x(mesh::Mesh2{I,F}) where {I,F}
#     cells = mesh.cells
#     out = zeros(F, length(cells))
#     @inbounds for i ∈ eachindex(cells)
#         out[i] = cells[i].centre[1]
#     end
#     return out
# end

# function y(mesh::Mesh2{I,F}) where {I,F}
#     cells = mesh.cells
#     out = zeros(F, length(cells))
#     @inbounds for i ∈ eachindex(cells)
#         out[i] = cells[i].centre[2]
#     end
#     return out
# end

# function z(mesh::Mesh2{I,F}) where {I,F}
#     cells = mesh.cells
#     out = zeros(F, length(cells))
#     @inbounds for i ∈ eachindex(cells)
#         out[i] = cells[i].centre[3]
#     end
#     return out
# end

# function xf(mesh::Mesh2{I,F}) where {I,F}
#     faces = mesh.faces
#     out = zeros(F, length(faces))
#     @inbounds for i ∈ eachindex(faces)
#         out[i] = faces[i].centre[1]
#     end
#     return out
# end

# function yf(mesh::Mesh2{I,F}) where {I,F}
#     faces = mesh.faces
#     out = zeros(F, length(faces))
#     @inbounds for i ∈ eachindex(faces)
#         out[i] = faces[i].centre[2]
#     end
#     return out
# end

# function zf(mesh::Mesh2{I,F}) where {I,F}
#     faces = mesh.faces
#     out = zeros(F, length(faces))
#     @inbounds for i ∈ eachindex(faces)
#         out[i] = faces[i].centre[3]
#     end
#     return out
# end

# Static normalise function
function norm_static(arr, p = 2)
    sum = 0
    for i in eachindex(arr)
        val = (abs(arr[i]))^p
        sum += val
    end
    return sum^(1/p)
end