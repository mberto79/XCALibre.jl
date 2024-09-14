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

function bounding_box(mesh::AbstractMesh)
    (; faces, nodes) = mesh
    nbfaces = total_boundary_faces(mesh)
    TF = _get_float(mesh)
    z = zero(TF)
    xmin, ymin, zmin = z, z, z
    xmax, ymax, zmax = z, z, z
    for fID ∈ 1:nbfaces
        face = faces[fID]
        for nID ∈ face.nodes_range 
            node = nodes[nID]
            coords = node.coords
            xmin = min(xmin, coords[1])
            ymin = min(ymin, coords[2])
            zmin = min(zmin, coords[3])
            xmax = max(xmax, coords[1])
            ymax = max(ymax, coords[2])
            zmax = max(zmax, coords[3])
        end
    end
    pmin = (xmin, ymin, zmin)
    pmax = (xmax, ymax, zmax)
    return pmin, pmax
end


# function total_boundary_faces(mesh::Mesh2{I,F}) where {I,F}
function total_boundary_faces(mesh::AbstractMesh)
    (; boundaries) = mesh
    nbfaces = zero(_get_int(mesh))
    @inbounds for boundary ∈ boundaries
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
    I = Integer; S = Symbol
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