export _get_float, _get_int, _get_backend, _convert_array!
export total_boundary_faces, boundary_index
export norm_static
# export number_symbols
export x, y, z # access cell centres
export xf, yf, zf # access face centres

_get_int(mesh) = eltype(mesh.get_int)
_get_float(mesh) = eltype(mesh.get_float)
# _get_backend(mesh) = get_backend(mesh.cells)

# function _convert_array!(arr, backend::CPU)
#     return arr
# end
# function _convert_array!(arr, backend::CUDABackend)
#     return adapt(CuArray, arr)
# end

# function total_boundary_faces(mesh::Mesh2{I,F}) where {I,F}
function total_boundary_faces(mesh::AbstractMesh)
    (; boundaries) = mesh
    # nbfaces = zero(I)
    nbfaces = zero(_get_int(mesh))
    @inbounds for boundary ∈ boundaries
        nbfaces += length(boundary.IDs_range)
    end
    nbfaces
end

function boundary_index(
    boundaries, name
    )
    for i in eachindex(boundaries)
        if boundaries[i].Name == name
            return boundaries[i].ID
        end
    end
end

function x(mesh::AbstractMesh)
    cells = mesh.cells
    out = zeros(_get_float(mesh), length(cells))
    @inbounds for i ∈ eachindex(cells)
        out[i] = cells[i].centre[1]
    end
    return out
end

function y(mesh::AbstractMesh)
    cells = mesh.cells
    out = zeros(_get_float(mesh), length(cells))
    @inbounds for i ∈ eachindex(cells)
        out[i] = cells[i].centre[2]
    end
    return out
end

function z(mesh::AbstractMesh)
    cells = mesh.cells
    out = zeros(_get_float(mesh), length(cells))
    @inbounds for i ∈ eachindex(cells)
        out[i] = cells[i].centre[3]
    end
    return out
end

function xf(mesh::AbstractMesh) 
    faces = mesh.faces
    out = zeros(_get_float(mesh), length(faces))
    @inbounds for i ∈ eachindex(faces)
        out[i] = faces[i].centre[1]
    end
    return out
end

function yf(mesh::AbstractMesh)
    faces = mesh.faces
    out = zeros(_get_float(mesh), length(faces))
    @inbounds for i ∈ eachindex(faces)
        out[i] = faces[i].centre[2]
    end
    return out
end

function zf(mesh::AbstractMesh) 
    faces = mesh.faces
    out = zeros(_get_float(mesh), length(faces))
    @inbounds for i ∈ eachindex(faces)
        out[i] = faces[i].centre[3]
    end
    return out
end

function norm_static(arr, p = 2)
    sum = 0
    for i in eachindex(arr)
        val = (abs(arr[i]))^p
        sum += val
    end
    return sum^(1/p)
end

# function number_symbols(mesh)
#     symbol_mapping = Dict{Symbol, Int}()

#     for (i, boundary) in enumerate(mesh.boundaries)
#         if haskey(symbol_mapping, boundary.name)
#             # Do nothing, the symbol is already mapped
#         else
#             new_number = length(symbol_mapping) + 1
#             symbol_mapping[boundary.name] = new_number
#         end
#     end
    
#     return symbol_mapping
# end