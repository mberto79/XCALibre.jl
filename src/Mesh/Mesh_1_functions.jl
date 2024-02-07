export _get_float, _get_int, _get_backend
export total_boundary_faces, boundary_index
# export number_symbols
export x, y, z # access cell centres
export xf, yf, zf # access face centres

_get_int(mesh) = eltype(mesh.get_int)
_get_float(mesh) = eltype(mesh.get_float)
_get_backend(mesh) = get_backend(mesh.cells)

# function total_boundary_faces(mesh::Mesh2{I,F}) where {I,F}
function total_boundary_faces(mesh::Mesh2)
    (; boundaries) = mesh
    # nbfaces = zero(I)
    nbfaces = zero(_get_int(mesh))
    @inbounds for boundary ∈ boundaries
        nbfaces += length(boundary.facesID)
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

function x(mesh::Mesh2{I,F}) where {I,F}
    cells = mesh.cells
    out = zeros(F, length(cells))
    @inbounds for i ∈ eachindex(cells)
        out[i] = cells[i].centre[1]
    end
    return out
end

function y(mesh::Mesh2{I,F}) where {I,F}
    cells = mesh.cells
    out = zeros(F, length(cells))
    @inbounds for i ∈ eachindex(cells)
        out[i] = cells[i].centre[2]
    end
    return out
end

function z(mesh::Mesh2{I,F}) where {I,F}
    cells = mesh.cells
    out = zeros(F, length(cells))
    @inbounds for i ∈ eachindex(cells)
        out[i] = cells[i].centre[3]
    end
    return out
end

function xf(mesh::Mesh2{I,F}) where {I,F}
    faces = mesh.faces
    out = zeros(F, length(faces))
    @inbounds for i ∈ eachindex(faces)
        out[i] = faces[i].centre[1]
    end
    return out
end

function yf(mesh::Mesh2{I,F}) where {I,F}
    faces = mesh.faces
    out = zeros(F, length(faces))
    @inbounds for i ∈ eachindex(faces)
        out[i] = faces[i].centre[2]
    end
    return out
end

function zf(mesh::Mesh2{I,F}) where {I,F}
    faces = mesh.faces
    out = zeros(F, length(faces))
    @inbounds for i ∈ eachindex(faces)
        out[i] = faces[i].centre[3]
    end
    return out
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