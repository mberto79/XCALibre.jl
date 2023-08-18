export Div
export div! 

# Define Divergence type and functionality

struct Div{VF<:VectorField,FVF<:FaceVectorField,F,M}
    vector::VF
    face_vector::FVF
    values::Vector{F}
    mesh::M
end
Div(vector::VectorField) = begin
    mesh = vector.mesh
    face_vector = FaceVectorField(mesh)
    values = zeros(F, length(mesh.cells))
    Div(vector, face_vector, values, mesh)
end


# function grad!(grad::Grad{Linear,I,F}, phif, phi, BCs; source=false) where {I,F}
#     # interpolate!(get_scheme(grad), phif, phi, BCs)
#     interpolate!(phif, phi)
#     correct_boundaries!(phif, phi, BCs)
#     green_gauss!(grad, phif; source)

#     # correct phif field 
#     if grad.correct
#         phif0 = copy(phif.values) # it would be nice to find a way to avoid this!
#         for i ∈ 1:grad.correctors
#             correct_interpolation!(get_scheme(grad), phif, grad, phif0)
#             green_gauss!(grad, phif)
#         end
#         phif0 = nothing
#     end
# end

# function div!(div::Div{I,F}, BCs) where {I,F}
#     (; mesh, values, vector, face_vector) = div
#     (; cells, faces) = mesh
#     # interpolate!(face_vector, vector, BCs)
#     interpolate!(face_vector, vector)
#     correct_boundaries!(face_vector, vector, BCs)

#     for ci ∈ eachindex(cells)
#         (; facesID, nsign, volume) = cells[ci]
#         values[ci] = zero(F)
#         for fi ∈ eachindex(facesID)
#             fID = facesID[fi]
#             (; area, normal) = faces[fID]
#             values[ci] += face_vector(fID)⋅(area*normal*nsign[fi])
#         end
#     end
#     # Add boundary faces contribution
#     nbfaces = total_boundary_faces(mesh)
#     for i ∈ 1:nbfaces
#         face = faces[i]
#         (; ownerCells, area, normal) = face
#         cID = ownerCells[1] 
#         # Boundary normals are correct by definition
#         values[cID] += face_vector(i)⋅(area*normal) 
#     end
# end


# function flux!(phif::FS, psif::FV) where {FS<:FaceScalarField,FV<:FaceVectorField}
#     (; mesh, values) = phif
#     (; faces) = mesh 
#     @inbounds for fID ∈ eachindex(faces)
#         (; area, normal) = faces[fID]
#         Sf = area*normal
#         values[fID] = psif[fID]⋅Sf
#     end
# end

function div!(phi::ScalarField, psif::FaceVectorField)
    mesh = phi.mesh
    (; cells, faces) = mesh
    F = eltype(mesh.nodes[1].coords)

    for ci ∈ eachindex(cells)
        (; facesID, nsign, volume) = cells[ci]
        phi.values[ci] = zero(F)
        for fi ∈ eachindex(facesID)
            fID = facesID[fi]
            (; area, normal) = faces[fID]
            Sf = area*normal
            phi.values[ci] += psif[fID]⋅Sf*nsign[fi]/volume
        end
    end
    # Add boundary faces contribution
    nbfaces = total_boundary_faces(mesh)
    for fID ∈ 1:nbfaces
        cID = faces[fID].ownerCells[1]
        volume = cells[cID].volume
        (; area, normal) = faces[fID]
        Sf = area*normal
        # Boundary normals are correct by definition
        phi.values[cID] += psif[fID]⋅Sf/volume
    end
end

# function div!(phi::ScalarField, phif::FaceScalarField)
#     (; mesh, values) = phif
#     (; cells, faces) = mesh
#     F = eltype(mesh.nodes[1].coords)

#     for ci ∈ eachindex(cells)
#         (; facesID, nsign, volume) = cells[ci]
#         phi.values[ci] = zero(F)
#         for fi ∈ eachindex(facesID)
#             fID = facesID[fi]
#             phi.values[ci] += values[fID]*nsign[fi]
#         end
#     end
#     # Add boundary faces contribution
#     nbfaces = total_boundary_faces(mesh)
#     for fID ∈ 1:nbfaces
#         cID = faces[fID].ownerCells[1]
#         # Boundary normals are correct by definition
#         phi.values[cID] += values[fID]
#     end
# end
