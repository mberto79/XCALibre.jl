export grad!, div! 
export source!

function source!(grad::Grad{Linear,I,F}, phif, phi, BCs; source=true) where {I,F}
    grad!(grad, phif, phi, BCs; source=source)
end

function grad!(grad::Grad{Linear,I,F}, phif, phi, BCs; source=false) where {I,F}
    # interpolate!(get_scheme(grad), phif, phi, BCs)
    interpolate!(phif, phi)
    correct_boundaries!(phif, phi, BCs)
    green_gauss!(grad, phif; source)

    # correct phif field 
    if grad.correct
        phif0 = copy(phif.values) # it would be nice to find a way to avoid this!
        for i ∈ 1:grad.correctors
            correct_interpolation!(get_scheme(grad), phif, grad, phif0)
            green_gauss!(grad, phif)
        end
        phif0 = nothing
    end
end

function div!(div::Div{I,F}, BCs) where {I,F}
    (; mesh, values, vector, face_vector) = div
    (; cells, faces) = mesh
    # interpolate!(face_vector, vector, BCs)
    interpolate!(face_vector, vector)
    correct_boundaries!(face_vector, vector, BCs)

    for ci ∈ eachindex(cells)
        (; facesID, nsign, volume) = cells[ci]
        values[ci] = zero(F)
        for fi ∈ eachindex(facesID)
            fID = facesID[fi]
            (; area, normal) = faces[fID]
            values[ci] += face_vector(fID)⋅(area*normal*nsign[fi])
        end
    end
    # Add boundary faces contribution
    nbfaces = total_boundary_faces(mesh)
    for i ∈ 1:nbfaces
        face = faces[i]
        (; ownerCells, area, normal) = face
        cID = ownerCells[1] 
        # Boundary normals are correct by definition
        values[cID] += face_vector(i)⋅(area*normal) 
    end
end

function div!(phi::ScalarField, phif::FaceScalarField{I,F}) where {I,F}
    (; mesh, values) = phif
    (; cells, faces) = mesh

    for ci ∈ eachindex(cells)
        (; facesID, nsign, volume) = cells[ci]
        phi.values[ci] = zero(F)
        for fi ∈ eachindex(facesID)
            fID = facesID[fi]
            phi.values[ci] += values[fID]*nsign[fi]
        end
    end
    # Add boundary faces contribution
    nbfaces = total_boundary_faces(mesh)
    for fID ∈ 1:nbfaces
        cID = faces[fID].ownerCells[1]
        # Boundary normals are correct by definition
        phi.values[cID] += values[fID]
    end
end

function green_gauss!(grad::Grad{S,I,F}, phif; source=false) where {S,I,F}
    (; x, y, z) = grad
    (; mesh, values) = phif
    (; cells, faces) = mesh
    for ci ∈ eachindex(cells)
        (; facesID, nsign, volume) = cells[ci]
        res = SVector{3,F}(0.0,0.0,0.0)
        for fi ∈ eachindex(facesID)
            fID = facesID[fi]
            (; area, normal) = faces[fID]
            res += values[fID]*(area*normal*nsign[fi])
        end
        if !source
            res /= volume
        end
        x[ci] = res[1]
        y[ci] = res[2]
        z[ci] = res[3]
    end
    # Add boundary faces contribution
    nbfaces = total_boundary_faces(mesh)
    for i ∈ 1:nbfaces
        face = faces[i]
        (; ownerCells, area, normal) = face
        cID = ownerCells[1] 
        (; volume) = cells[cID]
        res = values[i]*(area*normal)
        if !source
            res /= volume
        end
        x[cID] += res[1]
        y[cID] += res[2]
        z[cID] += res[3]
    end
end