export grad!, div! 



function grad!(grad::Grad{Linear,I,F}, phif, phi) where {I,F}
    interpolate!(get_scheme(grad), phif, phi)
    green_gauss!(grad, phif)
    # correct phif field 
    if grad.correct
        phif0 = copy(phif.values)
        for i ∈ 1:grad.correctors
            correct_interpolation!(get_scheme(grad), phif, grad, phif0)
            green_gauss!(grad, phif)
        end
        phif0 = nothing
    end
end

function div!()
    nothing
end

function green_gauss!(grad::Grad{S,I,F}, phif) where {S,I,F}
    (; x, y, z) = grad
    (; mesh) = phif
    (; cells, faces) = mesh
    for ci ∈ eachindex(cells)
        (; facesID, nsign, volume) = cells[ci]
        res = SVector{3,F}(0.0,0.0,0.0)
        for fi ∈ eachindex(facesID)
            fID = facesID[fi]
            (; area, normal) = faces[fID]
            res += phif.values[fID]*(area*normal*nsign[fi])
        end
        res /= volume
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
        res = phif.values[i]*(area*normal)
        x[cID] += res[1]/volume
        y[cID] += res[2]/volume
        z[cID] += res[3]/volume
    end
end