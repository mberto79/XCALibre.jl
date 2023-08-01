function green_gauss!(grad::Grad, phif; source=false)
    (; x, y, z) = grad
    (; mesh, values) = phif
    (; cells, faces) = mesh
    F = eltype(mesh.nodes[1].coords)
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