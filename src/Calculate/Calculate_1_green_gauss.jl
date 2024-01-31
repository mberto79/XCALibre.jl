# function green_gauss!(grad::Grad, phif; source=false)
function green_gauss!(dx, dy, dz, phif)
    # (; x, y, z) = grad.result
    (; mesh, values) = phif
    # (; cells, faces) = mesh
    (; faces, cells, cell_faces, cell_nsign) = mesh
    F = eltype(mesh.nodes[1].coords)
    for ci ∈ eachindex(cells)
        # (; facesID, nsign, volume) = cells[ci]
        cell = cells[ci]
        (; volume) = cell
        res = SVector{3,F}(0.0,0.0,0.0)
        # for fi ∈ eachindex(facesID)
        for fi ∈ cell.faces_range
            # fID = facesID[fi]
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]
            (; area, normal) = faces[fID]
            # res += values[fID]*(area*normal*nsign[fi])
            res += values[fID]*(area*normal*nsign)
        end
        res /= volume
        dx[ci] = res[1]
        dy[ci] = res[2]
        dz[ci] = res[3]
    end
    # Add boundary faces contribution
    nbfaces = total_boundary_faces(mesh)
    for i ∈ 1:nbfaces
        face = faces[i]
        (; ownerCells, area, normal) = face
        cID = ownerCells[1] 
        (; volume) = cells[cID]
        res = values[i]*(area*normal)
        res /= volume
        dx[cID] += res[1]
        dy[cID] += res[2]
        dz[cID] += res[3]
    end
end