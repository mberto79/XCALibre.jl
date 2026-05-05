export laplacian!

function laplacian!(phi_out, phif_in ,phi_in, BCs, time, config; disp_warn=true)
    interpolate!(phif_in, phi_in, config)
    correct_boundaries!(phif_in, phi_in, BCs, time, config)
    mesh = phi_out.mesh

    fill!(phi_out.values, zero(eltype(phi_out.values)))

    n_bfaces = length(mesh.boundary_cellsID)

    for fID ∈ 1:n_bfaces
        face = mesh.faces[fID]
        cID = mesh.boundary_cellsID[fID]
        flux = face.area * (phif_in[fID] - phi_in[cID]) / face.delta
        phi_out[cID] += flux / mesh.cells[cID].volume
    end

    for fID ∈ (n_bfaces + 1):length(mesh.faces)
        face = mesh.faces[fID]
        cID1 = face.ownerCells[1]
        cID2 = face.ownerCells[2]
        flux = face.area * (phi_in[cID2] - phi_in[cID1]) / face.delta
        phi_out[cID1] += flux / mesh.cells[cID1].volume
        phi_out[cID2] -= flux / mesh.cells[cID2].volume
    end

end
