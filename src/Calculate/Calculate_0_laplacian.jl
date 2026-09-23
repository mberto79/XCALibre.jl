export laplacian!, surface_gradient!

function surface_gradient!(gradf, phif, phi, BCs, time, config)
    interpolate!(phif, phi, config)
    correct_boundaries!(phif, phi, BCs, time, config)

    mesh = phi.mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(mesh.faces)
    n_bfaces = length(mesh.boundary_cellsID)
    kernel! = _sized(_surface_gradient!, backend, workgroup, ndrange)
    kernel!(gradf, phif, phi, mesh.faces, mesh.boundary_cellsID, n_bfaces)
end

@kernel function _surface_gradient!(gradf, phif, phi, faces, boundary_cellsID, n_bfaces)
    fID = @index(Global)

    @uniform begin
        gx, gy, gz = gradf.x, gradf.y, gradf.z
    end

    @inbounds begin
        ownerCells, normal, delta = faces.ownerCells[fID], faces.normal[fID], faces.delta[fID]
        snGrad = if fID <= n_bfaces
            (phif[fID] - phi[boundary_cellsID[fID]]) / delta
        else
            (phi[ownerCells[2]] - phi[ownerCells[1]]) / delta
        end

        gx[fID] = snGrad * normal[1]
        gy[fID] = snGrad * normal[2]
        gz[fID] = snGrad * normal[3]
    end
end

function laplacian!(phi_out, phif_in ,phi_in, BCs, time, config; disp_warn=true)
    interpolate!(phif_in, phi_in, config)
    correct_boundaries!(phif_in, phi_in, BCs, time, config)
    mesh = phi_out.mesh

    fill!(phi_out.values, zero(eltype(phi_out.values)))

    n_bfaces = length(mesh.boundary_cellsID)

    for fID ∈ 1:n_bfaces
        cID = mesh.boundary_cellsID[fID]
        flux = mesh.faces.area[fID] * (phif_in[fID] - phi_in[cID]) / mesh.faces.delta[fID]
        phi_out[cID] += flux / mesh.cells.volume[cID]
    end

    for fID ∈ (n_bfaces + 1):length(mesh.faces)
        cID1 = mesh.faces.ownerCells[fID][1]
        cID2 = mesh.faces.ownerCells[fID][2]
        flux = mesh.faces.area[fID] * (phi_in[cID2] - phi_in[cID1]) / mesh.faces.delta[fID]
        phi_out[cID1] += flux / mesh.cells.volume[cID1]
        phi_out[cID2] -= flux / mesh.cells.volume[cID2]
    end

end
