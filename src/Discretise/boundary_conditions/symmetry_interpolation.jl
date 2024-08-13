function adjust_boundary!(b_cpu, BC::Symmetry, phif::FaceScalarField, phi, boundaries, boundary_cellsID, backend, workgroup)
    phif_values = phif.values
    phi_values = phi.values


    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_symmetry_scalar!(backend, workgroup)
    kernel!(BC, phif, phi, boundaries, boundary_cellsID, phif_values, phi_values, ndrange = kernel_range)
    # KernelAbstractions.synchronize(backend)
end

@kernel function adjust_boundary_symmetry_scalar!(BC, phif, phi, boundaries, boundary_cellsID, phif_values, phi_values)
    i = @index(Global)
    # i = BC.ID

    @inbounds begin
        # (; IDs_range) = boundaries[i]
        (; IDs_range) = boundaries[BC.ID]
        # for fID in IDs_range
        fID = IDs_range[i]
            cID = boundary_cellsID[fID]
            phif_values[fID] = phi_values[cID] 
        # end
    end
end

function adjust_boundary!(b_cpu, BC::Symmetry, psif::FaceVectorField, psi::VectorField, boundaries, boundary_cellsID, backend, workgroup)
    (; x, y, z) = psif

    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_symmetry_vector!(backend, workgroup)
    kernel!(BC, psif, psi, boundaries, boundary_cellsID, x, y, z, ndrange = kernel_range)
    # KernelAbstractions.synchronize(backend)
end

@kernel function adjust_boundary_symmetry_vector!(BC, psif, psi, boundaries, boundary_cellsID, x, y, z)
    i = @index(Global)
    # i = BC.ID

    @inbounds begin
        # (; IDs_range) = boundaries[i]
        (; IDs_range) = boundaries[BC.ID]
        # for fID in IDs_range
        fID = IDs_range[i]
            cID = boundary_cellsID[fID]
            face = psi.mesh.faces[fID]
            (; normal, delta) = face

            psi_cell = psi[cID]

            psi_normal = (psi_cell⋅normal)*normal

            x[fID] = psi_cell[1] - psi_normal[1]
            y[fID] = psi_cell[2] - psi_normal[2]
            z[fID] = psi_cell[3] - psi_normal[3]
        # end
    end
end