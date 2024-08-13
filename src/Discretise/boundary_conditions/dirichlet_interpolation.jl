function adjust_boundary!(b_cpu, BC::Dirichlet, phif::FaceScalarField, phi, boundaries, boundary_cellsID,  backend, workgroup)
    phif_values = phif.values
    phi_values = phi.values

    # Copy to CPU
    # facesID_range = get_boundaries(BC, boundaries)
    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_dirichlet_scalar!(backend, workgroup)
    kernel!(BC, phif, phi, boundaries, boundary_cellsID, phif_values, phi_values, ndrange = kernel_range)
    # KernelAbstractions.synchronize(backend)
end

# Dirichlet
@kernel function adjust_boundary_dirichlet_scalar!(BC, phif, phi, boundaries, boundary_cellsID, phif_values, phi_values)
    i = @index(Global)
    # i = BC.ID

    @inbounds begin
        # (; IDs_range) = boundaries[BC.ID]
        (; IDs_range) = boundaries[BC.ID]
        fID = IDs_range[i]
        # for fID in IDs_range
            phif_values[fID] = BC.value
        # end
    end
end

function adjust_boundary!(b_cpu, BC::Dirichlet, psif::FaceVectorField, psi::VectorField, boundaries, boundary_cellsID, backend, workgroup)
    (; x, y, z) = psif

    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_dirichlet_vector!(backend, workgroup)
    kernel!(BC, psif, psi, boundaries, boundary_cellsID, x, y, z, ndrange = kernel_range)
    # KernelAbstractions.synchronize(backend)
end

@kernel function adjust_boundary_dirichlet_vector!(BC, psif, psi, boundaries, boundary_cellsID, x, y, z)
    i = @index(Global)
    # i = BC.ID

    @inbounds begin
        # (; IDs_range) = boundaries[i]
        (; IDs_range) = boundaries[BC.ID]
        # for fID in IDs_range
        fID = IDs_range[i]
            # fID = IDs_range[j]
            x[fID] = BC.value[1]
            y[fID] = BC.value[2]
            z[fID] = BC.value[3]
        # end
    end
end