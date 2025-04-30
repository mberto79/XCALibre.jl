
function adjust_boundary!(b_cpu, BC::Empty, phif::FaceScalarField, phi, boundaries, boundary_cellsID, time, backend, workgroup)
    phif_values = phif.values
    phi_values = phi.values

    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_empty_scalar!(backend, workgroup)
    kernel!(BC, phif, phi, boundaries, boundary_cellsID, time, phif_values, phi_values, ndrange = kernel_range)
    # # KernelAbstractions.synchronize(backend)
end

# Neumann
@kernel function adjust_boundary_empty_scalar!(BC, phif, phi, boundaries, boundary_cellsID, time, phif_values, phi_values)
    i = @index(Global)

    @inbounds begin
        (; IDs_range) = boundaries[BC.ID]
        fID = IDs_range[i]

        # cID = boundary_cellsID[fID]
        # phif_values[fID] = phi_values[cID] 

        phif_values[fID] = 0.0 
    end
end

function adjust_boundary!(b_cpu, BC::Empty, psif::FaceVectorField, psi::VectorField, boundaries, boundary_cellsID, time, backend, workgroup)
    (; x, y, z) = psif

    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_empty_vector!(backend, workgroup)
    kernel!(BC, psif, psi, boundaries, boundary_cellsID, time, x, y, z, ndrange = kernel_range)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function adjust_boundary_empty_vector!(BC, psif, psi, boundaries, boundary_cellsID, time, x, y, z)
    i = @index(Global)

    @inbounds begin
        (; IDs_range) = boundaries[BC.ID]
        fID = IDs_range[i]

        # cID = boundary_cellsID[fID]
        # psi_cell = psi[cID]
        # x[fID] = psi_cell[1]
        # y[fID] = psi_cell[2]
        # z[fID] = psi_cell[3]
        
        x[fID] = 0.0
        y[fID] = 0.0
        z[fID] = 0.0
    end
end