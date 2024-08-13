function adjust_boundary!(b_cpu, BC::FixedTemperature, phif::FaceScalarField, phi, boundaries, boundary_cellsID,  backend, workgroup)
    phif_values = phif.values
    phi_values = phi.values

    # Copy to CPU
    # facesID_range = get_boundaries(BC, boundaries)
    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_fixedtemperature_scalar!(backend, workgroup)
    kernel!(BC, phif, phi, boundaries, boundary_cellsID, phif_values, phi_values, ndrange = kernel_range)
    # KernelAbstractions.synchronize(backend)
end

# FixedTemperature
@kernel function adjust_boundary_fixedtemperature_scalar!(BC, phif, phi, boundaries, boundary_cellsID, phif_values, phi_values)
    i = @index(Global)
    # i = BC.ID

    @inbounds begin
        # (; IDs_range) = boundaries[BC.ID]
        (; IDs_range) = boundaries[BC.ID]
        fID = IDs_range[i]
        # for fID in IDs_range
        # extract user provided information
        (; T, energy_model) = BC.value

        phif_values[fID] = energy_model.update_BC(T)
        # end
    end
end