function adjust_boundary!(BC::Wall, phif::FaceScalarField, phi, boundaries, boundary_cellsID, time,  backend, workgroup)
    phif_values = phif.values
    phi_values = phi.values

    kernel_range = length(BC.IDs_range)

    # kernel! = adjust_boundary_dirichlet_scalar!(backend, workgroup)
    kernel! = boundary_interpolation!(backend, workgroup)
    kernel!(BC, phif, phi, boundaries, boundary_cellsID, time, phif_values, phi_values, ndrange = kernel_range)
    # # KernelAbstractions.synchronize(backend)
end


function adjust_boundary!(BC::Wall, psif::FaceVectorField, psi::VectorField, boundaries, boundary_cellsID, time, backend, workgroup)
    (; x, y, z) = psif

    kernel_range = length(BC.IDs_range)

    kernel! = boundary_interpolation!(backend, workgroup)
    kernel!(BC, psif, psi, boundaries, boundary_cellsID, time, x, y, z, ndrange = kernel_range)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function boundary_interpolation!(BC::Wall, phif::FaceScalarField, phi, boundaries, boundary_cellsID, time, phif_values, phi_values)
    i = @index(Global)

    @inbounds begin
        (; IDs_range) = boundaries[BC.ID]
        fID = IDs_range[i]
        cID = boundary_cellsID[fID]
        phif_values[fID] = phi_values[cID] 
    end
end

@kernel function boundary_interpolation!(BC::Wall, psif::FaceVectorField, psi, boundaries, boundary_cellsID, time, x, y, z)
    i = @index(Global)

    @inbounds begin
        (; IDs_range) = boundaries[BC.ID]
        fID = IDs_range[i]
        x[fID] = BC.value[1]
        y[fID] = BC.value[2]
        z[fID] = BC.value[3]
    end
end