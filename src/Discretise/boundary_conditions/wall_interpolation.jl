function adjust_boundary!(b_cpu, BC::Wall, phif::FaceScalarField, phi, boundaries, boundary_cellsID,  backend, workgroup)
    phif_values = phif.values
    phi_values = phi.values

    # Copy to CPU
    # facesID_range = get_boundaries(BC, boundaries)
    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_dirichlet_scalar!(backend, workgroup)
    kernel!(BC, phif, phi, boundaries, boundary_cellsID, phif_values, phi_values, ndrange = kernel_range)
    # KernelAbstractions.synchronize(backend)
end


function adjust_boundary!(b_cpu, BC::Wall, psif::FaceVectorField, psi::VectorField, boundaries, boundary_cellsID, backend, workgroup)
    (; x, y, z) = psif

    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_dirichlet_vector!(backend, workgroup)
    kernel!(BC, psif, psi, boundaries, boundary_cellsID, x, y, z, ndrange = kernel_range)
    # KernelAbstractions.synchronize(backend)
end