function adjust_boundary!(b_cpu, BC::DirichletFunction, phif::FaceScalarField, phi, boundaries, boundary_cellsID, time, backend, workgroup)
    (; cells, faces) = phi.mesh
    phif_values = phif.values
    phi_values = phi.values

    facesID_range = b_cpu[BC.ID].IDs_range
    kernel_range = length(facesID_range)

    if !BC.value.steady
        config = (;hardware=(;backend=backend, workgroup=workgroup)) # temp solution
        update_user_boundary!(BC, faces, cells, facesID_range, time, config)
    end

    kernel! = adjust_boundary_dirichletFunction_scalar!(backend, workgroup)
    kernel!(BC, phif, phi, boundaries, faces, boundary_cellsID, time, phif_values, phi_values, ndrange = kernel_range)
    # KernelAbstractions.synchronize(backend)
end

# DirichletFunction
@kernel function adjust_boundary_dirichletFunction_scalar!(BC, phif, phi, boundaries, faces, boundary_cellsID, time, phif_values, phi_values)
    i = @index(Global)

    @inbounds begin
        (; IDs_range) = boundaries[BC.ID]
        fID = IDs_range[i]
        face = faces[fID]
        phif_values[fID] = BC.value(face.centre, time, i)
    end
end

function adjust_boundary!(b_cpu, BC::DirichletFunction, psif::FaceVectorField, psi::VectorField, boundaries, boundary_cellsID, time, backend, workgroup)
    (; x, y, z) = psif
    (; cells, faces) = psi.mesh

    facesID_range = b_cpu[BC.ID].IDs_range
    kernel_range = length(facesID_range)

    if !BC.value.steady
        config = (;hardware=(;backend=backend, workgroup=workgroup)) # temp solution
        update_user_boundary!(BC, faces, cells, facesID_range, time, config)
    end

    kernel! = adjust_boundary_dirichletFunction_vector!(backend, workgroup)
    kernel!(BC, psif, psi, boundaries, faces, boundary_cellsID, time, x, y, z, ndrange = kernel_range)
    # KernelAbstractions.synchronize(backend)
end

@kernel function adjust_boundary_dirichletFunction_vector!(BC, psif, psi, boundaries, faces, boundary_cellsID, time, x, y, z)
    i = @index(Global)

    @inbounds begin
        (; IDs_range) = boundaries[BC.ID]
        fID = IDs_range[i]
        face = faces[fID]
        value = BC.value(face.centre, time, i)
        x[fID] = value[1]
        y[fID] = value[2]
        z[fID] = value[3]
    end
end