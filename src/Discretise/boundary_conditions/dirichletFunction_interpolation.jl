function adjust_boundary!(b_cpu, BC::DirichletFunction, phif::FaceScalarField, phi, boundaries, boundary_cellsID,  backend, workgroup)
    (; faces) = phi.mesh
    phif_values = phif.values
    phi_values = phi.values

    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_dirichletFunction_scalar!(backend, workgroup)
    kernel!(BC, phif, phi, boundaries, faces, boundary_cellsID, phif_values, phi_values, ndrange = kernel_range)
    # KernelAbstractions.synchronize(backend)
end

# DirichletFunction
@kernel function adjust_boundary_dirichletFunction_scalar!(BC, phif, phi, boundaries, faces, boundary_cellsID, phif_values, phi_values)
    i = @index(Global)

    @inbounds begin
        (; IDs_range) = boundaries[BC.ID]
        fID = IDs_range[i]
        face = faces[fID]
        phif_values[fID] = BC.value(face.centre)
    end
end

function adjust_boundary!(b_cpu, BC::DirichletFunction, psif::FaceVectorField, psi::VectorField, boundaries, boundary_cellsID, backend, workgroup)
    (; x, y, z) = psif
    (; faces) = psi.mesh

    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_dirichletFunction_vector!(backend, workgroup)
    kernel!(BC, psif, psi, boundaries, faces, boundary_cellsID, x, y, z, ndrange = kernel_range)
    # KernelAbstractions.synchronize(backend)
end

@kernel function adjust_boundary_dirichletFunction_vector!(BC, psif, psi, boundaries, faces, boundary_cellsID, x, y, z)
    i = @index(Global)

    @inbounds begin
        (; IDs_range) = boundaries[BC.ID]
        fID = IDs_range[i]
        face = faces[fID]
        value = BC.value(face.centre)
        x[fID] = value[1]
        y[fID] = value[2]
        z[fID] = value[3]
    end
end