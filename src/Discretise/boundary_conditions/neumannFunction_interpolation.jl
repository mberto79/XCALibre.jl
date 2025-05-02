# Implementation to dispatch when user provides an simple function
function adjust_boundary!(b_cpu, BC::NeumannFunction{T,Test}, phif::FaceScalarField, phi, boundaries, boundary_cellsID, time, backend, workgroup
    ) where {T,Test<:Function}
    phif_values = phif.values
    phi_values = phi.values

    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_neumannFunction_scalar!(backend, workgroup)
    kernel!(BC, phif, phi, boundaries, boundary_cellsID, time, phif_values, phi_values, ndrange = kernel_range)
    # # KernelAbstractions.synchronize(backend)
end

function adjust_boundary!(b_cpu, BC::NeumannFunction{T,Test}, psif::FaceVectorField, psi::VectorField, boundaries, boundary_cellsID, time, backend, workgroup
    ) where {T,Test<:Function}
    (; x, y, z) = psif

    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_neumann_vector!(backend, workgroup)
    kernel!(BC, psif, psi, boundaries, boundary_cellsID, time, x, y, z, ndrange = kernel_range)
    # # KernelAbstractions.synchronize(backend)
end

# Implementation to dispatch when user provides an XCALibreUserFunctor
function adjust_boundary!(b_cpu, BC::NeumannFunction{T,Test}, phif::FaceScalarField, phi, boundaries, boundary_cellsID, time, backend, workgroup
    ) where {T,Test<:XCALibreUserFunctor}
    phif_values = phif.values
    phi_values = phi.values

    kernel_range = length(b_cpu[BC.ID].IDs_range)

    if !BC.value.steady
        config = (;hardware=(;backend=backend, workgroup=workgroup)) # temp solution
        update_user_boundary!(
            BC, eqnModel, component, faces, cells, facesID_range, time, config)
    end

    kernel! = adjust_boundary_neumannFunction_scalar!(backend, workgroup)
    kernel!(BC, phif, phi, boundaries, boundary_cellsID, time, phif_values, phi_values, ndrange = kernel_range)
    # # KernelAbstractions.synchronize(backend)
end

function adjust_boundary!(b_cpu, BC::NeumannFunction{T,Test}, psif::FaceVectorField, psi::VectorField, boundaries, boundary_cellsID, time, backend, workgroup
    ) where {T,Test<:XCALibreUserFunctor}
    (; x, y, z) = psif

    kernel_range = length(b_cpu[BC.ID].IDs_range)

    if !BC.value.steady
        config = (;hardware=(;backend=backend, workgroup=workgroup)) # temp solution
        update_user_boundary!(
            BC, eqnModel, component, faces, cells, facesID_range, time, config)
    end

    kernel! = adjust_boundary_neumann_vector!(backend, workgroup)
    kernel!(BC, psif, psi, boundaries, boundary_cellsID, time, x, y, z, ndrange = kernel_range)
    # # KernelAbstractions.synchronize(backend)
end

# Implement interpolation for scalars and vectors

@kernel function adjust_boundary_neumannFunction_scalar!(BC, phif, phi, boundaries, boundary_cellsID, time, phif_values, phi_values)
    i = @index(Global)

    @inbounds begin
        (; IDs_range) = boundaries[BC.ID]
        fID = IDs_range[i]
        cID = boundary_cellsID[fID]
        phif_values[fID] = phi_values[cID] 
    end
end

@kernel function adjust_boundary_neumannFunction_vector!(BC, psif, psi, boundaries, boundary_cellsID, time, x, y, z)
    i = @index(Global)

    @inbounds begin
        (; IDs_range) = boundaries[BC.ID]
        fID = IDs_range[i]
        cID = boundary_cellsID[fID]
        psi_cell = psi[cID]
        x[fID] = psi_cell[1]
        y[fID] = psi_cell[2]
        z[fID] = psi_cell[3]
    end
end