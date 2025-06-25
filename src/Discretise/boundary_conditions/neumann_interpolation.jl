
@inline function boundary_interpolation!(
    BC::Neumann, phif::FaceScalarField, phi, boundary_cellsID, time, fID)
    @inbounds begin
        (; faces) = phi.mesh
        face = faces[fID]
        (; delta) = face
        cID = boundary_cellsID[fID]
        phif[fID] = phi[cID] + delta*BC.value 
    end
    nothing
end


@inline function boundary_interpolation!(
    BC::Neumann, psif::FaceVectorField, psi, boundary_cellsID, time, fID)
    @inbounds begin
        error("Neumann boundary condition for vector fields is not implemented yet.")
    end
    nothing
end

# const NEUMANN = Union{Neumann, Extrapolated, KWallFunction, NutWallFunction, OmegaWallFunction}

# function adjust_boundary!(BC::NEUMANN, phif::FaceScalarField, phi, boundaries, boundary_cellsID, time, backend, workgroup)
#     phif_values = phif.values
#     phi_values = phi.values

#     kernel_range = length(BC.IDs_range)

#     kernel! = adjust_boundary_neumann_scalar!(backend, workgroup)
#     kernel!(BC, phif, phi, boundaries, boundary_cellsID, time, phif_values, phi_values, ndrange = kernel_range)
#     # # KernelAbstractions.synchronize(backend)
# end

# # Neumann
# @kernel function adjust_boundary_neumann_scalar!(BC, phif, phi, boundaries, boundary_cellsID, time, phif_values, phi_values)
#     i = @index(Global)

#     @inbounds begin
#         (; IDs_range) = boundaries[BC.ID]
#         fID = IDs_range[i]
#         cID = boundary_cellsID[fID]
#         phif_values[fID] = phi_values[cID] 
#     end
# end

# function adjust_boundary!(BC::NEUMANN, psif::FaceVectorField, psi::VectorField, boundaries, boundary_cellsID, time, backend, workgroup)
#     (; x, y, z) = psif

#     kernel_range = length(BC.IDs_range)

#     kernel! = adjust_boundary_neumann_vector!(backend, workgroup)
#     kernel!(BC, psif, psi, boundaries, boundary_cellsID, time, x, y, z, ndrange = kernel_range)
#     # # KernelAbstractions.synchronize(backend)
# end

# @kernel function adjust_boundary_neumann_vector!(BC, psif, psi, boundaries, boundary_cellsID, time, x, y, z)
#     i = @index(Global)

#     @inbounds begin
#         (; IDs_range) = boundaries[BC.ID]
#         fID = IDs_range[i]
#         cID = boundary_cellsID[fID]
#         psi_cell = psi[cID]
#         x[fID] = psi_cell[1]
#         y[fID] = psi_cell[2]
#         z[fID] = psi_cell[3]
#     end
# end