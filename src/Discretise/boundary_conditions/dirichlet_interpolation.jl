@inline function boundary_interpolation!(
    BC::Dirichlet, phif::FaceScalarField, phi, boundary_cellsID, time, fID)
    @inbounds phif[fID] = BC.value
    nothing
end

@inline function boundary_interpolation!(
    BC::Dirichlet, psif::FaceVectorField, psi, boundary_cellsID, time, fID)
    @inbounds psif[fID] = BC.value
    nothing
end

# function adjust_boundary!(BC::Dirichlet, phif::FaceScalarField, phi, boundaries, boundary_cellsID, time,  backend, workgroup)
#     # phif_values = phif.values
#     # phi_values = phi.values

#     kernel_range = length(BC.IDs_range)

#     kernel! = boundary_interpolation!(backend, workgroup)
#     kernel!(BC, phif, phi, boundaries, boundary_cellsID, time, ndrange = kernel_range)
#     # # KernelAbstractions.synchronize(backend)
# end

# # Dirichlet


# function adjust_boundary!(BC::Dirichlet, psif::FaceVectorField, psi::VectorField, boundaries, boundary_cellsID, time, backend, workgroup)
#     # (; x, y, z) = psif

#     kernel_range = length(BC.IDs_range)

#     kernel! = boundary_interpolation!(backend, workgroup)
#     kernel!(BC, psif, psi, boundaries, boundary_cellsID, time, ndrange = kernel_range)
#     # # KernelAbstractions.synchronize(backend)
# end

# @kernel function boundary_interpolation!(BC::Dirichlet, phif::FaceScalarField, phi, boundaries, boundary_cellsID, time)
#     i = @index(Global)

#     @inbounds begin
#         (; IDs_range) = boundaries[BC.ID]
#         fID = IDs_range[i]
#         phif[fID] = BC.value
#     end
# end

# @kernel function boundary_interpolation!(BC::Dirichlet, psif::FaceVectorField, psi, boundaries, boundary_cellsID, time)
#     i = @index(Global)

#     @inbounds begin
#         (; IDs_range) = boundaries[BC.ID]
#         fID = IDs_range[i]
#         psif[fID] = BC.value
#     end
# end