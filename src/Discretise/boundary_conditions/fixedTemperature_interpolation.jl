
@inline function boundary_interpolation!(
    BC::FixedTemperature, phif::FaceScalarField, phi, boundary_cellsID, time, fID)
    @inbounds begin
        (; T, energy_model) = BC.value
        phif[fID] = energy_model(T)
    end
    nothing
end

# function adjust_boundary!(BC::FixedTemperature, phif::FaceScalarField, phi, boundaries, boundary_cellsID, time,  backend, workgroup)
#     phif_values = phif.values
#     phi_values = phi.values

#     kernel_range = length(BC.IDs_range)

#     kernel! = adjust_boundary_fixedtemperature_scalar!(backend, workgroup)
#     kernel!(BC, phif, phi, boundaries, boundary_cellsID, time, phif_values, phi_values, ndrange = kernel_range)
#     # # KernelAbstractions.synchronize(backend)
# end

# # FixedTemperature
# @kernel function adjust_boundary_fixedtemperature_scalar!(BC, phif, phi, boundaries, boundary_cellsID, time, phif_values, phi_values)
#     i = @index(Global)

#     @inbounds begin
#         (; IDs_range) = boundaries[BC.ID]
#         fID = IDs_range[i]
#         (; T, energy_model) = BC.value
#         phif_values[fID] = energy_model.update_BC(T)
#     end
# end