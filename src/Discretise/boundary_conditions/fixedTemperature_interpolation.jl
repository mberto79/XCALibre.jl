
@inline function boundary_interpolation!(
    BC::FixedTemperature, phif::FaceScalarField, phi, boundary_cellsID, time, fID)
    @inbounds begin
        (; T, energy_model) = BC.value
        phif[fID] = energy_model(T)
    end
    nothing
end
