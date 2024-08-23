@define_boundary FixedTemperature Divergence{Linear} begin
    (; T, energy_model) = bc.value
    flux = term.flux[fID]
    ap = term.sign*(-flux)
    h = energy_model.update_BC(T) # To do: find nicer way to accomplish this
    0.0, ap*h
end


@define_boundary FixedTemperature Divergence{Upwind} begin
    (; T, energy_model) = bc.value
    flux = term.flux[fID]
    h = energy_model.update_BC(T)
    ap = term.sign*(flux)
    0.0, -ap*h
end


@define_boundary FixedTemperature Divergence{BoundedUpwind} begin
    (; T, energy_model) = bc.value
    flux = term.flux[fID]
    ap = term.sign*(flux)
    h = energy_model.update_BC(T)
    -flux, -ap*h
end

@define_boundary FixedTemperature Laplacian{Linear} begin
    J = term.flux[fID]
    (; area, delta) = face 
    (; T, energy_model) = bc.value
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    h = energy_model.update_BC(T)
    ap, ap*h
end

