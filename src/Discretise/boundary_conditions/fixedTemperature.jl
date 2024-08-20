@define_boundary FixedTemperature Divergence{Upwind} begin
    (; T, energy_model) = bc.value

    h = energy_model.update_BC(T)
    ap = term.sign[1]*(term.flux[fID])

    0.0, -ap*h
end


@define_boundary FixedTemperature Divergence{BoundedUpwind} begin

    (; T, energy_model) = bc.value

    h = energy_model.update_BC(T)
    ap = term.sign[1]*(term.flux[fID])
    vol = 1#cell.volume
    
    -term.flux[fID], -ap*h
end

@define_boundary FixedTemperature Laplacian{Linear} begin
    J = term.flux[fID]
    (; area, delta) = face 
    (; T, energy_model) = bc.value

    h = energy_model.update_BC(T)

    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    ap, ap*h
end

@define_boundary FixedTemperature Divergence{Linear} begin
    (; T, energy_model) = bc.value
    h = energy_model.update_BC(T)
    
    0.0, term.sign[1]*(-term.flux[fID]*h)
end