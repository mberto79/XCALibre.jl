@define_boundary Wall Laplacian{Linear} begin
    phi = term.phi 
    (; area, delta, normal) = face 

    U_boundary = SVector{3}(0.0,0.0,0.0) # user given vector

    velocity_diff = phi[cellID] .- U_boundary
    J = term.flux[fID]
    norm_vel = (velocity_diff⋅normal)*normal

    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    ap, ap*(U_boundary[component.value] + norm_vel[component.value])
end

@define_boundary Wall Divergence{Linear} begin
    ap = term.sign[1]*(term.flux[fID])

    0.0, 0.0
end

@define_boundary Wall Divergence{Upwind} begin
    ap = term.sign[1]*(term.flux[fID])
    0.0, 0.0
end

@define_boundary Wall Divergence{BoundedUpwind} begin
    ap = term.sign[1]*(term.flux[fID])
    vol = 1#cell.volume

    -term.flux[fID], 0.0
end