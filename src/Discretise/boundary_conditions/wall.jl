@define_boundary Wall Laplacian{Linear} begin
    # phi = term.phi 
    # (; area, delta, normal) = face 

    # U_boundary = SVector{3}(0.0,0.0,0.0) # user given vector

    # velocity_diff = phi[cellID] .- U_boundary
    # J = term.flux[fID]
    # norm_vel = (velocity_diff⋅normal)*normal

    # flux = J*area/delta
    # ap = term.sign[1]*(-flux)
    
    # ap, ap*(U_boundary[component.value] + norm_vel[component.value])

    # # version 2

    # phi = term.phi 
    # vc = phi[cellID]
    # vb = SVector{3}(0.0,0.0,0.0) # user given vector
    # J = term.flux[fID]
    # (; area, delta, normal) = face 

    # flux = J*area/delta
    # ap = term.sign[1]*(-flux)
    
    # vc_n = (vc⋅normal)*normal
    # vcc = vc[component.value]
    # # vc_p = vcc/(vcc - vc_n[component.value])# factor defining cevc[component.value]ntroid contribution
    # vc_p = (1 - vc_n[component.value]) # factor defining centroid contribution
    # vb_n = (vb⋅normal)*normal
    # vb_p = vb - vb_n # parallel component of given boundary vector
   
    # vc_p*ap, ap*(vb_p[component.value]) #+ vc_n[component.value])

    # version 3

    phi = term.phi 
    vc = phi[cellID]
    vb = SVector{3}(0.0,0.0,0.0) # user given vector
    J = term.flux[fID]
    (; area, delta, normal) = face 

    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    vc_n = (vc⋅normal)*normal
    vc_p = (vc - vc_n) # factor defining centroid contribution
    vb_n = (vb⋅normal)*normal
    vb_p = (vb - vb_n) # parallel component of given boundary vector
   
    ap, ap*(vb_p[component.value] + vc_n[component.value])
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