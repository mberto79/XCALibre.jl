@define_boundary Wall Laplacian{Linear} begin
    (; area, delta, normal) = face 
    phi = term.phi 
    J = term.flux[fID]
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    vb = SVector{3}(0.0,0.0,0.0) # do not hard-code in next version
    vc = phi[cellID]
    vc_n = (vc⋅normal)*normal
    vb_n = (vb⋅normal)*normal
    vb_p = (vb - vb_n) # parallel component of given boundary vector
   
    ap, ap*(vb_p[component.value] + vc_n[component.value])
end

# To-do: Add scala scalar variants of Wall BC in next version (currently using Neumann)

@define_boundary Wall Divergence{Linear} begin
    0.0, 0.0
end

@define_boundary Wall Divergence{Upwind} begin
    0.0, 0.0
end

@define_boundary Wall Divergence{BoundedUpwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    -flux, 0.0
end