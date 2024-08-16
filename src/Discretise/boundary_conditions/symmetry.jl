# Symmetry functor definition - Moukalled et al. 2016 Implementation of Boundary conditions in the finite-volume pressure-based method - Part 1
# http://dx.doi.org/10.1080/10407790.2016.1138748

@define_boundary Symmetry Laplacian{Linear} begin
    phi = term.phi 
    # velocity_cell = phi[cellID]
    vc = phi[cellID]
    J = term.flux[fID]
    (; area, delta, normal) = face 

    # norm_vel= (velocity_cell⋅normal)
    # norm_vel = norm_vel - velocity_cell[component.value]*normal[component.value]
    # flux = J*area/delta
    # ap = term.sign[1]*(-flux)

    # (2.0)*ap*normal[component.value]*normal[component.value]^2, (2.0)*ap*(norm_vel*normal[component.value])

    flux = 2*J*area/delta
    ap = term.sign[1]*(-flux)
    vn = (vc⋅normal)*normal
    vp = vc - vn
    ap, ap*vp[component.value]
end

@define_boundary Symmetry Divergence{Linear} begin
    ap = term.sign[1]*(term.flux[fID])
    0.0, 0.0
end

@define_boundary Symmetry Divergence{Upwind} begin
    ap = term.sign[1]*(term.flux[fID])
    0.0, 0.0
end