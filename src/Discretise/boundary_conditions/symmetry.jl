@define_boundary Symmetry Laplacian{Linear} begin
    (; area, delta, normal) = face 
    phi = term.phi 
    J = term.flux[fID]
    flux = 2*J*area/delta
    ap = term.sign[1]*(-flux)

    vc = phi[cellID]
    vn = (vc⋅normal)*normal
    vp = vc - vn
    ap, ap*vp[component.value]
end

# To-do: Add scala scalar variants of Wall BC in next version (currently using Neumann)

@define_boundary Symmetry Divergence{Linear} begin
    ap = term.sign[1]*(term.flux[fID])
    0.0, 0.0
end

@define_boundary Symmetry Divergence{Upwind} begin
    ap = term.sign[1]*(term.flux[fID])
    0.0, 0.0
end