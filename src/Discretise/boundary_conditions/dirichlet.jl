@define_boundary Dirichlet Si begin
    0.0, 0.0
end

@define_boundary Dirichlet Laplacian{Linear} begin
    J = term.flux[fID]
    (; area, delta) = face 

    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    ap, ap*bc.value
end

@define_boundary Dirichlet Divergence{Linear} begin
    0.0, term.sign[1]*(-term.flux[fID]*bc.value)
end

@define_boundary Dirichlet Divergence{BoundedUpwind} begin

    ap = term.sign[1]*(term.flux[fID])
    vol = 1#cell.volume

    -term.flux[fID], -ap*bc.value
end

@define_boundary Dirichlet Divergence{Upwind} begin
    ap = term.sign[1]*(term.flux[fID])
    0.0, -ap*bc.value
end