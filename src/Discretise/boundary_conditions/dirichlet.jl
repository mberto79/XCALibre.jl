@define_boundary Dirichlet Laplacian{Linear} begin
    J = term.flux[fID]
    (; area, delta) = face 
    flux = J*area/delta
    ap = term.sign*(-flux)
    ap, ap*bc.value
end

@define_boundary Dirichlet Divergence{Linear} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    0.0, ap*bc.value
end

@define_boundary Dirichlet Divergence{Upwind} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    0.0, ap*bc.value
end

@define_boundary Dirichlet Divergence{LUST} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    0.0, ap*bc.value
end

@define_boundary Dirichlet Divergence{BoundedUpwind} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    flux, ap*bc.value
end

@define_boundary Dirichlet Si begin
    0.0, 0.0
end
