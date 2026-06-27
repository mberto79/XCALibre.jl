export Robin

@kwdef struct RobinValue{F}
    a::F
    b::F
    value::F
end
Adapt.@adapt_structure RobinValue

struct Robin{I,V,R<:UnitRange} <: AbstractBoundary
    ID::I
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Robin

Robin(name::Symbol; a=1.0, b=0.0, value=0.0) = begin
    Robin(name, RobinValue(a=a, b=b, value=value), 0:0)
end

adapt_value(value::RobinValue, mesh) = begin
    F = _get_float(mesh)
    RobinValue(F(value.a), F(value.b), F(value.value))
end

@define_boundary Robin Laplacian{Linear} begin
    J = term.flux[fID]
    (; area, delta) = face
    (; a, b, value) = bc.value
    denom = a*delta + b
    coeff = J*area/denom
    ap = term.sign*(-coeff*a)
    bp = term.sign*(-coeff*value)
    ap, bp
end

@define_boundary Robin Divergence{Linear} begin
    flux = -term.flux[fID]
    (; delta) = face
    (; a, b, value) = bc.value
    denom = a*delta + b
    ap = term.sign*(flux)
    ap*b/denom, ap*value*delta/denom
end

@define_boundary Robin Divergence{Upwind} begin
    flux = -term.flux[fID]
    (; delta) = face
    (; a, b, value) = bc.value
    denom = a*delta + b
    ap = term.sign*(flux)
    ap*b/denom, ap*value*delta/denom
end

@define_boundary Robin Divergence{LUST} begin
    flux = -term.flux[fID]
    (; delta) = face
    (; a, b, value) = bc.value
    denom = a*delta + b
    ap = term.sign*(flux)
    ap*b/denom, ap*value*delta/denom
end

@define_boundary Robin Si begin
    0.0, 0.0
end

@define_boundary Robin Time{SteadyState} begin
    0.0, 0.0
end

@define_boundary Robin Time{Euler} begin
    0.0, 0.0
end

@define_boundary Robin Divergence{BoundedUpwind} begin
    flux = -term.flux[fID]
    (; delta) = face
    (; a, b, value) = bc.value
    denom = a*delta + b
    ap = term.sign*(flux)
    ac = max(ap, 0.0)
    an = -max(-ap, 0.0)
    # phi_f = (value*delta + b*phi_P) / denom
    # Term is ac*phi_P + an*phi_f
    # = ac*phi_P + an*(value*delta + b*phi_P)/denom
    # = (ac + an*b/denom)*phi_P + an*value*delta/denom
    ac + an*b/denom, an*value*delta/denom
end
