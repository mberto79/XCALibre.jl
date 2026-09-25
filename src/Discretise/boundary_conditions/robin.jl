export Robin

@kwdef struct RobinValue{F}
    a::F
    b::F
    value::F
end
Adapt.@adapt_structure RobinValue

"""
    Robin <: AbstractBoundary

Robin (mixed) boundary condition `a·φ + b·∇φ·n = value` for scalar fields. `a=1, b=0` recovers `Dirichlet` and `a=0, b=1` recovers `Neumann`. Not supported by the density-based (Godunov) solvers.

# Inputs
- `ID` Name of the boundary given as a symbol (e.g. :inlet). Internally it gets replaced with the boundary index ID
- `a` coefficient of the boundary value (default 1)
- `b` coefficient of the face normal gradient (default 0)
- `value` right-hand side of the constraint (default 0)

# Example
    Robin(:wall, a=1.0, b=0.5, value=10.0)
"""
struct Robin{I,V,R<:UnitRange} <: AbstractBoundary
    ID::I
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Robin

Robin(name::Symbol; a=1.0, b=0.0, value=0.0) = begin
    iszero(a) && iszero(b) && throw(ArgumentError("Robin(:$name): `a` and `b` cannot both be zero"))
    Robin(name, RobinValue(promote(a, b, value)...), 0:0)
end

adapt_value(value::RobinValue, mesh) = begin
    F = _get_float(mesh)
    RobinValue(F(value.a), F(value.b), F(value.value))
end

# φf = (value·δ + b·φP)/(a·δ + b), so ∇φ·n = (value - a·φP)/(a·δ + b)
@define_boundary Robin Laplacian{Linear} ScalarField begin
    J = term.flux[fID]
    (; area, delta) = face
    (; a, b, value) = bc.value
    coeff = J*area/(a*delta + b)
    ap = term.sign*(-coeff*a)
    bp = term.sign*(-coeff*value)
    ap, bp
end

@define_boundary Robin Divergence{Linear} ScalarField begin
    (; delta) = face
    (; a, b, value) = bc.value
    ap = term.sign*(term.flux[fID])/(a*delta + b)
    ap*b, -ap*value*delta
end

@define_boundary Robin Divergence{Upwind} ScalarField begin
    (; delta) = face
    (; a, b, value) = bc.value
    ap = term.sign*(term.flux[fID])/(a*delta + b)
    ap*b, -ap*value*delta
end

@define_boundary Robin Divergence{LUST} ScalarField begin
    (; delta) = face
    (; a, b, value) = bc.value
    ap = term.sign*(term.flux[fID])/(a*delta + b)
    ap*b, -ap*value*delta
end

# Bounded = upwind boundary with -Sp(div phi): subtract ap from the diagonal
@define_boundary Robin Divergence{BoundedUpwind} ScalarField begin
    (; delta) = face
    (; a, b, value) = bc.value
    ap = term.sign*(term.flux[fID])
    apf = ap/(a*delta + b)
    apf*b - ap, -apf*value*delta
end

@define_boundary Robin Si ScalarField begin
    0.0, 0.0
end
