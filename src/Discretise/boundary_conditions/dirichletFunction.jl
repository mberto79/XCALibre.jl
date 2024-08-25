export DirichletFunction

struct DirichletFunction{I,V} <: AbstractBoundary
    ID::I
    value::V
end
Adapt.@adapt_structure DirichletFunction

function fixedValue(BC::DirichletFunction, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return DirichletFunction{I,typeof(value)}(ID, value)
        # Exception 2: value is a tupple
    elseif V <: Function
        return DirichletFunction{I,V}(ID, value)
    # Error if value is not scalar or tuple
    else
        throw("The value provided should be a scalar or a tuple")
    end
end


@define_boundary DirichletFunction Laplacian{Linear} begin
    J = term.flux[fID]
    (; area, delta, centre) = face 
    flux = J*area/delta
    ap = term.sign*(-flux)
    value = bc.value(centre, time)[component.value]
    ap, ap*value
end

@define_boundary DirichletFunction Divergence{Linear} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    value = bc.value(face.centre, time)[component.value]
    0.0, ap*value
end

@define_boundary DirichletFunction Divergence{Upwind} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    value = bc.value(face.centre, time)[component.value]
    0.0, ap*value
end

@define_boundary DirichletFunction Divergence{BoundedUpwind} begin
    flux = -term.flux[fID]
    ap = term.sign*(flux)
    value = bc.value(face.centre, time)[component.value]
    flux, ap*value
end

@define_boundary DirichletFunction Si begin
    0.0, 0.0
end
