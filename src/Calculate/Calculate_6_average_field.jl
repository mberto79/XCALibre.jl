export calculate_field_property!
export FieldAverage

struct FieldAverage{FLAG,T<:AbstractScalarField,I<:Integer}
    field::T
    start::I
    finish::I
end

# base constructor used by all the convenience wrappers
function FieldAverage{FLAG}(field::T,start::I,finish::I) where {FLAG,T<:AbstractScalarField,I<:Integer}
    @assert FLAG in (:Ux, :Uy, :Uz) "Unsupported averaging tag $FLAG"
    start  > 0      || throw(ArgumentError("start must be strictly positive (got $start)"))
    finish > start  || throw(ArgumentError("finish ($finish) must be greater than start ($start)"))
    FieldAverage{FLAG,T,I}(field, start, finish)
end

#convenience wrapper for when just a field is passed
function FieldAverage{FLAG}(field::T) where {FLAG,T<:AbstractScalarField} 
    FieldAverage{FLAG}(field, 1, typemax(Int))
end
# convenience wrapper 2 – only a mesh
function FieldAverage{FLAG}(mesh::M) where {FLAG,M<:Mesh2}
    field = ScalarField(mesh)
    FieldAverage{FLAG}(field, 1, typemax(Int))
end
#convenience wrapper 3 - mesh and start and finish 
function FieldAverage{FLAG}(mesh::M, start::I, finish::I) where
        {FLAG,M<:Mesh2,I<:Integer}
    field = ScalarField(mesh)
    FieldAverage{FLAG}(field, start, finish)
end

#methods for the calculate_field_property function
#internal helper; shared arithmetic
function _update_running_mean!(field_vals, current_vals, n)
    a = 1.0 / n 
    b = 1.0 - a
    @. field_vals = b * field_vals + a * current_vals 
    return nothing 
end

function _update_over_averaging_window!(f::FieldAverage, current_vals, iter::Integer, n_iterations::Integer)
    eff_finish = min(f.finish, n_iterations)
    if iter >= f.start && iter <= eff_finish
        n = iter - f.start + 1
        _update_running_mean!(f.field.values, current_vals, n)
    end
    return nothing
end
# specialised entry points — one tiny method per component

function calculate_field_property!(f::FieldAverage{:Ux}, model, iter::Integer, n_iterations::Integer)
    _update_over_averaging_window!(f, model.momentum.U.x.values, iter, n_iterations)
end
function calculate_field_property!(f::FieldAverage{:Uy}, model, iter::Integer, n_iterations::Integer)
    _update_over_averaging_window!(f, model.momentum.U.y.values, iter, n_iterations)
end
function calculate_field_property!(f::FieldAverage{:Uz}, model, iter::Integer, n_iterations::Integer)
    _update_over_averaging_window!(f, model.momentum.U.z.values, iter, n_iterations)
end
function calculate_field_property!(f::NamedTuple{()}, model, iter::Integer,n_iterations)
    return nothing
end
