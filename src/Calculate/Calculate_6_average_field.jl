export calculate_field_average!
export FieldAverage

struct FieldAverage{FLAG,T<:AbstractScalarField,I<:Integer}
    field::T
    start::I
    finish::I
end

struct FieldRMS{FLAG,T<:AbstractScalarField,I<:Integer}
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




#internal helper; shared arithmetic
@inline function _update_values!(field_vals, current_vals, n)
    @. field_vals = ((n - 1) / n) * field_vals + current_vals / n
end

# specialised entry points — one tiny method per component
@inline function calculate_field_average!(f::FieldAverage{:Ux}, model, iter::Integer)
    n = iter - f.start + 1
    _update_values!(f.field.values, model.momentum.U.x.values, n)
end

@inline function calculate_field_average!(f::FieldAverage{:Uy}, model, iter::Integer)
    n = iter - f.start + 1
    _update_values!(f.field.values, model.momentum.U.y.values, n)
end

@inline function calculate_field_average!(f::FieldAverage{:Uz}, model, iter::Integer)
    n = iter - f.start + 1
    _update_values!(f.field.values, model.momentum.U.z.values, n)
end

@inline function calculate_field_average!(f::NamedTuple{()}, model, iter::Integer)
    return nothing
end