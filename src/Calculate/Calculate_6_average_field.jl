export calculate_field_average!
export FieldAverage

struct FieldAverage{FLAG,T<:AbstractScalarField,I<:Integer}
    field::T
    start::I
    finish::I
end
function FieldAverage{FLAG}(field::T, start::I, finish::I) where
        {FLAG,T<:AbstractScalarField,I<:Integer}
    @assert FLAG in (:Ux, :Uy, :Uz) "Unsupported averaging tag $FLAG"
    FieldAverage{FLAG,T,I}(field, start, finish)
end

function FieldAverage{FLAG}(field::T) where
        {FLAG,T<:AbstractScalarField}
    @assert FLAG in (:Ux, :Uy, :Uz) "Unsupported averaging tag $FLAG"
    FieldAverage{FLAG}(field, 1, typemax(Int))
end

function FieldAverage{FLAG}(mesh::M) where
        {FLAG,M<:Mesh2}
    @assert FLAG in (:Ux, :Uy, :Uz) "Unsupported averaging tag $FLAG"
    field = ScalarField(mesh)
    FieldAverage{FLAG}(field, 1, typemax(Int))
end

# function FieldAverage(fieldname::S,model) where {S<:AbstractString}
#     field = string_into_field(fieldname,model)
#     FieldAverage{T,Int}(field, 1, typemax(Int))
# end


#internal helper; shared arithmetic
@inline function _update_values!(field_vals, current_vals, n)
    @. field_vals = ((n - 1) / n) * field_vals + current_vals / n
end

# specialised entry points — one tiny method per component
@inline function calculate_field_average!(fa::FieldAverage{:Ux}, model, iter::Integer)
    n = iter - fa.start + 1
    _update_values!(fa.field.values, model.momentum.U.x.values, n)
end

@inline function calculate_field_average!(fa::FieldAverage{:Uy}, model, iter::Integer)
    n = iter - fa.start + 1
    _update_values!(fa.field.values, model.momentum.U.y.values, n)
end

@inline function calculate_field_average!(fa::FieldAverage{:Uz}, model, iter::Integer)
    n = iter - fa.start + 1
    _update_values!(fa.field.values, model.momentum.U.z.values, n)
end