export calculate_field_average!
export FieldAverage

struct FieldAverage{T<:AbstractScalarField,S<:Symbol,I<:Integer}
    field::T
    flag::S
    start::I
    finish::I
end


function calculate_field_average!(field_accumulator::FieldAverage,model,iteration)
    tag = field_accumulator.flag
        if tag === :Ux
        current_field = model.momentum.U.x.values

    elseif tag === :Uy
        current_field = model.momentum.U.y.values

    elseif tag === :Uz
        current_field = model.momentum.U.z.values
    else
        throw(ArgumentError("unknown averaging tag $tag"))
    end
    if field_accumulator.start <= iteration && iteration <= field_accumulator.finish
        n = iteration - field_accumulator.start + 1
        running_mean = (n-1)/n .* field_accumulator.field.values .+ current_field ./n
        field_accumulator.field.values .= running_mean
    end
    return nothing 
end

# function FieldAverage(field::T) where {T<:AbstractScalarField}
#     FieldAverage{T,Int}(field, 1, typemax(Int))
# end

# function FieldAverage(field::T, start::I, finish::I) where {T<:AbstractScalarField, I<:Integer}
#     if start < 1
#         throw(ArgumentError("`start` must be ≥ 1, got $start"))
#     end
#     if finish <= start
#         throw(ArgumentError("`finish` must be > start (=$start), got $finish"))
#     end
#     FieldAverage{T,I}(field, start, finish)
# end

# function FieldAverage(field, args...)
#     throw(ArgumentError("`field` must be a subtype of AbstractScalarField, got $(typeof(field))"))
# end
# function FieldAverage(fieldname::S,model) where {S<:AbstractString}
#     field = string_into_field(fieldname,model)
#     FieldAverage{T,Int}(field, 1, typemax(Int))
# end

# function FieldAverage(field::T, start::I, finish::I) where {T<:AbstractScalarField, I<:Integer}
#     if start < 1
#         throw(ArgumentError("`start` must be ≥ 1, got $start"))
#     end
#     if finish <= start
#         throw(ArgumentError("`finish` must be > start (=$start), got $finish"))
#     end
#     FieldAverage{T,I}(field, start, finish)
# end

# function FieldAverage(field, args...)
#     throw(ArgumentError("`field` must be a subtype of AbstractScalarField, got $(typeof(field))"))
# end



# function parse_path(path::AbstractString)
#     parts = Symbol.(split(path, '.'))      # e.g. (:model, :momentum, :U, :x)
#     return Tuple(parts[2:end])             # -> (:momentum, :U, :x)
# end
# function make_accessor(path::AbstractString)
#     parts = parse_path(path)
#     # Closure that remembers `parts` and looks up the field each call
#     return model -> foldl(getproperty, parts; init = model)
# end
# function string_into_field(path::AbstractString,model)
#     get_values = make_accessor(path)
#     field = get_values(model)
# end
