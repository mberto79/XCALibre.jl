export calculate_field_property!
export FieldAverage

@kwdef struct FieldAverage{S<:AbstractString,T<:AbstractScalarField,I<:Integer}
    fieldname::S
    average::T
    start::I
    stop::I
end

function FieldAverage(mesh::M; fieldname::AbstractString,start::Integer=1, stop::Integer=typemax(Int)) where {M<:Mesh2}
    @assert fieldname in ("Ux", "Uy", "Uz") "Unsupported averaging tag $field"
    start  > 0      || throw(ArgumentError("Start iteration must be strictly positive (got $start)"))
    stop > 0      || throw(ArgumentError("Stop iteration must be strictly positive (got $stop)"))
    stop > start  || throw(ArgumentError("Stop ($stop) must be greater than start ($start)"))
    storage = ScalarField(mesh)
    return FieldAverage(fieldname=fieldname, average=storage, start=start, stop=stop)
end


#methods for the calculate_field_property function
#internal helper; shared arithmetic
function _update_running_mean!(stored_field_vals, current_vals, n)
    a = 1.0 / n 
    b = 1.0 - a
    @. stored_field_vals = b * stored_field_vals + a * current_vals 
    return nothing 
end

function _update_over_averaging_window!(f::FieldAverage, current_vals, iter::Integer, n_iterations::Integer)
    eff_stop = min(f.stop, n_iterations)
    if iter >= f.start && iter <= eff_stop
        n = iter - f.start + 1
        _update_running_mean!(f.average.values, current_vals, n)
    end
    return nothing
end

#make sure iter and iterations are made consistent
#When a vector is passed, broadcast the function over all elements in the vector
function calculate_field_property!(f::Vector, model,iter::Integer,n_iterations::Integer)
    calculate_field_property!.(f::Vector,Ref(model),Ref(iter),Ref(n_iterations))
end

function calculate_field_property!(f::FieldAverage,model,iter::Integer,n_iterations::Integer)
    #Need to check which string is contained in the FieldAverage fieldname 
    fieldname = f.fieldname
    if fieldname == "Ux"
        field = model.momentum.U.x.values
    elseif fieldname == "Uy"
        field = model.momentum.U.y.values
    elseif fieldname == "Uz"
        field = model.momentum.U.z.values
    end
    _update_over_averaging_window!(f,field,iter,n_iterations)
end

function calculate_field_property!(f::NamedTuple{()}, model, iter::Integer,n_iterations)
    return nothing
end
