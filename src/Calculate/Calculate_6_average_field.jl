export calculate_field_property!
export FieldAverage
export get_field_from_path #check if this actually has to be exported because it might not have to be 
@kwdef struct FieldAverage{T<:AbstractField,S<:String,I}
    field::T
    mean::T
    name::S
    start::I
    stop::I
    write_interval::I
end
#new implementation is FieldAverage(model.momentum.U,:U)
"""
    FieldAverage(model, path; start::Integer,stop::Integer,write_interval::Integer)
Constructor to allocate memory to store the averaged field over the averaging window (in terms of iterations). 

# Input arguments 
- `field` the `VectorField` or `ScalarField` to be averaged, e.g , `model.momentum.U`.
- `name::String` the name of the field to be averaged, e.g "U"
- `start::Integer` optional keyword which specifies the start iteration of the averaging window. Default value is 1. 
- `stop::Integer` optional keyword which specifies the end iteration of the averaging window. Default value is typemax(Int) (i.e just an arbitrarily large number). 
- `write_interval::Integer` optional keyword which specifies how often the averaged field is updated and stored in solver iterations (default value is 1). 
"""
function FieldAverage(field;name::String,start::Integer=1,stop::Integer=typemax(Int),write_interval::Integer=1)
    start > 0      || throw(ArgumentError("Start iteration must be a positive value (got $start)"))
    stop  > 0      || throw(ArgumentError("Stop iteration must be a positive value (got $stop)"))
    stop  > start  || throw(ArgumentError("Stop iteration($stop) must be greater than start ($start) iteration"))
    write_interval >= 1 || throw(ArgumentError("write interval must be ≥1 (got $write_interval)"))
    if field isa ScalarField
        storage = ScalarField(field.mesh) 
    elseif field isa VectorField
        storage = VectorField(field.mesh)
    else
        throw(ArgumentError("Unsupported field type: $(typeof(field))"))
    end
    return FieldAverage(field=field,mean=storage,name=name,start=start,stop=stop,write_interval=write_interval)
end



function calculate_field_property!(f::FieldAverage,iter::Integer,n_iterations::Integer)# add a write interval 

    _update_over_averaging_window!(f,iter,n_iterations)
    return f.name,f.mean
end
function calculate_field_property!(f::Vector, model,iter::Integer,n_iterations::Integer)
    calculate_field_property!.(f::Vector,Ref(model),Ref(iter),Ref(n_iterations))
end

function calculate_field_property!(nothing,iter::Integer,n_iterations::Integer)
    return nothing
end
"""
    _update_over_averaging_window!(f::FieldAverage, cur::VectorField,
                                   iter::Integer, n_iterations::Integer) -> nothing
    _update_over_averaging_window!(f::FieldAverage, cur::ScalarField,
                                   iter::Integer, n_iterations::Integer) -> nothing

Internal helper: conditionally updates the running mean stored in `f.field`
for the current iteration. The effective averaging window is

    eff_stop = min(f.stop, n_iterations)
    iter ∈ [f.start, eff_stop]

If `iter` lies in this inclusive window, the routine computes

    n = iter - f.start + 1

and performs an in-place running-mean update of `f.field` using
`_update_running_mean!`. For `VectorField`, the update is applied
component-wise (`x`, `y`, `z`); for `ScalarField`, to `values`.

Input arguments
- `f`: `FieldAverage` accumulator holding the destination field and window.
- `cur`: the current source field (scalar or vector) read from the model.
- `iter`: current solver iteration (1-based).
- `n_iterations`: total number of iterations for the current run.

"""



function _update_over_averaging_window!(f::FieldAverage,iter::Integer,n_iterations::Integer)
    current_field = f.field
    eff_stop = min(f.stop, n_iterations)
    if iter >= f.start && iter <= eff_stop && (mod(iter - f.start, f.write_interval) == 0)
        n = iter - f.start + 1
            _update_running_mean!(f.mean.x.values,current_field.x.values,n)
            _update_running_mean!(f.mean.y.values,current_field.y.values,n)
            _update_running_mean!(f.mean.z.values,current_field.z.values,n)
    end
    return nothing 
end

function _update_over_averaging_window!(f::FieldAverage, current_field::ScalarField,iter::Integer,n_iterations::Integer)
    eff_stop = min(f.stop, n_iterations)
    if iter >= f.start && iter <= eff_stop && (mod(iter - f.start, f.write_interval) == 0)
        n = iter - f.start + 1
            _update_running_mean!(f.field.values,current_field.values,n)
    end
    return nothing 
end

"""
    _update_running_mean!(stored_field_vals, current_vals, n)
Internal helper: updates `stored_field_vals` **in place** to be the running mean after `n` samples, using the latest values in `current_vals`
"""
# #internal helper; shared arithmetic
function _update_running_mean!(stored_field_vals, current_vals, n)
    a = 1.0 / n 
    b = 1.0 - a
    @. stored_field_vals = b * stored_field_vals + a * current_vals 
    return nothing 
end
