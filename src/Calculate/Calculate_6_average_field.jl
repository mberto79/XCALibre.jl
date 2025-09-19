export calculate_field_property!
export FieldAverage
@kwdef struct FieldAverage{T<:AbstractField,S<:String,I}
    field::T
    name::S
    mean::T
    start::I
    stop::I
    write_interval::I
end

"""
    FieldAverage(
    #required arguments
    field;
    name::String,

    #optional keyword arguments
    start::Integer,
    stop::Integer,
    write_interval::Integer)

Constructor to allocate memory to store the averaged field over the averaging window (in terms of iterations). Once created, should be passed to the `Configuration` object as an argument with keyword `postprocess`

## Input arguments 
- `field` the `VectorField` or `ScalarField` to be averaged, e.g , `model.momentum.U`.
- `name::String` the name of the field to be averaged, e.g "U_mean", this is used only when exporting to .vtk format

## Optional arguments
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
    return FieldAverage(field=field,name=name,mean=storage,start=start,stop=stop,write_interval=write_interval)
end

function calculate_field_property!(f::FieldAverage,iter::Integer,n_iterations::Integer) 
    _update_over_averaging_window!(f,f.field,iter,n_iterations)
    return ((f.name,f.mean),)
end
function calculate_field_property!(f::Vector,iter::Integer,n_iterations::Integer)
    vector_of_tuples = calculate_field_property!.(f,Ref(iter),Ref(n_iterations))
    return Tuple(first.(vector_of_tuples))
end

calculate_field_property(::Nothing,::Integer,::Integer) = nothing

function _update_over_averaging_window!(f::FieldAverage,current_field::VectorField,iter::Integer,n_iterations::Integer)
    eff_stop = min(f.stop, n_iterations)
    if iter >= f.start && iter <= eff_stop && (mod(iter - f.start, f.write_interval) == 0)
        n = div(iter - f.start,f.write_interval) + 1
            _update_running_mean!(f.mean.x.values,current_field.x.values,n)
            _update_running_mean!(f.mean.y.values,current_field.y.values,n)
            _update_running_mean!(f.mean.z.values,current_field.z.values,n)
    end
    return nothing 
end

function _update_over_averaging_window!(f::FieldAverage,field::ScalarField,iter::Integer,n_iterations::Integer)
    current_field = field
    eff_stop = min(f.stop, n_iterations)
    if iter >= f.start && iter <= eff_stop && (mod(iter - f.start, f.write_interval) == 0)
        n = div(iter - f.start,f.write_interval) + 1
            _update_running_mean!(f.mean.values,current_field.values,n)
    end
    return nothing 
end

"""
    _update_running_mean!(stored_field_vals, current_vals, n)
Internal helper: updates `stored_field_vals` **in place** to be the running mean after `n` samples, using the latest values in `current_vals`
"""
function _update_running_mean!(stored_field_vals, current_vals, n)
    a = 1.0 / n 
    b = 1.0 - a
    @. stored_field_vals = b * stored_field_vals + a * current_vals 
    return nothing 
end