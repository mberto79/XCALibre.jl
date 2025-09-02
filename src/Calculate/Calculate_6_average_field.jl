export calculate_field_property!
export FieldAverage
export get_field_from_path #check if this actually has to be exported because it might not have to be 
@kwdef struct FieldAverage{T<:AbstractField,N}
    field::T
    path::NTuple{N,Symbol}
    start::Integer
    stop::Integer
    write_interval::Integer
end
"""
    FieldAverage(model, path; start::Integer,stop::Integer,write_interval::Integer)
Constructor to allocate memory to store the averaged field over the averaging window (in terms of iterations). 

# Input arguments 
- `model` the `Physics` model object needs to be passed to allocate the right amount of memory
- `path` tuple of symbols e.g `(:momentum,:U)` which are used to access the correct field to average
- `start::Integer` optional keyword which specifies the start iteration of the averaging window. Default value is 1. 
- `stop::Integer` optional keyword which specifies the end iteration of the averaging window. Default value is typemax(Int) (i.e just an arbitrarily large number). 
- `write_interval::Integer` optional keyword which specifies how often the averaged field is updated and stored in solver iterations (default value is 1). 
"""
function FieldAverage(model,path;start::Integer=1,stop::Integer=typemax(Int),write_interval::Integer=1)
    start > 0      || throw(ArgumentError("Start iteration must be a positive value (got $start)"))
    stop  > 0      || throw(ArgumentError("Stop iteration must be a positive value (got $stop)"))
    stop  > start  || throw(ArgumentError("Stop iteration($stop) must be greater than start ($start) iteration"))
    write_interval >= 1 || throw(ArgumentError("write interval must be ≥1 (got $write_interval)"))
    #Check that the field is actually supported 
    field = get_field_from_path(model,path)
    if field isa ScalarField
        storage = ScalarField(model.domain)  # Example constructor
    elseif field isa VectorField
        storage = VectorField(model.domain)
    else
    end
    return FieldAverage(field=storage,path=path,start=start,stop=stop,write_interval=write_interval)
end

"""
    get_field_from_path(model, path::NTuple{N,Symbol}) where {N}

Internal helper: follows `path` on `model` by repeatedly calling `getproperty`
and returns the nested object. For example,

    get_field_from_path(model, (:momentum, :U)) === model.momentum.U

# Input arguments
- `model`: root object to traverse.
- `path`: tuple of property names to follow (in order).

# Returns
The Scalar or Vector field object referenced by `model.(path...)`.

"""
function get_field_from_path(model,path::NTuple{N,Symbol}) where {N}
    acc = model 
    for name in path
        acc = getproperty(acc,name)
    end
    return acc
end


"""
    calculate_field_property!(f::FieldAverage, model, iter::Integer, n_iterations::Integer) -> nothing
    calculate_field_property!(f::AbstractVector, model, iter::Integer, n_iterations::Integer) -> nothing
    calculate_field_property!(f::NamedTuple,  model, iter::Integer, n_iterations::Integer) -> nothing

Internal helper used inside solver loops (PISO / SIMPLE / CPISO, etc.).

Updates the running mean stored in `f.field` for the target field at
`model.(f.path...)` **only if** the current iteration `iter` lies within the
inclusive averaging window `fa.start : min(fa.stop, n_iterations)`. 

"""

function calculate_field_property!(f::FieldAverage,model,iter::Integer,n_iterations::Integer)# add a write interval 
    path = f.path
    field = get_field_from_path(model,path)
    _update_over_averaging_window!(f,field,iter,n_iterations)
end
function calculate_field_property!(f::Vector, model,iter::Integer,n_iterations::Integer)
    calculate_field_property!.(f::Vector,Ref(model),Ref(iter),Ref(n_iterations))
end

function calculate_field_property!(f::NamedTuple{()}, model, iter::Integer,n_iterations)
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



function _update_over_averaging_window!(f::FieldAverage, current_field::VectorField,iter::Integer,n_iterations::Integer)
    eff_stop = min(f.stop, n_iterations)
    if iter >= f.start && iter <= eff_stop && (mod(iter - f.start, f.write_interval) == 0)
        n = iter - f.start + 1
            _update_running_mean!(f.field.x.values,current_field.x.values,n)
            _update_running_mean!(f.field.y.values,current_field.y.values,n)
            _update_running_mean!(f.field.z.values,current_field.z.values,n)
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

#Need to generalise the implementation to handle things like model.turbulence etc 


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
