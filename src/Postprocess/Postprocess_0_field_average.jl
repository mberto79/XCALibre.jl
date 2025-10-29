export runtime_postprocessing!
export FieldAverage
export convert_time_to_iterations
@kwdef struct FieldAverage{T<:AbstractField,S<:AbstractString}
    field::T
    name::S
    mean::T
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end

"""
    FieldAverage(
    #required arguments
    field;
    name::String,

    #optional keyword arguments
    start::Real,
    stop::Real,
    update_interval::Real)

Constructor to allocate memory to store the time averaged field. Once created, should be passed to the `Configuration` object as an argument with keyword `postprocess`

## Input arguments 
- `field` the `VectorField` or `ScalarField` to be averaged, e.g , `model.momentum.U`.
- `name::String` the name of the field to be averaged, e.g "U_mean", this is used only when exporting to .vtk format

## Optional arguments
- `start::Real` optional keyword which specifies the start time/iteration of the averaging window, for **steady** simulations, this is in **iterations**, for **transient** simulations it is in **flow time**.   
- `stop::Real` optional keyword which specifies the end iteration/time of the averaging window. Default value is the last iteration/timestep. 
- `update_interval::Real` optional keyword which specifies how often the time average of the field is updated and stored (default value is 1 i.e RMS updates every timestep/iteration). Note that the frequency of writing the post-processed fields is specified by the `write_interval` in `Configuration`. 
"""
function FieldAverage(field; name::AbstractString, start::Union{Real,Nothing}=nothing, stop::Union{Real,Nothing}=nothing,update_interval::Union{Real,Nothing}=nothing)
    if field isa ScalarField
        storage = ScalarField(field.mesh) 
    elseif field isa VectorField
        storage = VectorField(field.mesh)
    else
        throw(ArgumentError("Unsupported field type: $(typeof(field))"))
    end
    return FieldAverage(field=field, name=name, mean=storage,start=start, stop=stop, update_interval=update_interval)
end

function runtime_postprocessing!(avg::FieldAverage{T,S},iter::Integer,n_iterations::Integer) where {T<:ScalarField,S}
    if must_calculate(avg,iter,n_iterations)
        n = div(iter - avg.start,avg.update_interval) + 1
        current_field = avg.field
        _update_running_mean!(avg.mean.values,current_field.values,n)
    end
    return nothing
end

function runtime_postprocessing!(avg::FieldAverage{T,S},iter::Integer,n_iterations::Integer) where {T<:VectorField,S}
    if must_calculate(avg,iter,n_iterations)
        n = div(iter - avg.start,avg.update_interval) + 1
        current_field = avg.field
            _update_running_mean!(avg.mean.x.values,current_field.x.values,n)
            _update_running_mean!(avg.mean.y.values,current_field.y.values,n)
            _update_running_mean!(avg.mean.z.values,current_field.z.values,n)
    end
    return nothing 
end

function runtime_postprocessing!(avg::Vector,iter::Integer,n_iterations::Integer)
    runtime_postprocessing!.(avg,Ref(iter),Ref(n_iterations))
    return nothing
end

runtime_postprocessing!(::Nothing,::Integer,::Integer) = ()


function _update_running_mean!(stored_field_vals, current_vals, n)
    a = 1.0 / n 
    b = 1.0 - a
    @. stored_field_vals = b * stored_field_vals + a * current_vals 
    return nothing 
end

function must_calculate(field_struct,iter::Integer,n_iterations::Integer)
    eff_stop = min(field_struct.stop, n_iterations)
    interval = field_struct.update_interval
    start = field_struct.start
    iter ∈ start:interval:eff_stop
end

function convert_time_to_iterations(avg::FieldAverage, model,dt,iterations)
    if model.time === Transient()
        if avg.start === nothing
            start = 1
        else 
            avg.start >= 0  || throw(ArgumentError("Start must be a positive value (got $(avg.start))"))
            start = clamp(ceil(Int, avg.start / dt), 1, iterations) 
        end

        if avg.stop === nothing 
            stop = iterations
        else
            avg.stop ≥ 0 || throw(ArgumentError("stop must be ≥ 0 (got $(avg.stop))"))
            stop = clamp(floor(Int,avg.stop / dt), 1, iterations)
        end

        if avg.update_interval === nothing 
            update_interval = 1
        else
            avg.update_interval > 0 || throw(ArgumentError("update interval must be > 0 (got $(avg.update_interval))"))
            update_interval = max(1, floor(Int,avg.update_interval / dt))
        end
        stop >= start || throw(ArgumentError("After conversion with dt=$dt the averaging window is empty (start = $start, stop = $stop)"))
        return FieldAverage(field=avg.field,name=avg.name,mean=avg.mean,start=start,stop=stop,update_interval=update_interval)

    else #for Steady runs use iterations 
        if avg.start === nothing
            start = 1
        else 
            avg.start isa Integer || throw(ArgumentError("For steady runs, start must be specified in iterations and therefore be an integer (got $(avg.start))"))
            avg.start >=1     || throw(ArgumentError("Start must be ≥1 (got $(avg.start))"))
            start = avg.start
        end

        if avg.stop === nothing 
            stop = iterations
        else
            avg.stop isa Integer || throw(ArgumentError("For steady runs, stop must be specified in iterations and therefore be an integer (got $(avg.stop))"))
            avg.stop >=1     || throw(ArgumentError("Stop must be ≥1 (got $(avg.stop))"))
            stop = avg.stop
        end

        if avg.update_interval === nothing 
            update_interval = 1
        else
            avg.update_interval isa Integer || throw(ArgumentError("For steady runs, update_interval must be specified in iterations and therefore be an integer (got $(avg.update_interval))"))
            avg.update_interval >= 1 || throw(ArgumentError("update interval must be ≥1 (got $(avg.update_interval))"))
            update_interval = avg.update_interval
        end

        stop >= start || throw(ArgumentError("stop iteration needs to be ≥ start  (got start = $start, stop = $stop)"))
        stop <= iterations || throw(ArgumentError("stop ($stop) must be ≤ iterations ($iterations)"))
        return FieldAverage(field=avg.field,name=avg.name,mean=avg.mean,start=start,stop=stop,update_interval=update_interval)
    end
end


function convert_time_to_iterations(avg::Vector, model,dt,iterations)
    convert_time_to_iterations.(avg::Vector, Ref(model),Ref(dt),Ref(iterations))
end

convert_time_to_iterations(::Nothing,model,dt,iterations) = nothing