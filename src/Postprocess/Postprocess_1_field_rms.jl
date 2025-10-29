export FieldRMS
@kwdef struct FieldRMS{T<:AbstractField,S<:AbstractString}
    field::T 
    name::S
    mean::T
    mean_sq::T
    rms::T
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end  
"""
    FieldRMS(
    #required arguments
    field;
    name::String,

    #optional keyword arguments
    start::Real,
    stop::Real,
    update_interval::Real)
Constructor to allocate memory to store the root mean square of the fluctuations of a field over the averaging window. Once created, should be passed to the `Configuration` object as an argument with keyword `postprocess`

## Input arguments 
- `field` the `VectorField` or `ScalarField`, e.g , `model.momentum.U`.
- `name::String` the name/label of the field, e.g "U_rms", this is used only when exporting to .vtk format


## Optional arguments
- `start::Real` optional keyword which specifies the start of the RMS calculation window, for **steady** simulations, this is in **iterations**, for **transient** simulations it is in **flow time**.   
- `stop::Real` optional keyword which specifies the end iteration/time of the RMS calculation window. Default value is the last iteration/timestep. 
- `update_interval::Real` optional keyword which specifies how often the RMS of the field is updated and stored (default value is 1 i.e RMS updates every timestep/iteration). Note that the frequency of writing the post-processed fields is specified by the `write_interval` in `Configuration`. 
"""
function FieldRMS(field; name::AbstractString, start::Union{Real,Nothing}=nothing, stop::Union{Real,Nothing}=nothing,update_interval::Union{Real,Nothing}=nothing)
    if field isa ScalarField
        mean = ScalarField(field.mesh)
        mean_sq = ScalarField(field.mesh)
        rms = ScalarField(field.mesh)
    elseif field isa VectorField
        mean = VectorField(field.mesh)
        mean_sq = VectorField(field.mesh)
        rms = VectorField(field.mesh)
    else
        throw(ArgumentError("Unsupported field type: $(typeof(field))"))
    end
    return  FieldRMS(field=field,name=name,mean=mean,mean_sq=mean_sq,rms=rms,start=start,stop=stop,update_interval=update_interval)
end


function runtime_postprocessing!(RMS::FieldRMS{T,S},iter::Integer,n_iterations::Integer) where {T<:ScalarField,S}
    if must_calculate(RMS,iter,n_iterations)
        current_field = RMS.field
        n = div(iter - RMS.start,RMS.update_interval) + 1
        _update_running_mean!(RMS.mean.values, current_field.values, n)
        _update_running_mean!(RMS.mean_sq.values, current_field.values .^2 ,n)

        u_mean  = RMS.mean.values
        uu_mean = RMS.mean_sq.values
        z = zero(eltype(RMS.rms.values))
        @. RMS.rms.values = sqrt(max(uu_mean - u_mean^2, z))
    end
    return nothing 
end

function runtime_postprocessing!(RMS::FieldRMS{T,S},iter::Integer,n_iterations::Integer) where {T<:VectorField,S}
    if must_calculate(RMS,iter,n_iterations)
        current_field = RMS.field
        n = div(iter - RMS.start,RMS.update_interval) + 1
        _update_running_mean!(RMS.mean.x.values, current_field.x.values, n)
        _update_running_mean!(RMS.mean_sq.x.values, current_field.x.values .^2,n)
        _update_running_mean!(RMS.mean.y.values, current_field.y.values, n)
        _update_running_mean!(RMS.mean_sq.y.values, current_field.y.values .^2,n)
        _update_running_mean!(RMS.mean.z.values, current_field.z.values, n)
        _update_running_mean!(RMS.mean_sq.z.values, current_field.z.values .^2,n)

        z = zero(eltype(RMS.rms.x.values))
        @. RMS.rms.x.values = sqrt(max(RMS.mean_sq.x.values - RMS.mean.x.values^2, z)) 
        @. RMS.rms.y.values = sqrt(max(RMS.mean_sq.y.values - RMS.mean.y.values^2, z)) 
        @. RMS.rms.z.values = sqrt(max(RMS.mean_sq.z.values - RMS.mean.z.values^2, z)) 
        
    end
    return nothing
end

function convert_time_to_iterations(RMS::FieldRMS, model,dt,iterations)
    if model.time === Transient()
        if RMS.start === nothing
            start = 1
        else 
            RMS.start >= 0  || throw(ArgumentError("Start must be a positive value (got $(RMS.start))"))
            start = clamp(ceil(Int, RMS.start / dt), 1, iterations) 
        end

        if RMS.stop === nothing 
            stop = iterations
        else
            RMS.stop ≥ 0 || throw(ArgumentError("stop must be ≥ 0 (got $(RMS.stop))"))
            stop = clamp(floor(Int,RMS.stop / dt), 1, iterations)
        end

        if RMS.update_interval === nothing 
            update_interval = 1
        else
            RMS.update_interval > 0 || throw(ArgumentError("update interval must be > 0 (got $(RMS.update_interval))"))
            update_interval = max(1, floor(Int,RMS.update_interval / dt))
        end
        stop >= start || throw(ArgumentError("After conversion with dt=$dt the averaging window is empty (start = $start, stop = $stop)"))
        return FieldRMS(field=RMS.field,name=RMS.name,mean=RMS.mean,mean_sq=RMS.mean_sq,rms = RMS.rms, start=start,stop=stop,update_interval=update_interval)

    else #for Steady runs use iterations 
        if RMS.start === nothing
            start = 1
        else 
            RMS.start isa Integer || throw(ArgumentError("For steady runs, start must be specified in iterations and therefore be an integer (got $(RMS.start))"))
            RMS.start >=1     || throw(ArgumentError("Start must be ≥1 (got $(RMS.start))"))
            start = RMS.start
        end

        if RMS.stop === nothing 
            stop = iterations
        else
            RMS.stop isa Integer || throw(ArgumentError("For steady runs, stop must be specified in iterations and therefore be an integer (got $(RMS.stop))"))
            RMS.stop >=1     || throw(ArgumentError("Stop must be ≥1 (got $(RMS.stop))"))
            stop = RMS.stop
        end

        if RMS.update_interval === nothing 
            update_interval = 1
        else
            RMS.update_interval isa Integer || throw(ArgumentError("For steady runs, update_interval must be specified in iterations and therefore be an integer (got $(RMS.update_interval))"))
            RMS.update_interval >= 1 || throw(ArgumentError("update interval must be ≥1 (got $(RMS.update_interval))"))
            update_interval = RMS.update_interval
        end

        stop >= start || throw(ArgumentError("stop iteration needs to be ≥ start  (got start = $start, stop = $stop)"))
        stop <= iterations || throw(ArgumentError("stop ($stop) must be ≤ iterations ($iterations)"))
        return FieldRMS(field=RMS.field,name=RMS.name,mean=RMS.mean,mean_sq=RMS.mean_sq,rms = RMS.rms, start=start,stop=stop,update_interval=update_interval)
    end
end