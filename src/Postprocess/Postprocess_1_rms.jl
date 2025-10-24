export RMS
@kwdef struct RMS{T<:AbstractField,S<:String}
    field::T 
    name::S
    mean::T
    mean_sq::T
    rms::T
    start::Real
    stop::Real
    update_interval::Real
end  
"""
    RMS(
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
function RMS(field; name::AbstractString, start::Real=1, stop::Real=typemax(Int),update_interval::Real=1)
    start > 0      || throw(ArgumentError("Start must be a positive value (got $start)"))
    stop  >= start  || throw(ArgumentError("Stop ($stop) must be greater than or equal to start ($start)"))
    update_interval > 0 || throw(ArgumentError("update interval must be >0 (got $update_interval)"))
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
    return  RMS(field=field,name=name,mean=mean,mean_sq=mean_sq,rms=rms,start=start,stop=stop,update_interval=update_interval)
end


function runtime_postprocessing!(_RMS::RMS{T,S},iter::Integer,n_iterations::Integer) where {T<:ScalarField,S}
    if must_calculate(_RMS,iter,n_iterations)
        current_field = _RMS.field
        n = div(iter - _RMS.start,_RMS.update_interval) + 1
        _update_running_mean!(_RMS.mean.values, current_field.values, n)
        _update_running_mean!(_RMS.mean_sq.values, current_field.values .^2 ,n)

        u_mean  = _RMS.mean.values
        uu_mean = _RMS.mean_sq.values
        z = zero(eltype(_RMS.rms.values))
        @. _RMS.rms.values = sqrt(max(uu_mean - u_mean^2, z))
    end
    return nothing 
end

function runtime_postprocessing!(_RMS::RMS{T,S},iter::Integer,n_iterations::Integer) where {T<:VectorField,S}
    if must_calculate(_RMS,iter,n_iterations)
        current_field = _RMS.field
        n = div(iter - _RMS.start,_RMS.update_interval) + 1
        _update_running_mean!(_RMS.mean.x.values, current_field.x.values, n)
        _update_running_mean!(_RMS.mean_sq.x.values, current_field.x.values .^2,n)
        _update_running_mean!(_RMS.mean.y.values, current_field.y.values, n)
        _update_running_mean!(_RMS.mean_sq.y.values, current_field.y.values .^2,n)
        _update_running_mean!(_RMS.mean.z.values, current_field.z.values, n)
        _update_running_mean!(_RMS.mean_sq.z.values, current_field.z.values .^2,n)

        z = zero(eltype(_RMS.rms.x.values))
        @. _RMS.rms.x.values = sqrt(max(_RMS.mean_sq.x.values - _RMS.mean.x.values^2, z)) 
        @. _RMS.rms.y.values = sqrt(max(_RMS.mean_sq.y.values - _RMS.mean.y.values^2, z)) 
        @. _RMS.rms.z.values = sqrt(max(_RMS.mean_sq.z.values - _RMS.mean.z.values^2, z)) 
        
    end
    return nothing
end

function convert_time_to_iterations(_RMS::RMS, model,dt,iterations)
    if model.time === Transient()
        start = Int(ceil(_RMS.start / dt))
        stop = Int(min(_RMS.stop,dt*iterations) / dt )
        update_interval = max(1, Int(floor(_RMS.update_interval / dt)))
        update_interval >= 1 || throw(ArgumentError("update interval must be ≥1 (got $update_interval)"))
        stop >= start || throw(ArgumentError("After conversion with dt=$dt the RMS calculation window is empty (start = $start, stop = $stop)"))
        return _RMS(field=_RMS.field,name=_RMS.name,mean=_RMS.mean,mean_sq=_RMS.mean_sq,rms = _RMS.rms, start=start,stop=stop,update_interval=update_interval)
    else
        isinteger(_RMS.start) && isinteger(_RMS.stop) && isinteger(_RMS.update_interval) || throw(ArgumentError("For steady runs, start/stop/update_interval must be integers."))

        return _RMS
    end
end