export FieldRMS
@kwdef struct FieldRMS{T<:AbstractField,S<:String,I}
    field::T 
    name::S
    mean::T
    mean_sq::T
    rms::T
    start::I
    stop::I
    save_interval::I
end  
"""
    FieldRMS(
    #required arguments
    field;
    name::String,

    #optional keyword arguments
    start::Integer,
    stop::Integer,
    save_interval::Integer)
Constructor to allocate memory to store the root mean square of the fluctuations of a field over the averaging window (in terms of iterations). Once created, should be passed to the `Configuration` object as an argument with keyword `postprocess`

## Input arguments 
- `field` the `VectorField` or `ScalarField`, e.g , `model.momentum.U`.
- `name::String` the name/label of the field to be averaged, e.g "U_rms", this is used only when exporting to .vtk format


## Optional arguments
- `start::Integer` optional keyword which specifies the start iteration of the averaging window. Default value is 1. 
- `stop::Integer` optional keyword which specifies the end iteration of the averaging window. Default value is the final iteration. 
- `save_interval::Integer` optional keyword which specifies how often the averaged field is updated and stored in solver iterations (default value is 1). 
"""
function FieldRMS(field;name::String,start::Integer=1,stop::Integer=typemax(Int),save_interval::Integer=1)
    start > 0      || throw(ArgumentError("Start iteration must be a positive value (got $start)"))
    stop  >= start  || throw(ArgumentError("Stop iteration($stop) must be greater than or equal to start ($start) iteration"))
    save_interval >= 1 || throw(ArgumentError("save interval must be ≥1 (got $save_interval)")) 
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
    return  FieldRMS(field=field,name=name,mean=mean,mean_sq=mean_sq,rms=rms,start=start,stop=stop,save_interval=save_interval)
end



function calculate_postprocessing!(RMS::FieldRMS,iter::Integer,n_iterations::Integer)
    _update_RMS!(RMS,RMS.field,iter,n_iterations)
    return ((RMS.name,RMS.rms),)
end

#this updates the values stored in the RMS struct depending on the type of field that is passed to it
function _update_RMS!(RMS::FieldRMS, current_field::ScalarField, iter::Integer, n_iterations::Integer)
    eff_stop = min(RMS.stop, n_iterations)
    if iter >= RMS.start && iter <= eff_stop && (mod(iter - RMS.start, RMS.save_interval) == 0)
        n = div(iter - RMS.start,RMS.save_interval) + 1
        _update_running_mean!(RMS.mean.values, current_field.values, n)
        _update_running_mean!(RMS.mean_sq.values, current_field.values .^2 ,n)

        u_mean  = RMS.mean.values
        uu_mean = RMS.mean_sq.values
        z = zero(eltype(RMS.rms.values))
        @. RMS.rms.values = sqrt(max(uu_mean - u_mean^2, z))
    end
    return nothing
end
function _update_RMS!(RMS::FieldRMS, current_field::VectorField, iter::Integer, n_iterations::Integer)
    eff_stop = min(RMS.stop, n_iterations)
    if iter >= RMS.start && iter <= eff_stop && (mod(iter - RMS.start, RMS.save_interval) == 0)
        n = div(iter - RMS.start,RMS.save_interval) + 1
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