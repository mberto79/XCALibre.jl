export FieldRMS
@kwdef struct FieldRMS{T<:AbstractField,S<:String,I}
    field::T 
    name::S
    mean::T
    mean_sq::T
    rms::T
    start::I
    stop::I
    write_interval::I
end  
"""
    FieldRMS(
    #required arguments
    field;
    name::String,

    #optional keyword arguments
    start::Integer,
    stop::Integer,
    write_interval::Integer)
Constructor to allocate memory to store the root mean square of the fluctuations of a field over the averaging window (in terms of iterations). Once created, should be passed to the `Configuration` object as an argument with keyword `postprocess`

## Input arguments 
- `field` the `VectorField` or `ScalarField`, e.g , `model.momentum.U`.
- `name::String` the name/label of the field to be averaged, e.g "U_mean", this is used only when exporting to .vtk format


## Optional arguments
- `start::Integer` optional keyword which specifies the start iteration of the averaging window. Default value is 1. 
- `stop::Integer` optional keyword which specifies the end iteration of the averaging window. Default value is typemax(Int) (i.e just an arbitrarily large number). 
- `write_interval::Integer` optional keyword which specifies how often the averaged field is updated and stored in solver iterations (default value is 1). 
"""
function FieldRMS(field;name::String,start::Integer=1,stop::Integer=typemax(Int),write_interval::Integer=1)
    start > 0      || throw(ArgumentError("Start iteration must be a positive value (got $start)"))
    stop  > 0      || throw(ArgumentError("Stop iteration must be a positive value (got $stop)"))
    stop  >= start  || throw(ArgumentError("Stop iteration($stop) must be greater than start ($start) iteration"))
    write_interval >= 1 || throw(ArgumentError("write interval must be ≥1 (got $write_interval)")) 
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
    return FieldRMS(field=field,name=name,mean=mean,mean_sq=mean_sq,rms=rms,start=start,stop=stop,write_interval=write_interval)
end



function calculate_field_property!(f::FieldRMS,iter::Integer,n_iterations::Integer)
    _update_RMS!(f,f.field,iter,n_iterations)
    return ((f.name,f.rms),)
end

#this updates the values stored in the RMS struct depending on the type of field that is passed to it
function _update_RMS!(f::FieldRMS, current_field::ScalarField, iter::Integer, n_iterations::Integer)
    eff_stop = min(f.stop, n_iterations)
    if iter >= f.start && iter <= eff_stop && (mod(iter - f.start, f.write_interval) == 0)
        n = div(iter - f.start,f.write_interval) + 1
        _update_running_mean!(f.mean.values, current_field.values, n)
        _update_running_mean!(f.mean_sq.values, current_field.values .^2 ,n)

        u_mean  = f.mean.values
        uu_mean = f.mean_sq.values
        z = zero(eltype(f.rms.values))
        @. f.rms.values = sqrt(max(uu_mean - u_mean^2, z))
    end
    return nothing
end
function _update_RMS!(f::FieldRMS, current_field::VectorField, iter::Integer, n_iterations::Integer)
    eff_stop = min(f.stop, n_iterations)
    if iter >= f.start && iter <= eff_stop && (mod(iter - f.start, f.write_interval) == 0)
        n = div(iter - f.start,f.write_interval) + 1
        _update_running_mean!(f.mean.x.values, current_field.x.values, n)
        _update_running_mean!(f.mean_sq.x.values, current_field.x.values .^2,n)
        _update_running_mean!(f.mean.y.values, current_field.y.values, n)
        _update_running_mean!(f.mean_sq.y.values, current_field.y.values .^2,n)
        _update_running_mean!(f.mean.z.values, current_field.z.values, n)
        _update_running_mean!(f.mean_sq.z.values, current_field.z.values .^2,n)

        z = zero(eltype(f.rms.x.values))
        @. f.rms.x.values = sqrt(max(f.mean_sq.x.values - f.mean.x.values^2, z)) 
        @. f.rms.y.values = sqrt(max(f.mean_sq.y.values - f.mean.y.values^2, z)) 
        @. f.rms.z.values = sqrt(max(f.mean_sq.z.values - f.mean.z.values^2, z)) 
        
    end
    return nothing
end