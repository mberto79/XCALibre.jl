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
#this needs to be updated so that the field rms is computed every write_interval
"""
    FieldRMS(model; name::String, start::Integer,stop::Integer,write_interval::Integer)
Constructor to allocate memory to store the root mean square of the fluctuations of a field over the averaging window (in terms of iterations). 

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
    stop  > start  || throw(ArgumentError("Stop iteration($stop) must be greater than start ($start) iteration"))
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




# specialised entry points — one tiny method per component
#This updates the RMS from inside the solver loop 
"""
    calculate_field_property!(f::NamedTuple,  model, iter::Integer, n_iterations::Integer) -> nothing

Internal helper used inside solver loops (PISO / SIMPLE / CPISO, etc.).

Updates the running mean stored in `f.field` for the target field at
`model.(f.path...)` **only if** the current iteration `iter` lies within the
inclusive averaging window `fa.start : min(fa.stop, n_iterations)`. 

"""
function calculate_field_property!(f::FieldRMS,iter::Integer,n_iterations::Integer)
    _update_RMS!(f,f.field,iter,n_iterations)
    return ((f.name,f.rms),)
end

#this updates the values stored in the RMS struct depending on the type of field that is passed to it
function _update_RMS!(f::FieldRMS, current_field::ScalarField, iter::Integer, n_iterations::Integer)
    eff_stop = min(f.stop, n_iterations)
    #create memory for a squared version of the field
    # current_field_sq = ScalarField(model.domain)
    # current_field_sq.values .= current_field.values .^2
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
function _update_RMS!(f::FieldRMS,model, current_field::VectorField, iter::Integer, n_iterations::Integer)
    eff_stop = min(f.stop, n_iterations)
    #create memory for a squared version of the field
    current_field_sq = VectorField(model.domain)
    current_field_sq.x.values .= current_field.x.values .^2
    current_field_sq.y.values .= current_field.y.values .^2
    current_field_sq.z.values .= current_field.z.values .^2
    if iter >= f.start && iter <= eff_stop && (mod(iter - f.start, f.write_interval) == 0)
        n = iter - f.start + 1
        #ADD the x y z versions of this and it should be done 
        _update_running_mean!(f.mean.x.values, current_field.x.values, n)
        _update_running_mean!(f.mean_sq.x.values, current_field_sq.x.values,n)
        _update_running_mean!(f.mean.y.values, current_field.y.values, n)
        _update_running_mean!(f.mean_sq.y.values, current_field_sq.y.values,n)
        _update_running_mean!(f.mean.z.values, current_field.z.values, n)
        _update_running_mean!(f.mean_sq.z.values, current_field_sq.z.values,n)

        if iter == eff_stop
            u_x_mean = f.mean.x.values
            uu_x_mean = f.mean_sq.x.values
            u_y_mean = f.mean.y.values
            uu_y_mean = f.mean_sq.y.values
            u_z_mean = f.mean.z.values
            uu_z_mean = f.mean_sq.z.values
            z = zero(eltype(f.rms.x.values))
            @. f.rms.x.values = sqrt(max(uu_x_mean - u_x_mean^2, z)) 
            @. f.rms.y.values = sqrt(max(uu_y_mean - u_y_mean^2, z)) 
            @. f.rms.z.values = sqrt(max(uu_z_mean - u_z_mean^2, z)) 
        end
    end
    return nothing
end