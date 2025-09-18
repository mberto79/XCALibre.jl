export FieldRMS
@kwdef struct FieldRMS{T<:AbstractField,N}
    mean::T
    mean_sq::T
    path::NTuple{N,Symbol}
    rms::T
    start::Integer
    stop::Integer
    write_interval::Integer
end  
#this needs to be updated so that the field rms is computed every write_interval
#This creates the memory for the FieldRMS 
"""
    FieldRMS(model, path; start::Integer,stop::Integer,write_interval::Integer)
Constructor to allocate memory to store the root mean square of the fluctuations of a field over the averaging window (in terms of iterations). 

# Input arguments 
- `model` the `Physics` model object needs to be passed to allocate the right amount of memory
- `path` tuple of symbols e.g `(:momentum,:U)` which are used to access the correct field to average
- `start::Integer` optional keyword which specifies the start of the averaging window. Default value is 1. 
- `stop::Integer` optional keyword which specifies the end of the averaging window. Default value is typemax(Int) (i.e just an arbitrarily large number). 
- `write_interval::Integer` optional keyword which specifies how often the root mean square of the field fluctuations is updated and stored in solver iterations (default value is 1). 

"""
function FieldRMS(model,path;start::Integer=1,stop::Integer=typemax(Int),write_interval::Integer=1)
    start > 0      || throw(ArgumentError("Start iteration must be a positive value (got $start)"))
    stop  > 0      || throw(ArgumentError("Stop iteration must be a positive value (got $stop)"))
    stop  > start  || throw(ArgumentError("Stop iteration($stop) must be greater than start ($start) iteration"))
    write_interval >= 1 || throw(ArgumentError("write interval must be ≥1 (got $write_interval)"))
    #Check that the field is actually supported 
    field = get_field_from_path(model,path)
    if field isa ScalarField
        mean = ScalarField(model.domain)
        mean_sq = ScalarField(model.domain)
        rms = ScalarField(model.domain)
    elseif field isa VectorField
        mean = VectorField(model.domain)
        mean_sq = VectorField(model.domain)
        rms = VectorField(model.domain)
    else
    end
    return FieldRMS(mean=mean,mean_sq=mean_sq,path=path,rms=rms,start=start,stop=stop,write_interval=write_interval)
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
function calculate_field_property!(f::FieldRMS,model,iter::Integer,n_iterations::Integer)
    path = f.path
    field = get_field_from_path(model,path)
    _update_RMS!(f,model,field,iter,n_iterations)
    # _update_over_averaging_window!(f,field,iter,n_iterations)
end

#this updates the values stored in the RMS struct depending on the type of field that is passed to it
function _update_RMS!(f::FieldRMS,model, current_field::ScalarField, iter::Integer, n_iterations::Integer)
    eff_stop = min(f.stop, n_iterations)
    #create memory for a squared version of the field
    current_field_sq = ScalarField(model.domain)
    current_field_sq.values .= current_field.values .^2
    if iter >= f.start && iter <= eff_stop && (mod(iter - f.start, f.write_interval) == 0)
        n = iter - f.start + 1
        _update_running_mean!(f.mean.values, current_field.values, n)
        _update_running_mean!(f.mean_sq.values, current_field_sq.values,n)

        if iter == eff_stop
            u_mean = f.mean.values
            uu_mean = f.mean_sq.values
            z = zero(eltype(f.rms.values))
            @. f.rms.values = sqrt(max(uu_mean - u_mean^2, z)) 
        end
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