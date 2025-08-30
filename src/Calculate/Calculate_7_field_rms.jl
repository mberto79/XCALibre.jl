export FieldRMS
@kwdef struct FieldRMS{T<:AbstractField,N}
    mean::T
    mean_sq::T
    path::NTuple{N,Symbol}
    rms::T
    start::Integer
    stop::Integer
end  

#This creates the memory for the FieldRMS 
function FieldRMS(model,path;start::Integer=1,stop::Integer=typemax(Int))
    start > 0      || throw(ArgumentError("Start iteration must be a positive value (got $start)"))
    stop  > 0      || throw(ArgumentError("Stop iteration must be a positive value (got $stop)"))
    stop  > start  || throw(ArgumentError("Stop iteration($stop) must be greater than start ($start) iteration"))
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
    return FieldRMS(mean=mean,mean_sq=mean_sq,path=path,rms=rms,start=start,stop=stop)
end
# specialised entry points — one tiny method per component
#This updates the RMS from inside the solver loop 
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
    if iter >= f.start && iter <= eff_stop
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
    if iter >= f.start && iter <= eff_stop
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