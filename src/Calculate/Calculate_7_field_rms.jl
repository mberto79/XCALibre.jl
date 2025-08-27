export FieldRMS
@kwdef struct FieldRMS{T<:AbstractField,I<:Integer}
    mean::T
    mean_sq::T
    label::Symbol
    rms::T
    start::I
    stop::I
end  
#This creates the memory for the FieldRMS 
function FieldRMS(model_momentum,symbol,start::Integer=1,stop::Integer=typemax(Int))
    start > 0      || throw(ArgumentError("Start iteration must be a positive value (got $start)"))
    stop  > 0      || throw(ArgumentError("Stop iteration must be a positive value (got $stop)"))
    stop  > start  || throw(ArgumentError("Stop iteration($stop) must be greater than start ($start) iteration"))
    #Check that the field is actually supported 
    field = getproperty(model_momentum,symbol)
    if field isa ScalarField
        mean = ScalarField(field.mesh)
        mean_sq = ScalarField(field.mesh)
        rms = ScalarField(field.mesh)
    elseif field isa VectorField
        mean = VectorField(field.mesh)
        mean_sq = VectorField(field.mesh)
        rms = VectorField(field.mesh)
    else
    end
    return FieldRMS(mean=mean,mean_sq=mean_sq,label=symbol,rms=rms,start=start,stop=stop)
end
# specialised entry points — one tiny method per component
#This updates the RMS from inside the solver loop 
function calculate_field_property!(f::FieldRMS,model,iter::Integer,n_iterations::Integer)
    label = f.label
    field = getproperty(model.momentum,label)
    _update_RMS!(f,field,model,field,iter,n_iterations)
    # _update_over_averaging_window!(f,field,iter,n_iterations)
end

#this updates the values stored in the RMS struct depending on the type of field that is passed to it

function _update_RMS!(f::FieldRMS,model, current_field::ScalarField, iter::Integer, n_iterations::Integer)
    eff_finish = min(f.finish, n_iterations)
    #create memory for a squared version of the field
    current_field_sq = ScalarField(current_field.mesh)
    current_field_sq.values .= current_field.values
    if iter >= f.start && iter <= eff_finish
        n = iter - f.start + 1
        _update_running_mean!(f.mean.values, current_field, n)
        _update_running_mean!(f.mean_sq.values, current_field_sq,n)

        if iter == eff_finish
            u_mean = f.mean.values
            uu_mean = f.mean_sq.values
            z = zero(eltype(f.rms.values))
            @. f.rms.values = sqrt(max(uu_mean - u_mean^2, z)) 
        end
    end
    return nothing
end