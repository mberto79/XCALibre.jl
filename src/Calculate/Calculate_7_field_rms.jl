export FieldRMS
@kwdef struct FieldRMS{T<:AbstractScalarField,I<:Integer}
    mean::T
    mean_sq::T
    label::Symbol
    rms::T
    start::I
    finish::I
end  

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

function _update_RMS!(f::FieldRMS, current_field,current_field_sq, iter::Integer, n_iterations::Integer)
    eff_finish = min(f.finish, n_iterations)
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

function calculate_field_property!(f::FieldRMS{:Ux},model,iter,n_iterations)
    _update_RMS!(f,model.momentum.U.x.values,model.momentum.U.x.values .^2, iter,n_iterations)
end
function calculate_field_property!(f::FieldRMS{:Uy},model,iter,n_iterations)
    _update_RMS!(f,model.momentum.U.y.values,model.momentum.U.y.values .^2, iter,n_iterations)
end
function calculate_field_property!(f::FieldRMS{:Uz},model,iter,n_iterations)
    _update_RMS!(f,model.momentum.U.z.values,model.momentum.U.z.values .^2, iter,n_iterations)
end

