export FieldRMS
struct FieldRMS{FLAG,T<:AbstractScalarField,I<:Integer}
    mean::T
    mean_sq::T
    rms::T
    start::I
    finish::I
end  


function FieldRMS{FLAG}(mean::T,mean_sq::T,rms::T,start::I,finish::I) where {FLAG,T<:AbstractScalarField,I<:Integer}
    @assert FLAG in (:Ux, :Uy, :Uz) "Unsupported averaging tag $FLAG"
    start  > 0      || throw(ArgumentError("start must be strictly positive (got $start)"))
    finish > start  || throw(ArgumentError("finish iteration($finish) must be greater than start iteration ($start)"))
    FieldRMS{FLAG,T,I}(mean,mean_sq,rms,start,finish)
end
#When just a mesh is supplied 
function FieldRMS{FLAG}(mesh::M) where {FLAG,M<:Mesh2}
    field1 = ScalarField(mesh)
    field2 = ScalarField(mesh)
    field3 = ScalarField(mesh)
    FieldRMS{FLAG}(field1,field2,field3,1,typemax(Int))
end
#When a mesh is supplied with an averaging window 
function FieldRMS{FLAG}(mesh::M, start::I, finish::I) where {FLAG,M<:Mesh2,I<:Integer}
    field1 = ScalarField(mesh)
    field2 = ScalarField(mesh)
    field3 = ScalarField(mesh)
    FieldRMS{FLAG}(field1,field2,field3,start,finish)
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

