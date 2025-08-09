export calculate_field_average!
export FieldAverage
export FieldRMS
struct FieldAverage{FLAG,T<:AbstractScalarField,I<:Integer}
    field::T
    start::I
    finish::I
end

struct FieldRMS{FLAG,T<:AbstractScalarField,I<:Integer}
    mean::T
    mean_sq::T
    rms::T
    start::I
    finish::I
end  


# base constructor used by all the convenience wrappers
function FieldAverage{FLAG}(field::T,start::I,finish::I) where {FLAG,T<:AbstractScalarField,I<:Integer}
    @assert FLAG in (:Ux, :Uy, :Uz) "Unsupported averaging tag $FLAG"
    start  > 0      || throw(ArgumentError("start must be strictly positive (got $start)"))
    finish > start  || throw(ArgumentError("finish ($finish) must be greater than start ($start)"))
    FieldAverage{FLAG,T,I}(field, start, finish)
end

#convenience wrapper for when just a field is passed
function FieldAverage{FLAG}(field::T) where {FLAG,T<:AbstractScalarField} 
    FieldAverage{FLAG}(field, 1, typemax(Int))
end
# convenience wrapper 2 – only a mesh
function FieldAverage{FLAG}(mesh::M) where {FLAG,M<:Mesh2}
    field = ScalarField(mesh)
    FieldAverage{FLAG}(field, 1, typemax(Int))
end
#convenience wrapper 3 - mesh and start and finish 
function FieldAverage{FLAG}(mesh::M, start::I, finish::I) where
        {FLAG,M<:Mesh2,I<:Integer}
    field = ScalarField(mesh)
    FieldAverage{FLAG}(field, start, finish)
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


#methods for the calculate_field_average function
#internal helper; shared arithmetic
function _update_running_mean!(field_vals, current_vals, n)
    a = 1.0 / n 
    b = 1.0 - a
    @. field_vals = b * field_vals + a * current_vals 
    return nothing 
end

function _update_over_averaging_window!(f::FieldAverage, current_vals, iter::Integer, n_iterations::Integer)
    eff_finish = min(f.finish, n_iterations)
    if iter >= f.start && iter <= eff_finish
        n = iter - f.start + 1
        _update_running_mean!(f.field.values, current_vals, n)
    end
    return nothing
end
# specialised entry points — one tiny method per component

function calculate_field_average!(f::FieldAverage{:Ux}, model, iter::Integer, n_iterations::Integer)
    _update_over_averaging_window!(f, model.momentum.U.x.values, iter, n_iterations)
end
function calculate_field_average!(f::FieldAverage{:Uy}, model, iter::Integer, n_iterations::Integer)
    _update_over_averaging_window!(f, model.momentum.U.y.values, iter, n_iterations)
end
function calculate_field_average!(f::FieldAverage{:Uz}, model, iter::Integer, n_iterations::Integer)
    _update_over_averaging_window!(f, model.momentum.U.z.values, iter, n_iterations)
end
function calculate_field_average!(f::NamedTuple{()}, model, iter::Integer,n_iterations)
    return nothing
end


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

function calculate_field_average!(f::FieldRMS{:Ux},model,iter,n_iterations)
    _update_RMS!(f,model.momentum.U.x.values,model.momentum.U.x.values .^2, iter,n_iterations)
end
function calculate_field_average!(f::FieldRMS{:Uy},model,iter,n_iterations)
    _update_RMS!(f,model.momentum.U.y.values,model.momentum.U.y.values .^2, iter,n_iterations)
end
function calculate_field_average!(f::FieldRMS{:Uz},model,iter,n_iterations)
    _update_RMS!(f,model.momentum.U.z.values,model.momentum.U.z.values .^2, iter,n_iterations)
end

