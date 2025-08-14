export calculate_field_property!
export FieldAverage

@kwdef struct FieldAverage{T<:AbstractField,I<:Integer}
    field::T
    label::Symbol
    start::I
    stop::I
end


#Need to generalise the implementation to handle things like model.turbulence etc 

function FieldAverage(model_momentum,symbol,start::Integer=1,stop::Integer=typemax(Int))
    start > 0      || throw(ArgumentError("Start iteration must be a positive value (got $start)"))
    stop  > 0      || throw(ArgumentError("Stop iteration must be a positive value (got $stop)"))
    stop  > start  || throw(ArgumentError("Stop iteration($stop) must be greater than start ($start) iteration"))
    #Check that the field is actually supported 
    field = getproperty(model_momentum,symbol)
    if field isa ScalarField
        storage = ScalarField(field.mesh)  # Example constructor
    elseif field isa VectorField
        storage = VectorField(field.mesh)
    else
    end
    return FieldAverage(field=storage,label=symbol,start=start,stop=stop)
end

#decided on FieldAverage(model.momentum,:=;U)

#this functions job is to extract the correct property from 
function calculate_field_property!(f::FieldAverage,model,iter::Integer,n_iterations::Integer)
    #Need to check which string is contained in the FieldAverage fieldname 
    label = f.label
    field = getproperty(model.momentum,label)
    _update_over_averaging_window!(f,field,iter,n_iterations)
end

function _update_over_averaging_window!(f::FieldAverage, current_field::VectorField,iter::Integer,n_iterations::Integer)
    eff_stop = min(f.stop, n_iterations)
    if iter >= f.start && iter <= eff_stop
        n = iter - f.start + 1
            _update_running_mean!(f.field.x.values,current_field.x.values,n)
            _update_running_mean!(f.field.y.values,current_field.y.values,n)
            _update_running_mean!(f.field.z.values,current_field.z.values,n)
    end
    return nothing 
end


function _update_over_averaging_window!(f::FieldAverage, current_field::ScalarField,iter::Integer,n_iterations::Integer)
    eff_stop = min(f.stop, n_iterations)
    if iter >= f.start && iter <= eff_stop
        n = iter - f.start + 1
            _update_running_mean!(f.field.values,current_field.values,n)
    end
    return nothing 
end

#Need to generalise the implementation to handle things like model.turbulence etc 

# #internal helper; shared arithmetic
function _update_running_mean!(stored_field_vals, current_vals, n)
    a = 1.0 / n 
    b = 1.0 - a
    @. stored_field_vals = b * stored_field_vals + a * current_vals 
    return nothing 
end

function calculate_field_property!(f::Vector, model,iter::Integer,n_iterations::Integer)
    calculate_field_property!.(f::Vector,Ref(model),Ref(iter),Ref(n_iterations))
end

function calculate_field_property!(f::NamedTuple{()}, model, iter::Integer,n_iterations)
    return nothing
end