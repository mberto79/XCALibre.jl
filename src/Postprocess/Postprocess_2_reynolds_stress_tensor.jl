export ReynoldsStress
@kwdef struct ReynoldsStress{T<:AbstractField,T2<:AbstractField,S<:AbstractString}
    field::T 
    name::S
    mean::T
    mean_sq::T2
    rs::T2
    start::Real
    stop::Real
    update_interval::Real
end  
"""
    ReynoldsStress(
    #required arguments
    model.momentum.U; 

    #optional keyword arguments
    start::Real,
    stop::Real,
    update_interval::Real)
Constructor to allocate memory to store the Reynolds Stress Tensor over the calculation window. Once created, should be passed to the `Configuration` object as an argument with keyword `postprocess`

## Input arguments 
- `field`, must be `model.momentum.U`

## Optional arguments
- `start::Real` optional keyword which specifies the start of the Reynolds Stress Tensor calculation window, for **steady** simulations, this is in **iterations**, for **transient** simulations it is in **flow time**.   
- `stop::Real` optional keyword which specifies the end iteration/time of the Reynolds Stress Tensor calculation window. Default value is the last iteration/timestep. 
- `update_interval::Real` optional keyword which specifies how often the Reynolds Stress Tensor is updated and stored (default value is 1 i.e Reynolds Stress Tensor updates every timestep/iteration). Note that the frequency of writing the post-processed fields is specified by the `write_interval` in `Configuration`. 
"""
function ReynoldsStress(field; name::String =  "Reynolds_Stress", start::Real=1, stop::Real=typemax(Int),update_interval::Real=1)
    start > 0      || throw(ArgumentError("Start must be a positive value (got $start)"))
    stop  >= start  || throw(ArgumentError("Stop ($stop) must be greater than or equal to start ($start)"))
    update_interval > 0 || throw(ArgumentError("update interval must be >0 (got $update_interval)"))
    if field isa VectorField
        rs = SymmetricTensorField(field.mesh)
        mean = VectorField(field.mesh)
        mean_sq = SymmetricTensorField(field.mesh)
    else
        throw(ArgumentError("Unsupported field type: $(typeof(field))"))
    end
    return  ReynoldsStress(field=field, name=name, rs=rs, mean=mean, mean_sq=mean_sq, start=start, stop=stop, update_interval=update_interval)
end

function runtime_postprocessing!(RS::ReynoldsStress{T,T2,S},iter::Integer,n_iterations::Integer) where {T<:VectorField,T2<:SymmetricTensorField,S}
    if must_calculate(RS,iter,n_iterations)
        current_field = RS.field
        n = div(iter - RS.start,RS.update_interval) + 1
        _update_running_mean!(RS.mean.x.values, current_field.x.values, n)
        _update_running_mean!(RS.mean.y.values, current_field.y.values, n)
        _update_running_mean!(RS.mean.z.values, current_field.z.values, n)
    

        _update_running_mean!(RS.mean_sq.xx.values, current_field.x.values .^2,n)
        _update_running_mean!(RS.mean_sq.xy.values, current_field.x.values .* current_field.y.values,n)
        _update_running_mean!(RS.mean_sq.xz.values, current_field.x.values .* current_field.z.values,n)
        _update_running_mean!(RS.mean_sq.yy.values, current_field.y.values .^2,n)
        _update_running_mean!(RS.mean_sq.yz.values, current_field.y.values .* current_field.z.values,n)
        _update_running_mean!(RS.mean_sq.zz.values, current_field.z.values .^2,n)

        @. RS.rs.xx.values = RS.mean_sq.xx.values - RS.mean.x.values^2
        @. RS.rs.xy.values = RS.mean_sq.xy.values - RS.mean.x.values * RS.mean.y.values
        @. RS.rs.xz.values = RS.mean_sq.xz.values - RS.mean.x.values * RS.mean.z.values

        @. RS.rs.yy.values = RS.mean_sq.yy.values - RS.mean.y.values^2
        @. RS.rs.yz.values = RS.mean_sq.yz.values - RS.mean.y.values * RS.mean.z.values

        @. RS.rs.zz.values = RS.mean_sq.zz.values - RS.mean.z.values^2
    end
    return nothing
end

function convert_time_to_iterations(RS::ReynoldsStress, model,dt,iterations)
    if model.time === Transient()
        start = Int(ceil(RS.start / dt))
        stop = Int(min(RS.stop,dt*iterations) / dt )
        update_interval = max(1, Int(floor(RS.update_interval / dt)))
        update_interval >= 1 || throw(ArgumentError("update interval must be ≥1 (got $update_interval)"))
        stop >= start || throw(ArgumentError("After conversion with dt=$dt the RMS calculation window is empty (start = $start, stop = $stop)"))
        return ReynoldsStress(field=RS.field,name=RS.name,mean=RS.mean,mean_sq=RS.mean_sq,rs = RS.rs, start=start,stop=stop,update_interval=update_interval)
    else
        isinteger(RS.start) && isinteger(RS.stop) && isinteger(RS.update_interval) || throw(ArgumentError("For steady runs, start/stop/update_interval must be integers."))

        return RS
    end
end