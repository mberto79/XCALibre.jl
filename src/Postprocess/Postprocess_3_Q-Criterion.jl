export QCriterion
@kwdef struct QCriterion{T<:AbstractScalarField,T1,T2,V<:AbstractString}
    Q::T 
    S2::T1
    Ω2::T2
    name::V
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end  

function QCriterion(inputfield; name::String =  "Q-Criterion", start::Union{Real,Nothing}=nothing, stop::Union{Real,Nothing}=nothing,update_interval::Union{Real,Nothing}=nothing)
    if inputfield isa VectorField
        storage = ScalarField(inputfield.mesh)
        strain = ScalarField(inputfield.mesh)
        vorticity = ScalarField(inputfield.mesh)
    else
        throw(ArgumentError("Unsupported field type: $(typeof(inputfield))"))
    end
    return  QCriterion(Q=storage;S2 = strain, Ω2 = vorticity, name=name, start=start, stop=stop, update_interval=update_interval)
end

function runtime_postprocessing!(QC::QCriterion{T,T1,T2,V},iter::Integer,n_iterations::Integer,config,S) where {T<:ScalarField,V,T1,T2}
    magnitude2!(QC.S2, S, config)
    Ω = Vorticity(S.U,S.gradU)
    magnitude2!(QC.Ω2,Ω,config)
    #here is where the Q-Crtiterion needs to be calculated and stored
    QC.Q.values .=  0.5 .* (QC.Ω2.values .- QC.S2.values)      
    return nothing
end
function convert_time_to_iterations(QC::QCriterion, model,dt,iterations)
    if model.time === Transient()
        if QC.start === nothing
            start = 1
        else 
            QC.start >= 0  || throw(ArgumentError("Start must be a value ≥ 0 (got $(QC.start))"))
            start = clamp(ceil(Int, QC.start / dt), 1, iterations) 
        end

        if QC.stop === nothing 
            stop = iterations
        else
            QC.stop ≥ 0 || throw(ArgumentError("stop must be ≥ 0 (got $(QC.stop))"))
            stop = clamp(floor(Int,QC.stop / dt), 1, iterations)
        end

        if QC.update_interval === nothing 
            update_interval = 1
        else
            QC.update_interval > 0 || throw(ArgumentError("update interval must be > 0 (got $(QC.update_interval))"))
            update_interval = max(1, floor(Int,QC.update_interval / dt))
        end
        stop >= start || throw(ArgumentError("After conversion with dt=$dt the averaging window is empty (start = $start, stop = $stop)"))
        return QCriterion(Q=QC.Q,S2 = QC.S2, Ω2 = QC.Ω2, name=QC.name, start=start, stop=stop, update_interval=update_interval)

    else #for Steady runs use iterations 
        if QC.start === nothing
            start = 1
        else 
            QC.start isa Integer || throw(ArgumentError("For steady runs, start must be specified in iterations and therefore be an integer (got $(QC.start))"))
            QC.start >=1     || throw(ArgumentError("Start must be ≥1 (got $(QC.start))"))
            start = QC.start
        end

        if QC.stop === nothing 
            stop = iterations
        else
            QC.stop isa Integer || throw(ArgumentError("For steady runs, stop must be specified in iterations and therefore be an integer (got $(QC.stop))"))
            QC.stop >=1     || throw(ArgumentError("Stop must be ≥1 (got $(QC.stop))"))
            stop = QC.stop
        end

        if QC.update_interval === nothing 
            update_interval = 1
        else
            QC.update_interval isa Integer || throw(ArgumentError("For steady runs, update_interval must be specified in iterations and therefore be an integer (got $(QC.update_interval))"))
            QC.update_interval >= 1 || throw(ArgumentError("update interval must be ≥1 (got $(QC.update_interval))"))
            update_interval = QC.update_interval
        end

        stop >= start || throw(ArgumentError("stop iteration needs to be ≥ start  (got start = $start, stop = $stop)"))
        stop <= iterations || throw(ArgumentError("stop ($stop) must be ≤ iterations ($iterations)"))
        return QCriterion(Q=QC.Q;S2 = QC.S2, Ω2 = QC.Ω2,name=QC.name, start=start, stop=stop, update_interval=update_interval)
    end
end