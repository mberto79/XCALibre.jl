export Probe

abstract type ProbeOutput end
struct TXT <: ProbeOutput end
struct CSV <: ProbeOutput end
@kwdef struct Probe{T<:AbstractField,I<:Integer,S<:AbstractString,O<:ProbeOutput}
    field::T
    index::I
    name::S
    output::O = TXT()
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end

function Probe(field,mesh_cpu;location::AbstractVector, name::AbstractString,output::ProbeOutput = TXT(),start::Union{Real,Nothing}=nothing,stop::Union{Real,Nothing}=nothing,update_interval::Union{Real,Nothing}=nothing)
    length(location) == 3 || throw(ArgumentError("location must be a 3 element vector "))
    index, best_centre = find_nearest_cell_index(mesh_cpu,location)
    @info "Nearest cell centre located at $best_centre "

    return Probe(field=field, index=index,name=name,output=output,start=start,stop=stop,update_interval=update_interval)
end

function runtime_postprocessing!(prb::Probe{T,I,S,O},iter::Integer,n_iterations::Integer,Str,time,config) where {T<:VectorField,I,S,O}
    if must_calculate(prb,iter,n_iterations)
        index = prb.index

        fx = @allowscalar prb.field.x.values[index]
        fy = @allowscalar prb.field.y.values[index]
        fz = @allowscalar prb.field.z.values[index]
        write_probe_to_file(time,fx,fy,fz,prb.name,prb.output)
    end
    return nothing
end
function runtime_postprocessing!(prb::Probe{T,I,S,O}, iter::Integer, n_iterations::Integer,Str, time, config) where {T<:ScalarField,I,S,O}
    if must_calculate(prb, iter, n_iterations)
        index = prb.index

        f = @allowscalar prb.field.values[index]

        write_probe_to_file(time, f, prb.name, prb.output)
    end
    return nothing
end


#just write a function that writes out the time and the value of the scalar/vector field at that instant 
function write_probe_to_file(time, f, name, ::TXT)
    open(String(name) * ".txt", "a") do io
        println(io, time, ' ', f)
    end
end

function write_probe_to_file(time, f, name, ::CSV)
    filename = String(name) * ".csv"
    write_header = !isfile(filename)

    open(filename, "a") do io
        if write_header
            println(io, "time,f")
        end
        println(io, time, ',', f)
    end
end


function write_probe_to_file(time, fx, fy, fz, name, ::TXT)
    open(String(name) * ".txt", "a") do io
        println(io, time, ' ', fx, ' ', fy, ' ', fz)
    end
end

function write_probe_to_file(time, fx, fy, fz, name, ::CSV)
    filename = String(name) * ".csv"
    write_header = !isfile(filename)

    open(filename, "a") do io
        if write_header
            println(io, "time,fx,fy,fz")
        end
        println(io, time, ',', fx, ',', fy, ',', fz)
    end
end
function find_nearest_cell_index(mesh, vector_coords)
    best_index = 1
    best_distance = Inf

    @inbounds for i ∈ eachindex(mesh.cells)
        ctr = mesh.cells[i].centre
        distance =  (ctr[1] - vector_coords[1])^2 + (ctr[2] - vector_coords[2])^2 + (ctr[3] - vector_coords[3])^2
        if distance < best_distance
            best_distance = distance
            best_index = i
        end
    end
    best_centre = mesh.cells[best_index].centre

    return best_index, best_centre
end

function convert_time_to_iterations(prb::Probe, model,dt,iterations)
    if model.time === Transient()
        if prb.start === nothing
            start = 1
        else 
            prb.start >= 0  || throw(ArgumentError("Start must be a value ≥ 0 (got $(prb.start))"))
            start = clamp(ceil(Int, prb.start / dt), 1, iterations) 
        end

        if prb.stop === nothing 
            stop = iterations
        else
            prb.stop ≥ 0 || throw(ArgumentError("stop must be ≥ 0 (got $(prb.stop))"))
            stop = clamp(floor(Int,prb.stop / dt), 1, iterations)
        end

        if prb.update_interval === nothing 
            update_interval = 1
        else
            prb.update_interval > 0 || throw(ArgumentError("update interval must be > 0 (got $(prb.update_interval))"))
            update_interval = max(1, floor(Int,prb.update_interval / dt))
        end
        stop >= start || throw(ArgumentError("After conversion with dt=$dt the averaging window is empty (start = $start, stop = $stop)"))
        return Probe(field=prb.field, index=prb.index,name=prb.name,output=prb.output,start=start,stop=stop,update_interval=update_interval)

    else #for Steady runs use iterations 
        if prb.start === nothing
            start = 1
        else 
            prb.start isa Integer || throw(ArgumentError("For steady runs, start must be specified in iterations and therefore be an integer (got $(prb.start))"))
            prb.start >=1     || throw(ArgumentError("Start must be ≥1 (got $(prb.start))"))
            start = prb.start
        end

        if prb.stop === nothing 
            stop = iterations
        else
            prb.stop isa Integer || throw(ArgumentError("For steady runs, stop must be specified in iterations and therefore be an integer (got $(prb.stop))"))
            prb.stop >=1     || throw(ArgumentError("Stop must be ≥1 (got $(prb.stop))"))
            stop = prb.stop
        end

        if prb.update_interval === nothing 
            update_interval = 1
        else
            prb.update_interval isa Integer || throw(ArgumentError("For steady runs, update_interval must be specified in iterations and therefore be an integer (got $(prb.update_interval))"))
            prb.update_interval >= 1 || throw(ArgumentError("update interval must be ≥1 (got $(prb.update_interval))"))
            update_interval = prb.update_interval
        end

        stop >= start || throw(ArgumentError("stop iteration needs to be ≥ start  (got start = $start, stop = $stop)"))
        stop <= iterations || throw(ArgumentError("stop ($stop) must be ≤ iterations ($iterations)"))
        return Probe(field=prb.field, index=prb.index,name=prb.name,output=prb.output,start=start,stop=stop,update_interval=update_interval)

    end
end