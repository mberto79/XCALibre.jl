export Probe, TXT, CSV

abstract type ProbeOutput end
struct TXT <: ProbeOutput end
struct CSV <: ProbeOutput end
@kwdef struct Probe{T<:AbstractField,I<:Integer,S<:AbstractString,O<:ProbeOutput}
    field::T
    index::I
    name::S
    output::O = CSV()
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end
"""
    Probe(
    #required arguments
    field,
    mesh_cpu; 

    #required keyword arguments
    location::AbstractVector
    name::AbstractString

    #optional keyword arguments
    output::ProbeOutput 
    start::Union{Real,Nothing},
    stop::Union{Real,Nothing},
    update_interval::Union{Real,Nothing})
Constructor to probe an internal field during runtime and write it out to a .csv file. Once created, should be passed to the `Configuration` object as an argument with keyword `postprocess`

# Input arguments 
- `field`, the field you want to probe, such as `model.momentum.U`
- `mesh_cpu`, the mesh object stored on the CPU needs to be passed, this is to find the nearest cell centre to the location you want to probe. On a distributed mesh pass the distributed mesh (on the CPU) on every rank: the rank owning the nearest cell samples it and rank 0 writes the file

# Input Keyword arguments
- `location`, e.g `location = [1.0, 2.0, 0.0]` and must be a vector ∈ ℝ³ even for 2D simulations
- `name`, the name of the output file e.g `"Velocity probe at [1.0, 2.0, 3.0]"`

# Optional arguments
- `output::ProbeOutput` the format to output the probe, defaults to CSV, can only pass either `TXT()` or `CSV()`
- `start::Union{Real,Nothing}` optional keyword which specifies the start of the probe output window, for **steady** simulations, this is in **iterations**, for **transient** simulations it is in **flow time**.   
- `stop::Union{Real,Nothing}` optional keyword which specifies the end iteration/time of the probe output window. Default value is the last iteration/timestep. 
- `update_interval::Union{Real,Nothing}` optional keyword which specifies how often the probe's field is output to file (default value is 1 i.e probe writes field value every iteration/timestep). 
"""
function Probe(field,mesh_cpu;location::AbstractVector, name::AbstractString,output::ProbeOutput = CSV(),start::Union{Real,Nothing}=nothing,stop::Union{Real,Nothing}=nothing,update_interval::Union{Real,Nothing}=nothing)
    length(location) == 3 || throw(ArgumentError("location must be a 3 element vector "))
    index, best_centre = find_nearest_cell_index(mesh_cpu, location, field.mesh)
    is_report_rank(field.mesh) && @info "Nearest cell centre located at $best_centre "

    return Probe(field=field, index=index,name=name,output=output,start=start,stop=stop,update_interval=update_interval)
end

# On a distributed mesh only the rank owning the probed cell has index > 0; the others add zero to
# the sum over ranks, and rank 0 writes the file
function runtime_postprocessing!(prb::Probe{T,I,S,O},iter::Integer,n_iterations::Integer,Str,time,config) where {T<:VectorField,I,S,O}
    if must_calculate(prb,iter,n_iterations)
        (; index, field) = prb
        TF = eltype(field.x.values)
        f = index > 0 ? TF[@allowscalar(field.x.values[index]), @allowscalar(field.y.values[index]),
            @allowscalar(field.z.values[index])] : zeros(TF, 3)
        f = global_sum(f, field.mesh)
        is_report_rank(field.mesh) && write_probe_to_file(time,f[1],f[2],f[3],prb.name,prb.output)
    end
    return nothing
end
function runtime_postprocessing!(prb::Probe{T,I,S,O}, iter::Integer, n_iterations::Integer,Str, time, config) where {T<:ScalarField,I,S,O}
    if must_calculate(prb, iter, n_iterations)
        (; index, field) = prb
        f = index > 0 ? @allowscalar(field.values[index]) : zero(eltype(field.values))
        f = global_sum(f, field.mesh)
        is_report_rank(field.mesh) && write_probe_to_file(time, f, prb.name, prb.output)
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
# Nearest owned cell centre, ties to the lowest mesh-wide cell id (the serial choice). `mesh` holds
# the geometry on the host, `pmesh` (the field's mesh) the ranks; index is 0 on the other ranks.
function find_nearest_cell_index(mesh, vector_coords, pmesh=mesh)
    best_index = 0
    best_distance = Inf
    best_id = typemax(Int)
    @inbounds for i ∈ 1:n_owned_cells(pmesh)
        ctr = mesh.cells[i].centre
        distance = Float64((ctr[1] - vector_coords[1])^2 + (ctr[2] - vector_coords[2])^2 + (ctr[3] - vector_coords[3])^2)
        id = Int(global_cell_index(pmesh, i))
        if distance < best_distance || (distance == best_distance && id < best_id)
            best_distance = distance
            best_index = i
            best_id = id
        end
    end
    candidate = best_distance == global_min(best_distance, pmesh) ? best_id : typemax(Int)
    winner = global_min(candidate, pmesh) # collective: every rank must reach it
    owner = candidate < typemax(Int) && candidate == winner
    index = owner ? best_index : 0
    centre = owner ? Vector{Float64}(mesh.cells[index].centre) : zeros(3)
    best_centre = global_sum(centre, pmesh)

    return index, best_centre
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