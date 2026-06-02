# FOAM mesh validation/benchmark oracle. Bit-exact via raw-bit hashing (OOM-safe).
# Usage:
#   julia --project /tmp/foam_validate.jl hashes  <meshdir> [scale]
#   julia --project /tmp/foam_validate.jl bench    <meshdir> [scale]
#   julia --project /tmp/foam_validate.jl stages   <meshdir> [scale]
using XCALibre
using Printf, StaticArrays
const FM = XCALibre.FoamMesh

# raw-bit folding so -0.0/0.0 and NaN payloads are distinguished (true bit-exactness)
@inline feed(h::UInt, x::AbstractFloat) = hash(reinterpret(UInt64, Float64(x)), h)
@inline feed(h::UInt, x::Integer)       = hash(Int64(x), h)
@inline feed(h::UInt, x::Symbol)        = hash(x, h)
@inline feed(h::UInt, r::UnitRange)     = hash((Int64(first(r)), Int64(last(r))), h)
function feed(h::UInt, x::Union{StaticArrays.StaticArray,Tuple,AbstractVector})
    for el in x; h = feed(h, el); end
    h
end
function fieldhash(arr)
    h = hash(length(arr))
    for el in arr; h = feed(h, el); end
    h
end

function digest(mesh)
    fields = Pair{String,Any}[]
    push!(fields, "node.coords"      => [n.coords for n in mesh.nodes])
    push!(fields, "node.cells_range" => [n.cells_range for n in mesh.nodes])
    push!(fields, "cell.centre"      => [c.centre for c in mesh.cells])
    push!(fields, "cell.volume"      => [c.volume for c in mesh.cells])
    push!(fields, "cell.nodes_range" => [c.nodes_range for c in mesh.cells])
    push!(fields, "cell.faces_range" => [c.faces_range for c in mesh.cells])
    push!(fields, "face.nodes_range" => [f.nodes_range for f in mesh.faces])
    push!(fields, "face.ownerCells"  => [f.ownerCells for f in mesh.faces])
    push!(fields, "face.centre"      => [f.centre for f in mesh.faces])
    push!(fields, "face.normal"      => [f.normal for f in mesh.faces])
    push!(fields, "face.e"           => [f.e for f in mesh.faces])
    push!(fields, "face.area"        => [f.area for f in mesh.faces])
    push!(fields, "face.delta"       => [f.delta for f in mesh.faces])
    push!(fields, "face.weight"      => [f.weight for f in mesh.faces])
    push!(fields, "cell_nodes"       => mesh.cell_nodes)
    push!(fields, "cell_faces"       => mesh.cell_faces)
    push!(fields, "cell_neighbours"  => mesh.cell_neighbours)
    push!(fields, "cell_nsign"       => mesh.cell_nsign)
    push!(fields, "face_nodes"       => mesh.face_nodes)
    push!(fields, "node_cells"       => mesh.node_cells)
    push!(fields, "boundary_cellsID" => mesh.boundary_cellsID)
    push!(fields, "bnd.name"         => [b.name for b in mesh.boundaries])
    push!(fields, "bnd.IDs_range"    => [b.IDs_range for b in mesh.boundaries])
    return fields
end

function cmd_hashes(meshdir, scale)
    mesh = FOAM3D_mesh(meshdir; scale=scale)
    for (name, arr) in digest(mesh)
        println("FIELD ", name, " ", length(arr), " ", fieldhash(arr))
    end
end

function cmd_hashall(dirs)
    for d in dirs
        println("MESH ", d)
        mesh = FOAM3D_mesh(d; scale=1.0)
        for (name, arr) in digest(mesh)
            println("FIELD ", name, " ", length(arr), " ", fieldhash(arr))
        end
        mesh = nothing; GC.gc()
    end
end

function cmd_bench(meshdir, scale)
    FOAM3D_mesh(meshdir; scale=scale)            # warmup (compile)
    GC.gc()
    t = @timed FOAM3D_mesh(meshdir; scale=scale)
    @printf("TIME %.3f s | ALLOC %.3f GB | GC %.1f%%\n",
        t.time, t.bytes/2^30, 100*t.gctime/t.time)
end

function cmd_stages(meshdir, scale)
    TI, TF = Int64, Float64
    FOAM3D_mesh(meshdir; scale=scale)            # warmup all
    GC.gc()
    r = @timed FM.read_FOAM3D(meshdir, scale, TI, TF);    fd = r.value
    @printf("STAGE read     %.3f s | %.3f GB | GC %.1f%%\n", r.time, r.bytes/2^30, 100*r.gctime/max(r.time,1e-9))
    c = @timed FM.connect_mesh(fd, TI, TF);               cn = c.value
    @printf("STAGE connect  %.3f s | %.3f GB | GC %.1f%%\n", c.time, c.bytes/2^30, 100*c.gctime/max(c.time,1e-9))
    g = @timed FM.generate_mesh(fd, cn, TI, TF);          m = g.value
    @printf("STAGE generate %.3f s | %.3f GB | GC %.1f%%\n", g.time, g.bytes/2^30, 100*g.gctime/max(g.time,1e-9))
    gm = @timed FM.compute_geometry!(m)
    @printf("STAGE geometry %.3f s | %.3f GB | GC %.1f%%\n", gm.time, gm.bytes/2^30, 100*gm.gctime/max(gm.time,1e-9))
end

const CMD = ARGS[1]
const DIR = length(ARGS) >= 2 ? ARGS[2] : ""
const SCALE = (CMD != "hashall" && length(ARGS) >= 3) ? parse(Float64, ARGS[3]) : 1.0
CMD == "hashall" ? cmd_hashall(ARGS[2:end]) :
CMD == "hashes" ? cmd_hashes(DIR, SCALE) :
CMD == "bench"  ? cmd_bench(DIR, SCALE)  :
CMD == "stages" ? cmd_stages(DIR, SCALE) :
    error("unknown command $CMD")
