export gather, PFOAMWriter

# NEW SECTION: gather (rank-0 reconstruction in original cell order)

function _gather_owned(vals, dmesh, comm, root)
    part = dmesh.partition
    n = part.n_owned
    v = Array(view(vals, 1:n)) # host stage: device buffers must never reach MPI collectives
    ids = Array(view(dmesh.orig_cells, 1:n))
    counts = MPI.Gather(Int32(n), comm; root)
    if MPI.Comm_rank(comm) == root
        N = Int(sum(counts))
        allv = MPI.Gatherv!(v, MPI.VBuffer(similar(v, N), counts), comm; root)
        allids = MPI.Gatherv!(ids, MPI.VBuffer(similar(ids, N), counts), comm; root)
        out = similar(allv)
        out[allids] .= allv
        out
    else
        MPI.Gatherv!(v, nothing, comm; root)
        MPI.Gatherv!(ids, nothing, comm; root)
        nothing
    end
end

"""
    gather(field, dmesh; comm=dmesh.comm, root=0)

Gather a distributed field to `root` in ORIGINAL (pre-partition) cell order.
Returns a `Vector` (ScalarField) or `(x, y, z)` NamedTuple of Vectors (VectorField)
on `root`, `nothing` on other ranks.
"""
gather(f::ScalarField, dmesh::DistributedMesh; comm=getfield(dmesh, :comm), root=0) =
    _gather_owned(f.values, dmesh, comm, root)
gather(f::VectorField, dmesh::DistributedMesh; comm=getfield(dmesh, :comm), root=0) = (;
    x=_gather_owned(f.x.values, dmesh, comm, root),
    y=_gather_owned(f.y.values, dmesh, comm, root),
    z=_gather_owned(f.z.values, dmesh, comm, root))

# NEW SECTION: decomposed OpenFOAM writer (processor<rank>/ per rank, OpenFOAM binary)

# faces are written internal (owned-owned), physical patches, processor patches; `flip` marks a
# processor face whose stored owner is a ghost, written reversed so its normal leaves the owned cell
struct PFOAMWriter
    dir::String                 # processor<rank>
    order::Vector{Int}          # local face ids in written order
    flip::BitVector             # per written face
    n_internal::Int
    flux::Base.RefValue{Any}    # face flux attached by the solver, written as phi
end

_foam_header(class, location, object; binary=true) = """
FoamFile
{
    version     2.0;
    format      $(binary ? "binary" : "ascii");$(binary ? "\n    arch        \"LSB;label=32;scalar=64\";" : "")
    class       $class;
    location    "$location";
    object      $object;
}
"""

# OpenFOAM's binary list: count, then the raw little-endian values between parentheses
function _bin_list(io, v::AbstractVector)
    println(io)
    println(io, length(v))
    write(io, '(')
    write(io, v)
    write(io, ')')
end

_label(x) = x <= typemax(Int32) ? Int32(x) : error("OpenFOAM binary output uses 32-bit labels; $x does not fit")

function _pface_order(dmesh, nfaces)
    n_owned = dmesh.partition.n_owned
    nb = length(dmesh.mesh.boundary_cellsID)
    isproc = falses(nfaces)
    for pp ∈ dmesh.procs, f ∈ pp.faces
        isproc[f] = true
    end
    internal = [f for f ∈ nb+1:nfaces if !isproc[f]]
    procf = reduce(vcat, (Int.(pp.faces) for pp ∈ dmesh.procs); init=Int[])
    internal, nb, procf
end

function initialise_writer(format::OpenFOAM, dmesh::DistributedMesh)
    mesh = dmesh.mesh
    mesh isa Mesh3 || error("The OpenFOAM format can only be used for 3D simulations. Use `output=VTK()` instead.")
    rank = dmesh.partition.rank
    rank == 0 && touch("XCALibre.foam")
    dir = "processor$rank"
    polyMeshDir = mkpath(joinpath(dir, "constant", "polyMesh"))

    backend = _get_backend(mesh)
    nodes = get_data(mesh.nodes, backend)
    faces = get_data(mesh.faces, backend)
    face_nodes = get_data(mesh.face_nodes, backend)
    boundaries = get_data(mesh.boundaries, backend)
    n_owned = dmesh.partition.n_owned

    internal, nb, procf = _pface_order(dmesh, length(faces))
    order = vcat(internal, 1:nb, procf)
    flip = BitVector([k > length(internal) + nb && faces[f].ownerCells[1] > n_owned for (k, f) ∈ enumerate(order)])
    ni = length(internal)

    # points: only nodes referenced by faces (ghost-cell far-side nodes stay unwritten)
    used = falses(length(nodes))
    for face ∈ faces, j ∈ face.nodes_range
        used[face_nodes[j]] = true
    end
    n2c = cumsum(used)
    open(joinpath(polyMeshDir, "points"), "w") do io
        write(io, _foam_header("vectorField", "constant/polyMesh", "points"))
        _bin_list(io, [SVector{3,Float64}(n.coords) for (i, n) ∈ enumerate(nodes) if used[i]])
    end

    # faceCompactList: offsets then zero-based point labels
    offsets, labels = Int32[0], Int32[]
    for (k, f) ∈ enumerate(order)
        ids = [_label(n2c[face_nodes[j]] - 1) for j ∈ faces[f].nodes_range]
        flip[k] && reverse!(ids)
        append!(labels, ids)
        push!(offsets, _label(length(labels)))
    end
    open(joinpath(polyMeshDir, "faces"), "w") do io
        write(io, _foam_header("faceCompactList", "constant/polyMesh", "faces"))
        _bin_list(io, offsets)
        _bin_list(io, labels)
    end

    owner = Int32[_label(faces[f].ownerCells[flip[k] ? 2 : 1] - 1) for (k, f) ∈ enumerate(order)]
    note = "nPoints: $(Int(sum(used))) nCells: $n_owned nFaces: $(length(order)) nInternalFaces: $ni"
    open(joinpath(polyMeshDir, "owner"), "w") do io
        write(io, replace(_foam_header("labelList", "constant/polyMesh", "owner"),
            "    class" => "    note        \"$note\";\n    class"))
        _bin_list(io, owner)
    end
    open(joinpath(polyMeshDir, "neighbour"), "w") do io
        write(io, _foam_header("labelList", "constant/polyMesh", "neighbour"))
        _bin_list(io, Int32[_label(faces[f].ownerCells[2] - 1) for f ∈ internal])
    end

    open(joinpath(polyMeshDir, "boundary"), "w") do io
        println(io, _foam_header("polyBoundaryMesh", "constant/polyMesh", "boundary"; binary=false))
        println(io, length(boundaries) + length(dmesh.procs))
        println(io, "(")
        for b ∈ boundaries
            # first() (not [1]) so a patch with 0 local faces still yields its offset
            write(io, """
            $(b.name)
            {
                type            patch;
                nFaces          $(length(b.IDs_range));
                startFace       $(ni + first(b.IDs_range) - 1);
            }
            """)
        end
        start = ni + nb
        for pp ∈ dmesh.procs
            write(io, """
            procBoundary$(rank)to$(pp.neighbour)
            {
                type            processor;
                inGroups        1(processor);
                nFaces          $(length(pp.faces));
                startFace       $start;
                matchTolerance  0.0001;
                transform       unknown;
                myProcNo        $rank;
                neighbProcNo    $(pp.neighbour);
            }
            """)
            start += length(pp.faces)
        end
        println(io, ")")
    end

    # original cell ids, so reconstructPar and distribute(FOAMCase) recover the undecomposed order
    open(joinpath(polyMeshDir, "cellProcAddressing"), "w") do io
        write(io, _foam_header("labelList", "constant/polyMesh", "cellProcAddressing"))
        _bin_list(io, Int32[_label(c - 1) for c ∈ view(dmesh.orig_cells, 1:n_owned)])
    end

    PFOAMWriter(dir, order, flip, ni, Ref{Any}(nothing))
end

attach_flux!(w::PFOAMWriter, mdotf) = (w.flux[] = mdotf; nothing)

# NEW SECTION: writer dispatch (unified: solver bodies call initialise_writer/save_output)

# VTK has no decomposed writer: the sentinel lets write_interval=-1 runs proceed and makes any write
# an error rather than silence (the solver bodies only skip a `nothing` writer)
struct NoDistributedWriter end
initialise_writer(::VTK, ::DistributedMesh) = NoDistributedWriter()
write_results(iteration, time, ::DistributedMesh, ::NoDistributedWriter, args...; kwargs...) =
    error("VTK has no decomposed writer; use output=OpenFOAM() (3D meshes) or write_interval=-1")

# NEW SECTION: field output

_proc_patch_value(::ScalarField) = "uniform 0"
_proc_patch_value(::VectorField) = "uniform (0 0 0)"

_host_values(f::ScalarField, backend) = Float64.(copy_scalarfield_to_cpu(f.values, backend))
function _host_values(f::VectorField, backend)
    x, y, z = copy_to_cpu(f.x.values, f.y.values, f.z.values, backend)
    [SVector{3,Float64}(x[i], y[i], z[i]) for i ∈ eachindex(x)]
end

function write_results(iteration::TI, time, dmesh::DistributedMesh, w::PFOAMWriter,
        BCs, args...; suffix=nothing) where TI
    timedir = iteration == time ? (@sprintf "%i" iteration) : (@sprintf "%.8f" time)
    timedirpath = mkpath(joinpath(w.dir, timedir))
    mesh = dmesh.mesh
    rank = dmesh.partition.rank
    n_owned = dmesh.partition.n_owned
    backend = _get_backend(mesh)
    boundaries_cpu = get_data(mesh.boundaries, backend)

    for (label, field) ∈ args
        isscalar = field isa ScalarField
        isscalar || field isa VectorField || throw("""
        Input data should be a ScalarField or VectorField e.g. ("U", U)
        """)
        open(joinpath(timedirpath, label), "w") do io
            write(io, _foam_header(isscalar ? "volScalarField" : "volVectorField", "$timedir", label))
            write(io, IOFormats._FOAM_DIMENSIONS)
            write(io, "internalField   nonuniform List<$(isscalar ? "scalar" : "vector")>")
            _bin_list(io, _host_values(field, backend)[1:n_owned])
            println(io, ";")
            println(io, "boundaryField")
            println(io, "{")
            for BC ∈ getproperty(BCs, Symbol(label))
                println(io, "\t", boundaries_cpu[BC.ID].name)
                println(io, IOFormats._foam_boundary_entry(BC))
            end
            # ponytail: proc-patch values written as uniform zero; ParaView's decomposed
            # reader ignores them — write real ghost values if OF utilities ever need them
            for pp ∈ dmesh.procs
                println(io, "\tprocBoundary$(rank)to$(pp.neighbour)")
                println(io, """
                \t{
                \t\ttype processor;
                \t\tvalue $(_proc_patch_value(field));
                \t}
                """)
            end
            println(io, "}")
        end
    end
    w.flux[] === nothing || _write_phi(joinpath(timedirpath, "phi"), timedir, dmesh, w, backend)
end

# face flux in written face order; a flipped processor face carries the opposite sign
function _write_phi(path, timedir, dmesh, w, backend)
    v = Float64.(copy_scalarfield_to_cpu(w.flux[].values, backend))
    vals = [w.flip[k] ? -v[f] : v[f] for (k, f) ∈ enumerate(w.order)]
    boundaries = get_data(dmesh.mesh.boundaries, backend)
    nb = length(dmesh.mesh.boundary_cellsID)
    rank = dmesh.partition.rank
    open(path, "w") do io
        write(io, _foam_header("surfaceScalarField", timedir, "phi"))
        write(io, IOFormats._FOAM_DIMENSIONS)
        write(io, "internalField   nonuniform List<scalar>")
        _bin_list(io, vals[1:w.n_internal])
        println(io, ";")
        println(io, "boundaryField")
        println(io, "{")
        patch(name, type, r) = (println(io, "\t$name\n\t{\n\t\ttype $type;");
            write(io, "\t\tvalue nonuniform List<scalar>"); _bin_list(io, vals[r]); println(io, ";\n\t}"))
        for b ∈ boundaries
            patch(b.name, "calculated", w.n_internal .+ b.IDs_range)
        end
        start = w.n_internal + nb
        for pp ∈ dmesh.procs
            patch("procBoundary$(rank)to$(pp.neighbour)", "processor", start+1:start+length(pp.faces))
            start += length(pp.faces)
        end
        println(io, "}")
    end
end

# every `nonuniform List<...>` of an OpenFOAM binary file, in file order
function _read_foam_lists(path, ::Type{T}) where T
    b = read(path)
    tag = Vector{UInt8}("nonuniform List<")
    out = Vector{T}[]
    i = 1
    while (r = findnext(tag, b, i)) !== nothing
        j = findnext(==(UInt8('>')), b, last(r)) + 1
        while isspace(Char(b[j])); j += 1; end
        k = j
        while isdigit(Char(b[k])); k += 1; end
        n = parse(Int, String(b[j:k-1]))
        while b[k] != UInt8('('); k += 1; end
        v = Vector{T}(undef, n)
        copyto!(reinterpret(UInt8, v), 1, b, k + 1, n * sizeof(T))
        push!(out, v)
        i = k + 1 + n * sizeof(T)
    end
    out
end
