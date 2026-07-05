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
    gather(field, dmesh; comm=MPI.COMM_WORLD, root=0)

Gather a distributed field to `root` in ORIGINAL (pre-partition) cell order.
Returns a `Vector` (ScalarField) or `(x, y, z)` NamedTuple of Vectors (VectorField)
on `root`, `nothing` on other ranks.
"""
gather(f::ScalarField, dmesh::DistributedMesh; comm=MPI.COMM_WORLD, root=0) =
    _gather_owned(f.values, dmesh, comm, root)
gather(f::VectorField, dmesh::DistributedMesh; comm=MPI.COMM_WORLD, root=0) = (;
    x=_gather_owned(f.x.values, dmesh, comm, root),
    y=_gather_owned(f.y.values, dmesh, comm, root),
    z=_gather_owned(f.z.values, dmesh, comm, root))

# NEW SECTION: decomposed OpenFOAM writer (processor<rank>/ per rank)

struct PFOAMWriter
    dir::String        # processor<rank>
    n_internal::Int    # internal (owned-owned) face count, for reference
end

_foam_header(class, location, object) = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       $class;
    location    "$location";
    object      $object;
}
"""

# faces written in OF order: internal (owned-owned), physical patches, processor patches
function _pface_layout(dmesh, faces_cpu)
    n_owned = dmesh.partition.n_owned
    nb = length(dmesh.mesh.boundary_cellsID)
    proc_set = Set{Int}()
    for pp ∈ dmesh.procs, f ∈ pp.faces
        push!(proc_set, Int(f))
    end
    internal = [f for f ∈ (nb+1):length(faces_cpu) if !(f in proc_set)]
    internal, nb, n_owned
end

# proc-face normals must point OUT of the owned cell; flip node order when owner is ghost
_pface_owner_flip(face, n_owned) = face.ownerCells[1] > n_owned ?
    (Int(face.ownerCells[2]), true) : (Int(face.ownerCells[1]), false)

function _write_face_nodes(io, face, face_nodes, n2c, flip)
    nr = face.nodes_range
    ids = [n2c[face_nodes[j]] for j ∈ nr]
    flip && reverse!(ids)
    write(io, "$(length(ids))(")
    for id ∈ ids
        write(io, "$(id - 1) ") # OF zero-indexed
    end
    write(io, ")\n")
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

    internal, nb, n_owned = _pface_layout(dmesh, faces)
    ni = length(internal)
    nfaces_out = ni + nb + sum(length(pp.faces) for pp ∈ dmesh.procs; init=0)

    # points: only nodes referenced by faces (ghost-cell far-side nodes stay unwritten)
    used = falses(length(nodes))
    for face ∈ faces, j ∈ face.nodes_range
        used[face_nodes[j]] = true
    end
    n2c = cumsum(used) # local node id -> compact written id
    open(joinpath(polyMeshDir, "points"), "w") do io
        println(io, _foam_header("vectorField", "constant/polyMesh", "points"))
        println(io, Int(sum(used)))
        println(io, "(")
        for (i, node) ∈ enumerate(nodes)
            used[i] || continue
            c = node.coords
            println(io, @sprintf "(%g %g %g)" c[1] c[2] c[3])
        end
        println(io, ")")
    end

    open(joinpath(polyMeshDir, "faces"), "w") do io
        println(io, _foam_header("faceList", "constant/polyMesh", "faces"))
        println(io, nfaces_out)
        println(io, "(")
        for f ∈ internal
            _write_face_nodes(io, faces[f], face_nodes, n2c, false)
        end
        for f ∈ 1:nb
            _write_face_nodes(io, faces[f], face_nodes, n2c, false)
        end
        for pp ∈ dmesh.procs, f ∈ pp.faces
            _, flip = _pface_owner_flip(faces[f], n_owned)
            _write_face_nodes(io, faces[f], face_nodes, n2c, flip)
        end
        println(io, ")")
    end

    note = "nPoints: $(Int(sum(used))) nCells: $n_owned nFaces: $nfaces_out nInternalFaces: $ni"
    open(joinpath(polyMeshDir, "owner"), "w") do io
        write(io, """
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       labelList;
            note        "$note";
            location    "constant/polyMesh";
            object      owner;
        }
        """)
        println(io, nfaces_out)
        println(io, "(")
        for f ∈ internal
            println(io, Int(faces[f].ownerCells[1]) - 1)
        end
        for f ∈ 1:nb
            println(io, Int(faces[f].ownerCells[1]) - 1)
        end
        for pp ∈ dmesh.procs, f ∈ pp.faces
            owner, _ = _pface_owner_flip(faces[f], n_owned)
            println(io, owner - 1)
        end
        println(io, ")")
    end

    open(joinpath(polyMeshDir, "neighbour"), "w") do io
        println(io, _foam_header("labelList", "constant/polyMesh", "neighbour"))
        println(io, ni)
        println(io, "(")
        for f ∈ internal
            println(io, Int(faces[f].ownerCells[2]) - 1)
        end
        println(io, ")")
    end

    open(joinpath(polyMeshDir, "boundary"), "w") do io
        println(io, _foam_header("polyBoundaryMesh", "constant/polyMesh", "boundary"))
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

    PFOAMWriter(dir, ni)
end

# NEW SECTION: writer dispatch (unified: solver bodies call initialise_writer/save_output)

# VTK has no decomposed writer; distributed runs use OpenFOAM() or write_interval=-1.
# nothing writer => the solver body's `outputWriter === nothing || save_output(...)` skips.
initialise_writer(::VTK, ::DistributedMesh) = nothing

# NEW SECTION: field output

_proc_patch_value(::ScalarField) = "uniform 0"
_proc_patch_value(::VectorField) = "uniform (0 0 0)"

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
        filename = joinpath(timedirpath, label)
        isscalar = field isa ScalarField
        isscalar || field isa VectorField || throw("""
        Input data should be a ScalarField or VectorField e.g. ("U", U)
        """)
        open(filename, "w") do io
            write(io, _foam_header(isscalar ? "volScalarField" : "volVectorField",
                "$timedir", label), "\n")
            write(io, "internalField   nonuniform List<$(isscalar ? "scalar" : "vector")>\n")
            println(io, n_owned)
            println(io, "(")
            if isscalar
                vals = copy_scalarfield_to_cpu(field.values, backend)
                for i ∈ 1:n_owned
                    println(io, vals[i])
                end
            else
                x, y, z = copy_to_cpu(field.x.values, field.y.values, field.z.values, backend)
                for i ∈ 1:n_owned
                    println(io, "(", x[i], " ", y[i], " ", z[i], ")")
                end
            end
            println(io, ");")
            println(io, "boundaryField")
            println(io, "{")
            fieldBCs = getproperty(BCs, Symbol(label))
            for BC ∈ fieldBCs
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
end
