export FOAMCase

# NEW SECTION: decomposed OpenFOAM reader (processor<rank>/constant/polyMesh per rank)

"""
    FOAMCase(dir; scale=1, integer_type=Int32, float_type=Float64)

An OpenFOAM case already decomposed into `processor<rank>/` directories (by `decomposePar` or by a
distributed XCALibre run with `output=OpenFOAM()`). `distribute(FOAMCase(dir))` gives each rank its
own part without any rank reading the global mesh; the keywords are those of [`FOAM3D_mesh`](@ref).
"""
struct FOAMCase{TI<:Integer,TF<:AbstractFloat}
    dir::String
    scale::Float64
end
FOAMCase(dir; scale=1, integer_type=Int32, float_type=Float64) =
    FOAMCase{integer_type,float_type}(String(dir), Float64(scale))

const _PROC_PATCH = r"^procBoundary(\d+)to(\d+)$"

_nprocessor_dirs(dir) = count(d -> occursin(r"^processor\d+$", d) && isdir(joinpath(dir, d)), readdir(dir))

"""
    distribute(case::FOAMCase; comm=MPI.COMM_WORLD)

Load a decomposed OpenFOAM case: each rank reads `processor<rank>/constant/polyMesh`, and ghost cells
are built by one exchange with each neighbour across the processor patches. Run under
`mpiexec -n <number of processor directories>`. Original cell ids come from `cellProcAddressing` when
the case has it, so `gather` returns fields in the undecomposed order; otherwise in rank-block order.
"""
function distribute(case::FOAMCase{TI,TF}; comm=MPI.COMM_WORLD) where {TI,TF}
    MPI.Initialized() || MPI.Init()
    quiet_nonroot!(comm)
    rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    nd = _nprocessor_dirs(case.dir)
    nd == nranks || error("$(case.dir) is decomposed into $nd processor directories but this run has " *
        "$nranks ranks; run under mpiexec -n $nd or decompose with numberOfSubdomains $nranks")
    poly = joinpath(case.dir, "processor$rank", "constant", "polyMesh")
    mesh = redirect_stdout(devnull) do
        FOAM3D_mesh(poly; scale=case.scale, integer_type=TI, float_type=Float64)
    end
    addr = joinpath(poly, "cellProcAddressing")
    orig = isfile(addr) ? redirect_stdout(() -> read_neighbour(addr, GlobalInt, Float64), devnull) : nothing
    dm = _attach_ghosts(mesh, orig, rank, comm)
    TF === Float64 && return dm
    DistributedMesh(convert_mesh_float(getfield(dm, :mesh), TF), getfield(dm, :partition),
        getfield(dm, :procs), getfield(dm, :orig_cells), getfield(dm, :orig_faces), HaloCache(), comm)
end

# processor patches become interior faces (owned owner, ghost neighbour) in place, so face ids and the
# normal out of the owned cell are kept; ghosts carry only interface faces and no nodes
function _attach_ghosts(mesh::Mesh3, orig, rank, comm)
    TI, TF = _get_int(mesh), _get_float(mesh)
    (; cells, faces, boundaries, cell_faces, cell_neighbours, cell_nsign, boundary_cellsID) = mesh
    n_owned = length(cells)
    iproc = findall(b -> occursin(_PROC_PATCH, string(b.name)), boundaries)
    nphys = length(boundaries) - length(iproc)
    iproc == nphys+1:length(boundaries) || error("processor patches must follow every physical patch")
    nb = nphys == 0 ? 0 : Int(last(boundaries[nphys].IDs_range))
    patches = map(boundaries[iproc]) do b
        m = match(_PROC_PATCH, string(b.name))
        parse(Int, m[1]) == rank || error("patch $(b.name) found in processor$rank")
        (parse(Int, m[2]), Int.(b.IDs_range))
    end

    counts = MPI.Allgather(n_owned, comm)
    row_start = sum(counts[1:rank]; init=0) + 1
    l2g_owned = collect(GlobalInt, row_start:row_start+n_owned-1)
    orig_owned = orig === nothing ? l2g_owned : orig

    # one record per interface face: owner centre, volume, global id, original id (ids exact in Float64)
    recv = [Matrix{Float64}(undef, 6, length(fs)) for (_, fs) ∈ patches]
    reqs = MPI.Request[]
    for (i, (q, fs)) ∈ enumerate(patches)
        send = Matrix{Float64}(undef, 6, length(fs))
        for (k, f) ∈ enumerate(fs)
            c = faces[f].ownerCells[1]
            send[:, k] = [cells[c].centre..., cells[c].volume, l2g_owned[c], orig_owned[c]]
        end
        push!(reqs, MPI.Irecv!(recv[i], comm; source=q, tag=17))
        push!(reqs, MPI.Isend(send, comm; dest=q, tag=17))
    end
    MPI.Waitall(reqs)

    # ghosts ordered by (owning rank, global id): patches are in ascending neighbour order
    issorted(first.(patches)) || error("processor patches of processor$rank are not in neighbour order")
    ghost_of = Dict{Int,Int}()
    gcells = eltype(cells)[]
    l2g_ghost, owner_ghost, orig_ghost = GlobalInt[], TI[], GlobalInt[]
    recv_ghosts = Vector{TI}[]
    for (i, (q, _)) ∈ enumerate(patches)
        r = recv[i]
        col = Dict{Int,Int}()
        foreach(k -> get!(col, Int(r[5, k]), k), axes(r, 2))
        first_g = n_owned + length(gcells) + 1
        for g ∈ sort!(collect(keys(col)))
            k = col[g]
            push!(gcells, Cell(SVector{3,TF}(r[1, k], r[2, k], r[3, k]), TF(r[4, k]),
                UnitRange{TI}(1, 0), UnitRange{TI}(1, 0)))
            push!(l2g_ghost, g); push!(owner_ghost, q); push!(orig_ghost, GlobalInt(r[6, k]))
            ghost_of[g] = n_owned + length(gcells)
        end
        push!(recv_ghosts, collect(TI, first_g:n_owned+length(gcells)))
    end
    all_cells = vcat(cells, gcells)

    # interface faces: neighbour becomes the ghost and e/delta/weight are those of an interior face
    new_faces = copy(faces)
    iface_ghost = Dict{Int,Int}()
    for (i, (_, fs)) ∈ enumerate(patches), (k, f) ∈ enumerate(fs)
        face = faces[f]
        g = ghost_of[Int(recv[i][5, k])]
        iface_ghost[f] = g
        oc, gc = all_cells[face.ownerCells[1]].centre, all_cells[g].centre
        weight, delta, e = weight_delta_e(face.centre - oc, face.centre - gc, gc - oc, face.normal)
        new_faces[f] = Face3D(face.nodes_range, SVector{2,TI}(face.ownerCells[1], g), face.centre,
            face.normal, e, face.area, delta, weight)
    end

    # cell-face CSR: owned cells keep their interior faces then gain their interface faces
    extra = [Tuple{TI,TI,TI}[] for _ ∈ 1:length(all_cells)]
    for (_, fs) ∈ patches, f ∈ fs
        o, g = faces[f].ownerCells[1], iface_ghost[f]
        push!(extra[o], (f, g, 1))
        push!(extra[g], (f, o, -1))
    end
    new_cf, new_cn, new_cs = TI[], TI[], TI[]
    for (c, cell) ∈ enumerate(all_cells)
        s = length(new_cf) + 1
        if c <= n_owned
            append!(new_cf, view(cell_faces, cell.faces_range))
            append!(new_cn, view(cell_neighbours, cell.faces_range))
            append!(new_cs, view(cell_nsign, cell.faces_range))
        end
        for (f, nb_c, sgn) ∈ extra[c]
            push!(new_cf, f); push!(new_cn, nb_c); push!(new_cs, sgn)
        end
        all_cells[c] = Cell(cell.centre, cell.volume, cell.nodes_range, UnitRange{TI}(s, length(new_cf)))
    end

    lmesh = Mesh3(all_cells, mesh.cell_nodes, new_cf, new_cn, new_cs, new_faces, mesh.face_nodes,
        boundaries[1:nphys], mesh.nodes, mesh.node_cells, mesh.get_float, mesh.get_int, boundary_cellsID[1:nb])
    n_ghost = length(gcells)
    partition = Partition(rank, MPI.Comm_size(comm), n_owned, n_ghost, vcat(l2g_owned, l2g_ghost),
        vcat(fill(TI(rank), n_owned), owner_ghost), row_start, row_start + n_owned - 1)
    procs = [ProcessorPatch(q, collect(TI, fs), sort!(unique(TI[faces[f].ownerCells[1] for f ∈ fs])),
        recv_ghosts[i]) for (i, (q, fs)) ∈ enumerate(patches)]
    DistributedMesh(lmesh, partition, procs, vcat(GlobalInt.(orig_owned), orig_ghost),
        zeros(GlobalInt, length(new_faces)), HaloCache(), comm)
end
