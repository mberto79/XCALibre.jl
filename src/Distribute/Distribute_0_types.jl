export Partition, ProcessorPatch, DistributedMesh
export AbstractDistributedSolver

# implemented by solver backends (PETSc/HYPRE extensions) from Phase 3
abstract type AbstractDistributedSolver end

"""
    Partition

Cell ownership metadata for one rank: owned cells occupy local ids `1:n_owned`, ghost
cells `n_owned+1:n_owned+n_ghost`. `local_to_global` maps local ids to the
block-contiguous global numbering where this rank owns rows `row_start:row_end`.
"""
struct Partition{VI<:AbstractVector{<:Integer}}
    rank::Int                 # MPI rank (0-based)
    nranks::Int
    n_owned::Int
    n_ghost::Int
    local_to_global::VI       # length n_owned+n_ghost, owned block first
    owner::VI                 # owning MPI rank per local cell
    row_start::Int
    row_end::Int
end

"""
    ProcessorPatch

Communication schedule with one neighbour rank. `send_cells` (owned, here) and
`recv_ghosts` (ghosts owned by the neighbour) are both sorted by original global cell id,
so the two sides align index-for-index without negotiation.
"""
struct ProcessorPatch{VI<:AbstractVector{<:Integer}}
    neighbour::Int            # neighbour MPI rank (0-based)
    faces::VI                 # local processor-face ids shared with neighbour
    send_cells::VI            # owned local cell ids to send
    recv_ghosts::VI           # ghost local cell ids to fill on receipt
end

"""
    DistributedMesh <: AbstractMesh

Wraps a rank-local mesh (owned + one ghost layer) with partition and communication
metadata. All non-metadata properties forward to the wrapped mesh, so fields, `Physics`
and kernels treat it as a normal mesh.
"""
struct DistributedMesh{M<:AbstractMesh,P<:Partition,PP<:ProcessorPatch,VI} <: AbstractMesh
    mesh::M                   # local Mesh3/Mesh2
    partition::P
    procs::Vector{PP}
    orig_cells::VI            # original global cell id per local cell (I/O, gather)
    orig_faces::VI            # original global face id per local face
end

const _DM_FIELDS = (:mesh, :partition, :procs, :orig_cells, :orig_faces)

Base.getproperty(dm::DistributedMesh, s::Symbol) =
    s in _DM_FIELDS ? getfield(dm, s) : getproperty(getfield(dm, :mesh), s)
Base.propertynames(dm::DistributedMesh) =
    (_DM_FIELDS..., propertynames(getfield(dm, :mesh))...)

Base.show(io::IO, dm::DistributedMesh) = begin
    p = getfield(dm, :partition)
    m = getfield(dm, :mesh)
    print(io, "DistributedMesh (rank $(p.rank+1)/$(p.nranks)): ",
        "$(p.n_owned) owned + $(p.n_ghost) ghost cells, ",
        "$(length(m.faces)) faces, $(length(getfield(dm, :procs))) processor patches")
end
