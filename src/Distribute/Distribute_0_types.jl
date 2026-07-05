export Partition, ProcessorPatch, DistributedMesh
export AbstractDistributedSolver
export bind_device!

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
# width-keyed halo-exchange cache; lazily filled on first sync! (per rank/backend). Mutable +
# built locally so it survives MPI.send of a DistributedMesh (requests/comm are rank-local) and
# composes with adapt(backend, dm) — the device copy starts empty and rebuilds on device.
mutable struct HaloCache
    w1::Any                   # HaloExchange (width 1) or nothing
    w3::Any                   # HaloExchange (width 3) or nothing
end
HaloCache() = HaloCache(nothing, nothing)

struct DistributedMesh{M<:AbstractMesh,P<:Partition,PP<:ProcessorPatch,VI} <: AbstractMesh
    mesh::M                   # local Mesh3/Mesh2
    partition::P
    procs::Vector{PP}
    orig_cells::VI            # original global cell id per local cell (I/O, gather)
    orig_faces::VI            # original global face id per local face
    halos::HaloCache          # lazily-built width-1/3 halo caches for self-syncing sync!
end

const _DM_FIELDS = (:mesh, :partition, :procs, :orig_cells, :orig_faces, :halos)

Base.getproperty(dm::DistributedMesh, s::Symbol) =
    s in _DM_FIELDS ? getfield(dm, s) : getproperty(getfield(dm, :mesh), s)
Base.propertynames(dm::DistributedMesh) =
    (_DM_FIELDS..., propertynames(getfield(dm, :mesh))...)

# NEW SECTION: GPU adaptation (Phase 6)

Adapt.@adapt_structure Partition
Adapt.@adapt_structure ProcessorPatch

# metadata stays on host: kernels never read it and HaloExchange/PETSc make their own device
# copies; only the wrapped mesh moves, so getproperty forwarding keeps working. A fresh empty
# HaloCache is used so the device copy rebuilds its halos on-device (on first sync!, via the
# GPU config backend); the host cache is not shared with the device mesh.
Adapt.adapt_structure(to, dm::DistributedMesh) = DistributedMesh(
    Adapt.adapt(to, getfield(dm, :mesh)), getfield(dm, :partition),
    getfield(dm, :procs), getfield(dm, :orig_cells), getfield(dm, :orig_faces), HaloCache())

"""
    bind_device!(backend, rank)

Bind this MPI rank to GPU `rank % ndevices` (one rank per device). No-op on CPU. Call
before `adapt(backend, dmesh)` or building fields/`HaloExchange` on a GPU backend.
"""
bind_device!(::KernelAbstractions.CPU, rank::Integer) = nothing
bind_device!(backend, rank::Integer) =
    error("bind_device!: no GPU extension loaded for $(typeof(backend)) — e.g. `using CUDA`")

# GPU exts declare their PETSc pairing: external-package name + device MPIAIJ mat type
# (CUDA → "cuda"/"mpiaijcusparse", AMD → "hip"/"mpiaijhipsparse")
petsc_device_info(nzval) =
    error("petsc_device_info: no PETSc device mapping for $(typeof(nzval)); use solve_on=CPU()")

Base.show(io::IO, dm::DistributedMesh) = begin
    p = getfield(dm, :partition)
    m = getfield(dm, :mesh)
    print(io, "DistributedMesh (rank $(p.rank+1)/$(p.nranks)): ",
        "$(p.n_owned) owned + $(p.n_ghost) ghost cells, ",
        "$(length(m.faces)) faces, $(length(getfield(dm, :procs))) processor patches")
end
