export HaloExchange, halo_exchange!, halo_exchange_adjoint!

"""
    HaloExchange

Reusable halo communication schedule built from `DistributedMesh.procs`. Holds one
send/recv buffer pair per neighbour (width 1 for scalars, 3 for vectors) plus
preallocated MPI requests so repeat exchanges do not allocate.
"""
struct HaloExchange{TF,VI,VB}
    comm::MPI.Comm
    neighbours::Vector{Int}
    send_idx::Vector{VI}          # owned cell ids to pack, per neighbour (backend arrays)
    recv_idx::Vector{VI}          # ghost cell ids to fill, per neighbour
    send_bufs::Vector{VB}
    recv_bufs::Vector{VB}
    host_send::Vector{Vector{TF}} # staging mirrors; empty when cuda_aware
    host_recv::Vector{Vector{TF}}
    send_reqs::Vector{MPI.Request}
    recv_reqs::Vector{MPI.Request}
    width::Int
    cuda_aware::Bool              # true = MPI reads device buffers directly (CPU or CUDA-aware MPI)
end

_on_backend(backend, v) = begin
    d = KernelAbstractions.allocate(backend, eltype(v), length(v))
    copyto!(d, v)
    d
end

# cuda_aware=false forces host staging on a GPU backend (path comparison / non-aware MPI)
function HaloExchange(dmesh::DistributedMesh, width::Integer, backend; comm=MPI.COMM_WORLD,
        cuda_aware::Bool = backend isa KernelAbstractions.CPU || MPI.has_cuda())
    TF = _get_float(dmesh)
    TI = _get_int(dmesh)
    procs = getfield(dmesh, :procs)
    # concrete element types so empty procs (n=1) still infer the struct parameters
    IdxT = typeof(KernelAbstractions.allocate(backend, TI, 0))
    BufT = typeof(KernelAbstractions.allocate(backend, TF, 0))
    send_idx = IdxT[_on_backend(backend, pp.send_cells) for pp ∈ procs]
    recv_idx = IdxT[_on_backend(backend, pp.recv_ghosts) for pp ∈ procs]
    send_bufs = BufT[KernelAbstractions.allocate(backend, TF, width*length(pp.send_cells)) for pp ∈ procs]
    recv_bufs = BufT[KernelAbstractions.allocate(backend, TF, width*length(pp.recv_ghosts)) for pp ∈ procs]
    host_send = Vector{TF}[Vector{TF}(undef, cuda_aware ? 0 : width*length(pp.send_cells)) for pp ∈ procs]
    host_recv = Vector{TF}[Vector{TF}(undef, cuda_aware ? 0 : width*length(pp.recv_ghosts)) for pp ∈ procs]
    HaloExchange(comm, Int[pp.neighbour for pp ∈ procs], send_idx, recv_idx,
        send_bufs, recv_bufs, host_send, host_recv,
        [MPI.Request() for _ ∈ procs], [MPI.Request() for _ ∈ procs],
        Int(width), cuda_aware)
end

# NEW SECTION: pack/unpack kernels

@kernel function _pack!(buf, phi::AbstractScalarField, idx)
    i = @index(Global)
    @inbounds buf[i] = phi[idx[i]]
end

@kernel function _pack!(buf, U::AbstractVectorField, idx)
    i = @index(Global)
    @inbounds begin
        u = U[idx[i]]
        buf[3i-2] = u[1]; buf[3i-1] = u[2]; buf[3i] = u[3]
    end
end

@kernel function _unpack!(phi::AbstractScalarField, buf, idx)
    i = @index(Global)
    @inbounds phi[idx[i]] = buf[i]
end

@kernel function _unpack!(U::AbstractVectorField, buf, idx)
    i = @index(Global)
    @inbounds U[idx[i]] = SVector{3}(buf[3i-2], buf[3i-1], buf[3i])
end

# adjoint scatter: same owned cell may receive from several neighbours concurrently
@kernel function _unpack_add!(phi::AbstractScalarField, buf, idx)
    i = @index(Global)
    @inbounds begin
        c = idx[i]
        Atomix.@atomic phi.values[c] += buf[i]
    end
end

@kernel function _unpack_add!(U::AbstractVectorField, buf, idx)
    i = @index(Global)
    @inbounds begin
        c = idx[i]
        Atomix.@atomic U.x.values[c] += buf[3i-2]
        Atomix.@atomic U.y.values[c] += buf[3i-1]
        Atomix.@atomic U.z.values[c] += buf[3i]
    end
end

@kernel function _zero!(phi::AbstractScalarField, idx)
    i = @index(Global)
    @inbounds phi[idx[i]] = zero(eltype(phi))
end

@kernel function _zero!(U::AbstractVectorField, idx)
    i = @index(Global)
    @inbounds U[idx[i]] = zero(SVector{3,eltype(U)})
end

# NEW SECTION: exchange

_mpi_send_buf(H, k) = H.cuda_aware ? H.send_bufs[k] : H.host_send[k]
_mpi_recv_buf(H, k) = H.cuda_aware ? H.recv_bufs[k] : H.host_recv[k]

"""
    halo_exchange!(phi, H::HaloExchange, backend, workgroup)

Fill ghost entries of `phi` (scalar or vector field) with the owning neighbours' values.
Irecv-first, pack, sync, Isend, wait, unpack; buffers and requests are reused.
"""
function halo_exchange!(phi, H::HaloExchange, backend, workgroup)
    for k ∈ eachindex(H.neighbours)
        MPI.Irecv!(_mpi_recv_buf(H, k), H.comm, H.recv_reqs[k]; source=H.neighbours[k], tag=0)
    end
    for k ∈ eachindex(H.neighbours)
        idx = H.send_idx[k]
        kernel! = _pack!(_setup(backend, workgroup, length(idx))...)
        kernel!(H.send_bufs[k], phi, idx)
    end
    KernelAbstractions.synchronize(backend)
    for k ∈ eachindex(H.neighbours)
        H.cuda_aware || copyto!(H.host_send[k], H.send_bufs[k])
        MPI.Isend(_mpi_send_buf(H, k), H.comm, H.send_reqs[k]; dest=H.neighbours[k], tag=0)
    end
    MPI.Waitall(H.recv_reqs)
    for k ∈ eachindex(H.neighbours)
        H.cuda_aware || copyto!(H.recv_bufs[k], H.host_recv[k])
        idx = H.recv_idx[k]
        kernel! = _unpack!(_setup(backend, workgroup, length(idx))...)
        kernel!(phi, H.recv_bufs[k], idx)
    end
    KernelAbstractions.synchronize(backend)
    MPI.Waitall(H.send_reqs)
    phi
end

"""
    halo_exchange_adjoint!(phi, H::HaloExchange, backend, workgroup)

Reverse scatter (transpose of `halo_exchange!`): ghost cotangents are sent owner-ward and
accumulated into the owned cells they copy from; ghost entries are zeroed. Used by AD (Phase 7).
"""
function halo_exchange_adjoint!(phi, H::HaloExchange, backend, workgroup)
    # message direction reverses, so buffer roles swap (recv_bufs sized for ghosts)
    for k ∈ eachindex(H.neighbours)
        MPI.Irecv!(_mpi_send_buf(H, k), H.comm, H.recv_reqs[k]; source=H.neighbours[k], tag=0)
    end
    for k ∈ eachindex(H.neighbours)
        idx = H.recv_idx[k]
        kernel! = _pack!(_setup(backend, workgroup, length(idx))...)
        kernel!(H.recv_bufs[k], phi, idx)
        zero! = _zero!(_setup(backend, workgroup, length(idx))...)
        zero!(phi, idx)
    end
    KernelAbstractions.synchronize(backend)
    for k ∈ eachindex(H.neighbours)
        H.cuda_aware || copyto!(H.host_recv[k], H.recv_bufs[k])
        MPI.Isend(_mpi_recv_buf(H, k), H.comm, H.send_reqs[k]; dest=H.neighbours[k], tag=0)
    end
    MPI.Waitall(H.recv_reqs)
    for k ∈ eachindex(H.neighbours)
        H.cuda_aware || copyto!(H.send_bufs[k], H.host_send[k])
        idx = H.send_idx[k]
        kernel! = _unpack_add!(_setup(backend, workgroup, length(idx))...)
        kernel!(phi, H.send_bufs[k], idx)
    end
    KernelAbstractions.synchronize(backend)
    MPI.Waitall(H.send_reqs)
    phi
end
