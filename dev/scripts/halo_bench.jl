# A/B per halo exchange: persistent requests (library) vs fresh Irecv!/Isend, same buffers, alternated.
# part: julia --project=<env> dev/scripts/halo_bench.jl part <mesh.unv> <n> <dir>
# run:  mpiexec -n <n> julia --project=<env> dev/scripts/halo_bench.jl run <dir> <reps> [width]
using XCALibre, MPI, KernelAbstractions, Statistics
import XCALibre.Distribute: _mpi_send_buf, _mpi_recv_buf, _pack!, _unpack!, _tag

function fresh_exchange!(phi, H, backend, workgroup)
    rr = [MPI.Irecv!(_mpi_recv_buf(H, k), H.comm; source=H.neighbours[k], tag=_tag(H.width)) for k ∈ eachindex(H.neighbours)]
    for k ∈ eachindex(H.neighbours)
        _pack!(_setup(backend, workgroup, length(H.send_idx[k]))...)(H.send_bufs[k], phi, H.send_idx[k])
    end
    KernelAbstractions.synchronize(backend)
    sr = [MPI.Isend(_mpi_send_buf(H, k), H.comm; dest=H.neighbours[k], tag=_tag(H.width)) for k ∈ eachindex(H.neighbours)]
    MPI.Waitall(rr)
    for k ∈ eachindex(H.neighbours)
        _unpack!(_setup(backend, workgroup, length(H.recv_idx[k]))...)(phi, H.recv_bufs[k], H.recv_idx[k])
    end
    KernelAbstractions.synchronize(backend)
    MPI.Waitall(sr)
    phi
end

function timed(f, comm, reps)
    MPI.Barrier(comm); t = MPI.Wtime()
    for _ ∈ 1:reps; f(); end
    MPI.Allreduce(MPI.Wtime() - t, max, comm) / reps
end

if ARGS[1] == "part"
    mesh = UNV3D_mesh(ARGS[2], scale=0.001)
    partition_mesh(mesh, parse(Int, ARGS[3]); dir=ARGS[4])
else
    MPI.Init(); comm = MPI.COMM_WORLD
    dm = distribute(ARGS[2]; comm)
    reps = parse(Int, ARGS[3]); width = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 3
    backend, workgroup = CPU(), 1024
    phi = width == 1 ? ScalarField(dm) : VectorField(dm)
    H = HaloExchange(dm, width, backend)
    lib() = halo_exchange!(phi, H, backend, workgroup)
    old() = fresh_exchange!(phi, H, backend, workgroup)
    lib(); old()
    a0 = @allocated lib(); a1 = @allocated old()
    tl, to = Float64[], Float64[]
    for _ ∈ 1:10
        push!(tl, timed(lib, comm, reps)); push!(to, timed(old, comm, reps))
    end
    MPI.Comm_rank(comm) == 0 && println("HALOBENCH n=$(MPI.Comm_size(comm)) width=$width reps=$reps ",
        "persistent_us=$(round(1e6median(tl), digits=2)) fresh_us=$(round(1e6median(to), digits=2)) ",
        "alloc_persistent=$a0 alloc_fresh=$a1")
end
