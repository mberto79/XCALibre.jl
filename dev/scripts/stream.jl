# Machine ceiling: MPI strong-scaling of axpy (y = a*x + y), the exact kernel PETSc's
# VecAXPY runs, at a cache-resident and a DRAM-resident total size. Reports aggregate GB/s
# and the clock the cores actually held, so throttling is separated from bandwidth.
using MPI, Printf
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm); n = MPI.Comm_size(comm)

max_mhz() = maximum(parse(Float64, split(l, ':')[2])
                    for l in eachline("/proc/cpuinfo") if startswith(l, "cpu MHz"))

function axpy_rate(Ntot, comm, rank, n; secs=3.0)
    nloc = Ntot ÷ n
    x = Vector{Float64}(undef, nloc); y = Vector{Float64}(undef, nloc)
    fill!(x, 1.0); fill!(y, 2.0)          # first touch before timing: undef is not resident
    a = 1.0000001
    @inbounds for i in eachindex(y); y[i] = a*x[i] + y[i]; end   # warmup
    MPI.Barrier(comm)
    t0 = time(); reps = 0
    while time() - t0 < secs
        @inbounds @simd for i in eachindex(y); y[i] = a*x[i] + y[i]; end
        reps += 1
    end
    el = time() - t0
    mhz = max_mhz()
    MPI.Barrier(comm)
    tmax = MPI.Allreduce(el, MPI.MAX, comm)
    rtot = MPI.Allreduce(reps, MPI.SUM, comm)
    mhzmax = MPI.Allreduce(mhz, MPI.MAX, comm)
    # 24 bytes moved and 2 flop per element
    gbs = rtot * nloc * 24 / tmax / 1e9
    gfs = rtot * nloc * 2  / tmax / 1e9
    (gbs, gfs, mhzmax, nloc, y[1])
end

for (label, Ntot) in (("L3res", 500_000), ("DRAM", 60_000_000))
    gbs, gfs, mhz, nloc, _ = axpy_rate(Ntot, comm, rank, n)
    rank == 0 && @printf("%-6s n=%d  Ntot=%d  perrank=%.1f MB/vec  %.1f GB/s  %.2f Gflop/s  maxMHz=%.0f\n",
                         label, n, Ntot, nloc*8/1e6, gbs, gfs, mhz)
end
MPI.Finalize()
