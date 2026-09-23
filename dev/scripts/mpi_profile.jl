# Rank-0 CPU profile of the motorBike MPI smoke; usage in dev/scripts/INDEX.md.
const PROF_ITERS = parse(Int, get(ENV, "PROF_ITERS", "20"))
using Profile
include(joinpath(@__DIR__, "motorbike_smoke.jl"))
init!(); pre!(model, config(PROF_ITERS))
Profile.clear(); Profile.init(n=10^7, delay=0.001)
t = @elapsed Profile.@profile run!(model, config(PROF_ITERS); progress=false)
if is_root()
    open(out * ".mpiprof", "w") do io
        println(io, "# rank 0, $(MPI.Comm_size(MPI.COMM_WORLD)) ranks, $PROF_ITERS iterations, $(round(t, digits=2)) s under the profiler")
        Profile.print(IOContext(io, :displaysize => (100000, 400)); format=:flat, sortedby=:count, mincount=20)
    end
end
