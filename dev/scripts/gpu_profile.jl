# GPU kernel and host-API profile of the motorBike smoke case; usage in dev/scripts/INDEX.md.
const PROF_OUT = ARGS[1]
empty!(ARGS); append!(ARGS, ["gpu", "1", PROF_OUT * "_smoke"])
include(joinpath(@__DIR__, "motorbike_smoke.jl"))
run!(model, config(2))
p = CUDA.@profile run!(model, config(3))
open(PROF_OUT * ".prof", "w") do io
    show(IOContext(io, :limit => false, :displaysize => (1000, 400)), p)
end
