# PTX of every kernel compiled by a 1-iteration motorBike GPU run, to <out>.ptx; usage in dev/scripts/INDEX.md.
const OUT = ARGS[1]
empty!(ARGS); append!(ARGS, ["gpu", "1", OUT * "_smoke"])
using CUDA
open(OUT * ".ptx", "w") do io
    CUDA.@device_code_ptx io=io include(expanduser("~/Julia/XCALibre.jl/dev/scripts/motorbike_smoke.jl"))
end
