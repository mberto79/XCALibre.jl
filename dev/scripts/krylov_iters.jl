# Krylov.jl iterations per solve on a threaded motorBike smoke run; usage in dev/scripts/INDEX.md.
using XCALibre
const K = XCALibre.Multithread.Krylov
const XV = XCALibre.Multithread.XVector{Float64}
const LOG = Tuple{Symbol,Int}[]
function K.cg!(ws::K.CgWorkspace{Float64,Float64,XV}, A::SparseXCSR, b::XV; kw...)
    invoke(K.cg!, Tuple{K.CgWorkspace{Float64,Float64,XV}, Any, AbstractVector{Float64}}, ws, A, b; kw...)
    push!(LOG, (:cg, ws.stats.niter)); ws
end
function K.bicgstab!(ws::K.BicgstabWorkspace{Float64,Float64,XV}, A::SparseXCSR, b::XV; kw...)
    invoke(K.bicgstab!, Tuple{K.BicgstabWorkspace{Float64,Float64,XV}, Any, AbstractVector{Float64}}, ws, A, b; kw...)
    push!(LOG, (:bicgstab, ws.stats.niter)); ws
end
include(joinpath(@__DIR__, "motorbike_smoke.jl"))
open(io -> foreach(e -> println(io, e), LOG), out * ".log_all", "w")
per = length(LOG) ÷ (iterations + 1)
open(out * ".iters", "w") do io
    println(io, "solves_per_iteration=$per total_solves=$(length(LOG))")
    for pos in 1:per
        its = [LOG[k][2] for k in pos:per:length(LOG)]
        println(io, "pos $pos $(LOG[pos][1]) mean=$(round(sum(its)/length(its), digits=1)) all=$(its)")
    end
end
