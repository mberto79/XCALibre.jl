# Plots dev/telemetry/scaling.csv and machine_bandwidth.csv. Uses Plots from the default
# environment, so run it as:  julia dev/scripts/plot_scaling.jl
# Writes PNGs to dev/telemetry/plots/.
ENV["GKSwstype"] = "100" # GR needs this to render without a display
using DelimitedFiles, Plots
using Plots.Measures
gr()
# axis labels are clipped at the default margins on multi-panel output
default(left_margin=6mm, bottom_margin=5mm, top_margin=3mm, right_margin=3mm,
        guidefontsize=10, titlefontsize=11, legendfontsize=7)

const TEL = joinpath(dirname(@__DIR__), "telemetry")
const OUT = joinpath(TEL, "plots")
mkpath(OUT)

# a tiny table: column name -> vector of strings, so no CSV.jl dependency is needed
function readtable(path)
    raw = readdlm(path, ',', String)
    Dict(strip(raw[1, j]) => raw[2:end, j] for j ∈ 1:size(raw, 2))
end

num(v) = parse.(Float64, v)

# rows matching every (column => value) pair, as an index vector
function pick(t, pairs...)
    idx = collect(1:length(first(values(t))))
    for (k, v) ∈ pairs
        idx = filter(i -> strip(t[k][i]) == v, idx)
    end
    sort(idx, by = i -> parse(Float64, t["ranks"][i]))
end

series(t, idx, col) = num(t[col][idx])

t = readtable(joinpath(TEL, "scaling.csv"))

# a named set of curves, each a (label, index vector)
function plot_curves(curves, col, ylabel, title; ideal=false, kwargs...)
    p = plot(; xlabel="MPI ranks", ylabel, title, legend=:topleft,
             xticks=([1,2,4,6,8], ["1","2","4","6","8"]), kwargs...)
    if ideal
        plot!(p, [1, 8], [1, 8], ls=:dash, c=:black, lw=1, label="ideal")
    end
    for (lab, idx) ∈ curves
        isempty(idx) && continue
        plot!(p, series(t, idx, "ranks"), series(t, idx, col),
              m=:circle, ms=4, lw=2, label=lab)
    end
    p
end

speedup(idx) = (v = series(t, idx, "s_per_iter"); v[1] ./ v)

P = "pinned_2200"
jac5  = pick(t, "code"=>"XCALibre", "solver"=>"Cg", "preconditioner"=>"Jacobi", "mesh"=>"bfs_tet_5mm", "clock_policy"=>P)
jac4  = pick(t, "code"=>"XCALibre", "solver"=>"Cg", "preconditioner"=>"Jacobi", "mesh"=>"bfs_unv_tet_4mm", "clock_policy"=>P)
amg5  = pick(t, "code"=>"XCALibre", "preconditioner"=>"BoomerAMG", "clock_policy"=>P)
gam5  = pick(t, "code"=>"XCALibre", "preconditioner"=>"GAMG", "clock_policy"=>P)
ofg   = pick(t, "code"=>"OpenFOAM", "solver"=>"GAMG", "clock_policy"=>P)
ofp   = pick(t, "code"=>"OpenFOAM", "solver"=>"PCG", "clock_policy"=>P)
free5 = pick(t, "code"=>"XCALibre", "solver"=>"Cg", "preconditioner"=>"Jacobi", "mesh"=>"bfs_tet_5mm", "clock_policy"=>"free")
cpw5  = pick(t, "code"=>"XCALibre", "solver"=>"Cg", "preconditioner"=>"Jacobi", "mesh"=>"bfs_tet_5mm", "clock_policy"=>"constant_power")

# 1. the headline: clock pinned vs clock free, same code, same case
p1 = plot(xlabel="MPI ranks", ylabel="parallel efficiency (%)", legend=:bottomleft,
          title="Throttling is what the original curve measured",
          xticks=([1,2,4,6,8], ["1","2","4","6","8"]), ylims=(0, 115))
hline!(p1, [100], ls=:dash, c=:black, lw=1, label="ideal")
for (lab, idx, c) ∈ (("clock pinned at 2200 MHz", jac5, :dodgerblue),
                     ("clock free (throttles 4400->3100)", free5, :crimson),
                     ("constant package power", cpw5, :seagreen))
    plot!(p1, series(t, idx, "ranks"), series(t, idx, "efficiency_pct"),
          m=:circle, ms=5, lw=2, c=c, label=lab)
end

# 2. efficiency, every configuration, clock pinned
p2 = plot_curves((("XCALibre Cg+Jacobi 500k", jac5), ("XCALibre Cg+Jacobi 1.32M", jac4),
                  ("XCALibre Cg+BoomerAMG 500k", amg5), ("XCALibre Cg+GAMG 500k", gam5),
                  ("OpenFOAM GAMG 500k", ofg), ("OpenFOAM PCG+diagonal 500k", ofp)),
                 "efficiency_pct", "parallel efficiency (%)",
                 "Efficiency at a pinned clock"; legend=:bottomleft, ylims=(0, 145))
hline!(p2, [100], ls=:dash, c=:black, lw=1, label="")

# 3. speedup
p3 = plot(xlabel="MPI ranks", ylabel="speedup", legend=:topleft,
          title="Speedup, clock pinned", xticks=([1,2,4,6,8], ["1","2","4","6","8"]))
plot!(p3, [1, 8], [1, 8], ls=:dash, c=:black, lw=1, label="ideal")
for (lab, idx) ∈ (("XCALibre Cg+Jacobi 500k", jac5), ("XCALibre Cg+Jacobi 1.32M", jac4),
                  ("XCALibre Cg+BoomerAMG 500k", amg5), ("XCALibre Cg+GAMG 500k", gam5),
                  ("OpenFOAM GAMG 500k", ofg), ("OpenFOAM PCG+diagonal 500k", ofp))
    plot!(p3, series(t, idx, "ranks"), speedup(idx), m=:circle, ms=4, lw=2, label=lab)
end

# 4. absolute cost: efficiency percentages hide that these differ by 6x
p4 = plot(xlabel="MPI ranks", ylabel="s / iteration (log)", yscale=:log10, legend=:topright,
          title="Absolute cost, same mesh and clock",
          xticks=([1,2,4,6,8], ["1","2","4","6","8"]))
for (lab, idx) ∈ (("XCALibre Cg+Jacobi", jac5), ("XCALibre Cg+BoomerAMG", amg5),
                  ("XCALibre Cg+GAMG", gam5), ("OpenFOAM GAMG", ofg), ("OpenFOAM PCG+diagonal", ofp))
    plot!(p4, series(t, idx, "ranks"), series(t, idx, "s_per_iter"), m=:circle, ms=4, lw=2, label=lab)
end

# 5. the machine's own ceiling, which is what makes the rest interpretable
b = readtable(joinpath(TEL, "machine_bandwidth.csv"))
bidx(ws) = sort(filter(i -> strip(b["working_set"][i]) == ws, 1:length(b["ranks"])),
                by = i -> parse(Float64, b["ranks"][i]))
p5 = plot(xlabel="MPI ranks", ylabel="aggregate GB/s (log)", yscale=:log10, legend=:topleft,
          title="Machine axpy ceiling: DRAM does not scale",
          xticks=([1,2,4,8], ["1","2","4","8"]))
for (lab, ws) ∈ (("cache resident", "cache_resident"), ("DRAM resident", "dram_resident"))
    i = bidx(ws)
    plot!(p5, num(b["ranks"][i]), num(b["GB_per_s"][i]), m=:circle, ms=4, lw=2, label=lab)
end
hline!(p5, [89.6], ls=:dash, c=:black, lw=1, label="DDR5-5600 theoretical 89.6")

for (name, p) ∈ (("throttling", p1), ("efficiency", p2), ("speedup", p3),
                 ("absolute_cost", p4), ("machine_ceiling", p5))
    savefig(p, joinpath(OUT, "$name.png"))
end
savefig(plot(p1, p2, p3, p4, layout=(2,2), size=(1500, 1050)), joinpath(OUT, "summary.png"))
println("wrote ", length(readdir(OUT)), " plots to ", OUT)
