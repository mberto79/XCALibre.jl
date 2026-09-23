# Max relative difference per residual series between two motorbike_smoke.jl .res files.
a, b = ARGS
ra = Dict(l[1] => parse.(Float64, l[2:end]) for l in split.(readlines(a)) if l[1] != "hash")
rb = Dict(l[1] => parse.(Float64, l[2:end]) for l in split.(readlines(b)) if l[1] != "hash")
for k in sort(collect(keys(ra)))
    x, y = ra[k], rb[k]; d = maximum(abs.(x .- y) ./ max.(abs.(x), 1e-300); init=0.0)
    println(rpad(k, 4), " maxrel=", d, " sigfigs≈", d == 0 ? Inf : round(-log10(d), digits=1))
end
