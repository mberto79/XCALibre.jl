# Times loading every part file in a directory in one process: second pass reported, so compilation is excluded.
using XCALibre
const D = XCALibre.Distribute

load(path, n) = D._read_part_file(path, n)

dir = ARGS[1]
files = sort(filter(f -> startswith(f, "rank_") && endswith(f, ".xdm"), readdir(dir)))
n = length(files)
paths = joinpath.(dir, files)
foreach(p -> load(p, n), paths)
GC.gc(true)
t = [@elapsed(load(p, n)) for p ∈ paths]
mb = sum(filesize, paths) / 2^20
println("PARTLOAD dir=$dir n=$n ext=$(splitext(files[1])[2]) total_s=$(round(sum(t); digits=3)) ",
    "max_s=$(round(maximum(t); digits=3)) MB=$(round(mb; digits=1))")
