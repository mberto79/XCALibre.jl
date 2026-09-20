# Per-rank face count of every patch, so the rank counts that leave a patch empty are known
# before a case is run. Metis partitions cells, so a patch small against the domain empties out.
using XCALibre, MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
grid = get(ENV, "PROBE_GRID", "backwardFacingStep_10mm.unv")
path = isabspath(grid) ? grid : joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), grid)
gmesh = rank == 0 ? UNV2D_mesh(path, scale=parse(Float64, get(ENV, "PROBE_SCALE", "0.001"))) : nothing
dm = distribute(gmesh; comm=comm)
counts = [(String(b.name), length(b.IDs_range)) for b ∈ dm.mesh.boundaries]
all = MPI.gather(counts, comm; root=0)
if rank == 0
    println("RANKS=", MPI.Comm_size(comm), " grid=", basename(path))
    for (r, c) ∈ enumerate(all)
        println("  rank ", r-1, ": ", join(["$(n)=$(v)" for (n, v) ∈ c], " "))
    end
    empties = [(r-1, n) for (r, c) ∈ enumerate(all) for (n, v) ∈ c if v == 0]
    println("EMPTY PATCHES: ", isempty(empties) ? "none" :
        join(["rank$(r):$(n)" for (r, n) ∈ empties], " "))
end
