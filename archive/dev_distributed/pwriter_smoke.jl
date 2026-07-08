# CPU smoke-test for the decomposed OpenFOAM writer (no GPU/PETSc). Run:
#   source dev/local_stack.sh && mpiexec -n 2 julia --project test/../dev/pwriter_smoke.jl
# Writes processor<rank>/ dirs into a temp cwd; checks files exist + owned-cell counts.
using XCALibre, MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
mesh_path = joinpath(pkgdir(XCALibre), "examples/0_GRIDS/BFS_UNV_3D_hex_5mm.unv")
gmesh = rank == 0 ? UNV3D_mesh(mesh_path, scale=0.001) : nothing
dm = distribute(gmesh; comm)
U = VectorField(dm); p = ScalarField(dm)  # default zeros; writer smoke ignores values
tmp = mktempdir(); cd(tmp)
w = initialise_writer(OpenFOAM(), dm)
write_results(1, 1, dm, w, (U=(), p=()), ("U", U), ("p", p))
n = dm.partition.n_owned
ok = isfile("processor$rank/constant/polyMesh/points") &&
     isfile("processor$rank/constant/polyMesh/boundary") &&
     isfile("processor$rank/1/U") && isfile("processor$rank/1/p")
println("rank $rank n_owned=$n procdir_ok=$ok dir=$tmp/processor$rank")
MPI.Barrier(comm)
rank == 0 && println(ok ? "PWRITER SMOKE OK" : "PWRITER SMOKE BAD")
