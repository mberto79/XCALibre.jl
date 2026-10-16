# bounding_box of a distributed mesh is the global box; the LES Empty distance of a mesh away
# from the origin matches serial on every rank.
using XCALibre, PETSc, MPI, StaticArrays, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
function moved_pitz()
    mesh = FOAM3D_mesh(joinpath(grids_dir, "OF_pitzDaily", "polyMesh"), scale=1)
    mesh.node_coords .+= Ref(SVector(1.0, 2.0, 3.0))
    mesh
end
pitz_bcs(mesh) = assign(region=mesh, (U = [Dirichlet(:inlet, [1.0, 0.0, 0.0]),
    Zerogradient(:outlet), Wall(:upperWall, [0.0, 0.0, 0.0]), Wall(:lowerWall, [0.0, 0.0, 0.0]),
    Empty(:frontAndBack)],)).U

gmesh = rank == 0 ? moved_pitz() : nothing
ref = rank == 0 ? (bounding_box(gmesh), XCALibre.ModelPhysics.delta_scaling(gmesh, pitz_bcs(gmesh))) : nothing
(gmin, gmax), (gdist, gpow) = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
pmin, pmax = bounding_box(dm)
dist, pow = XCALibre.ModelPhysics.delta_scaling(dm, pitz_bcs(dm))

@testset "distributed bounding_box (rank $rank)" begin
    @test pmin == gmin
    @test pmax == gmax
    @test dist ≈ gdist rtol=1e-12
    @test pow == gpow
end
