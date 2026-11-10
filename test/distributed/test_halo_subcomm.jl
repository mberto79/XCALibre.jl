# Halo exchange and reductions on a sub-communicator (ranks 0-1 together, the rest alone): nothing
# may read COMM_WORLD by name.
using XCALibre, MPI, Test
using StaticArrays

MPI.Init()
world = MPI.COMM_WORLD
wrank = MPI.Comm_rank(world)
sub = MPI.Comm_split(world, wrank < 2 ? 0 : 1, wrank)
srank = MPI.Comm_rank(sub)

gmesh = srank == 0 ?
    UNV3D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "3d_box_1000x1000x1000mm_10.unv"), scale=0.001) :
    nothing
dm = distribute(gmesh; comm=sub)
backend, workgroup = CPU(), 64
n_owned = dm.partition.n_owned
ghosts = n_owned+1:n_owned+dm.partition.n_ghost
f(x, y, z) = 1.3x + 2.7y - 0.9z + 0.31
ref_mean = MPI.bcast(srank == 0 ? sum(f(c.centre...) for c ∈ gmesh.cells) / length(gmesh.cells) : nothing, sub; root=0)

@testset "sub-communicator halo (world rank $wrank, size $(MPI.Comm_size(sub)))" begin
    config = (; hardware=(; backend, workgroup))
    phi = DistributedScalarField(dm, backend; comm=sub)
    initialise!(phi, (x, y, z) -> f(x, y, z))
    phi.field.values[ghosts] .= NaN
    sync!(phi, config)
    @test all(phi.field[i] === f(dm.mesh.cells[i].centre...) for i ∈ ghosts)
    @test pmean(phi) ≈ ref_mean rtol = 1e-14
    @test dm.comm == sub
    @test XCALibre.Solvers.global_max(Float64(wrank), dm) == (wrank < 2 ? 1.0 : MPI.Comm_size(world) - 1.0)
end
MPI.free(sub)
