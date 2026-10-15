# bound! (clipping of negative k, omega, ... to a neighbour average) on a distributed mesh: owned
# cells match the serial result and ghost cells equal their owners, though a ghost sees only the
# neighbours present on its rank.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

# negative in bands across the whole domain, so many bounded cells sit on processor boundaries
wavy(x) = sin(900.0x[1]) * cos(700.0x[2])
bounded(mesh, config) = begin
    phi = ScalarField(mesh)
    phi.values .= [wavy(c.centre) for c ∈ mesh.cells]
    XCALibre.ModelPhysics.bound!(phi, config)
    phi
end

gmesh = rank == 0 ? bfs_mesh() : nothing
ref = if rank == 0
    _, cs = incompressible_case(gmesh, bfs_bcs; iterations=1)
    collect(bounded(gmesh, cs).values)
else
    nothing
end
ref = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
_, config = incompressible_case(dm, bfs_bcs; iterations=1)
phi = bounded(dm, config)
n = dm.partition.n_owned
nloc = n + dm.partition.n_ghost
orig = dm.orig_cells

@testset "distributed bound! (rank $rank)" begin
    @test count(<(0), [wavy(c.centre) for c ∈ dm.cells]) > 0
    @test all(phi.values[i] == ref[orig[i]] for i ∈ 1:n)
    @test all(phi.values[i] == ref[orig[i]] for i ∈ n+1:nloc)
    @test check_ghosts(phi, dm, config) == 0
end
