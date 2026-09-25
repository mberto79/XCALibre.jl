# potential_flow! on a DistributedMesh vs serial, per rank under mpiexec. The projection runs
# before any solver, so it is the first thing on a rank to touch U's ghosts.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

ncorrectors = 2

gmesh = rank == 0 ? bfs_mesh() : nothing
ref = if rank == 0
    ms, cs = incompressible_case(gmesh, bfs_bcs; iterations=1)
    initialise!(ms.momentum.U, [0.5, 0.0, 0.0])
    r = potential_flow!(ms, cs; ncorrectors)
    (collect(ms.momentum.U.x.values), collect(ms.momentum.U.y.values),
     collect(r.potential.values))
else
    nothing
end
Us_x, Us_y, Phis = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
model, config = incompressible_case(dm, bfs_bcs; iterations=1)
initialise!(model.momentum.U, [0.5, 0.0, 0.0])
result = potential_flow!(model, config; ncorrectors)

n = dm.partition.n_owned
orig = dm.orig_cells
owned_err(loc, ser) = maximum(abs.(Array(loc)[1:n] .- ser[orig[1:n]]); init=0.0)

dux = owned_err(model.momentum.U.x.values, Us_x)
duy = owned_err(model.momentum.U.y.values, Us_y)
# Phi is defined up to a constant only when no patch fixes it; BFS fixes p at the outlet
dphi = owned_err(result.potential.values, Phis)

@testset "potential_flow! distributed (rank $rank)" begin
    @test dux < 1e-6
    @test duy < 1e-6
    @test dphi < 1e-6
    # non-vacuous: the projection changed U away from the uniform field it started from
    @test maximum(abs.(Array(model.momentum.U.x.values)[1:n] .- 0.5)) > 1e-3
end
rank == 0 && println("potential_flow bfs n=$(MPI.Comm_size(comm)) dux=$dux duy=$duy dphi=$dphi")
