# potential_flow! on a 3D periodic DistributedMesh vs serial. Covers what the 2D test cannot:
# the 3D reconstruction branch, periodic patches, and `pref`, which pins a reference cell that
# must be the same global cell on every rank.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "cascade_case.jl"))

ncorrectors = 2

gmesh = rank == 0 ? cascade_mesh() : nothing
ref = if rank == 0
    ms, cs = cascade_case(gmesh; iterations=1)
    r = potential_flow!(ms, cs; ncorrectors, pref=0.0)
    (collect(ms.momentum.U.x.values), collect(ms.momentum.U.y.values),
     collect(ms.momentum.U.z.values), collect(r.potential.values))
else
    nothing
end
Us_x, Us_y, Us_z, Phis = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm, periodic_patches=[(:top, :bottom)])
model, config = cascade_case(dm; iterations=1)
result = potential_flow!(model, config; ncorrectors, pref=0.0)

n = dm.partition.n_owned
orig = dm.orig_cells
owned_err(loc, ser) = maximum(abs.(Array(loc)[1:n] .- ser[orig[1:n]]); init=0.0)

dux = owned_err(model.momentum.U.x.values, Us_x)
duy = owned_err(model.momentum.U.y.values, Us_y)
duz = owned_err(model.momentum.U.z.values, Us_z)
dphi = owned_err(result.potential.values, Phis)

@testset "potential_flow! 3D periodic distributed (rank $rank)" begin
    @test dux < 1e-6
    @test duy < 1e-6
    @test duz < 1e-6
    @test dphi < 1e-6
    # non-vacuous: the projection moved U off the uniform field it started from
    @test maximum(abs.(Array(model.momentum.U.x.values)[1:n] .- cascade_vel[1])) > 1e-4
end
rank == 0 && println("potential_flow cascade3D n=$(MPI.Comm_size(comm)) dux=$dux duy=$duy duz=$duz dphi=$dphi")
