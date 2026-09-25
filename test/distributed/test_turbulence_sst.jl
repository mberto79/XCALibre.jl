# Distributed KOmegaSST (RANS) vs serial simple!, per rank under mpiexec. Mirrors
# test_turbulence.jl: exercises the SST blending functions + wall-distance solve through the
# distributed seam. Dirichlet wall BCs (no wall functions) keep the serial/parallel match clean.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "sst_case.jl"))

iterations = 100

gmesh = rank == 0 ? bfs_mesh() : nothing
ref = if rank == 0
    ms, cs = sst_case(gmesh, bfs_sst_bcs; iterations)
    simple!(ms, cs)
    (collect(ms.momentum.U.x.values), collect(ms.momentum.U.y.values),
     collect(ms.momentum.p.values), collect(ms.turbulence.k.values),
     collect(ms.turbulence.omega.values), collect(ms.turbulence.nut.values))
else
    nothing
end
Us_x, Us_y, ps, ks, ws, nuts = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
model, config = sst_case(dm, bfs_sst_bcs; iterations)
run!(model, config)

n = dm.partition.n_owned
nloc = n + dm.partition.n_ghost
orig = dm.orig_cells
owned_err(loc, ser) = maximum(abs.(Array(loc)[1:n] .- ser[orig[1:n]]); init=0.0)

dux = owned_err(model.momentum.U.x.values, Us_x)
duy = owned_err(model.momentum.U.y.values, Us_y)
dp = owned_err(model.momentum.p.values, ps)
dk = owned_err(model.turbulence.k.values, ks)
dw = owned_err(model.turbulence.omega.values, ws) / maximum(abs.(ws))
dnut = owned_err(model.turbulence.nut.values, nuts)

@testset "KOmegaSST distributed (rank $rank)" begin
    @test dux < 1e-6
    @test duy < 1e-6
    @test dp < 1e-6
    @test dk < 1e-6
    @test dw < 1e-6
    @test dnut < 1e-6
    # ghosts synced to the converged solution
    nv = model.turbulence.nut.values
    @test all(abs(nv[i] - nuts[orig[i]]) < 1e-6 for i ∈ n+1:nloc)
end
rank == 0 && println("SST bfs n=$(MPI.Comm_size(comm)) dux=$dux dp=$dp dk=$dk dw=$dw dnut=$dnut")
