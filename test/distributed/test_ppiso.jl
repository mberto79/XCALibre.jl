# Phase 5 ppiso! gate: transient cavity spin-up vs serial piso!, per rank under mpiexec.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

nsteps = 40
dt = 0.005

gmesh = rank == 0 ? cavity_mesh() : nothing
ref = if rank == 0
    model, config = incompressible_case(gmesh, cavity_bcs;
        iterations=nsteps, time=Transient(), time_step=dt)
    piso!(model, config; pref=0.0)
    (collect(model.momentum.U.x.values), collect(model.momentum.U.y.values),
     collect(model.momentum.p.values))
else
    nothing
end
Us_x, Us_y, ps = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
model, config = incompressible_case(dm, cavity_bcs;
    iterations=nsteps, time=Transient(), time_step=dt)
residuals = run!(model, config; pref=0.0)

dux, duy, dp = field_errors(dm, model, Us_x, Us_y, ps)
n = dm.partition.n_owned
nloc = n + dm.partition.n_ghost
orig = dm.orig_cells
px = model.momentum.U.x.values

@testset "ppiso cavity spin-up (rank $rank)" begin
    @test dux < 1e-6
    @test duy < 1e-6
    @test dp < 1e-6
    @test all(abs(px[i] - Us_x[orig[i]]) < 1e-6 for i ∈ n+1:nloc)
end
rank == 0 && println("PPISO cavity n=$(MPI.Comm_size(comm)) dux=$dux duy=$duy dp=$dp")
