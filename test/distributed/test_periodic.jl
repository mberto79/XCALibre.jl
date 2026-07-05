# Periodic gate: quad40 as a periodic Couette channel (moving top lid, bottom wall,
# periodic inlet↔outlet), distributed vs serial. Cross-partition periodics via COLOCATION:
# distribute(periodic_patches=...) contracts matched owner-cell pairs in the partition
# graph, so periodic coupling stays rank-local and construct_periodic on the
# DistributedMesh works per rank exactly as in serial.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

periodic_bcs(mesh) = begin
    periodic = construct_periodic(mesh, CPU(), :inlet, :outlet)
    assign(region=mesh, (
        U = [Dirichlet(:top, [1.0, 0.0, 0.0]), Wall(:bottom, [0.0, 0.0, 0.0]),
             periodic...],
        p = [Zerogradient(:top), Zerogradient(:bottom), periodic...]))
end

iterations = 300
gmesh = rank == 0 ? cavity_mesh() : nothing
ref = if rank == 0
    model_s, config_s = incompressible_case(gmesh, periodic_bcs; iterations)
    simple!(model_s, config_s; pref=0.0)
    (collect(model_s.momentum.U.x.values), collect(model_s.momentum.U.y.values),
     collect(model_s.momentum.p.values))
else
    nothing
end
Us_x, Us_y, ps = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm, periodic_patches=[(:inlet, :outlet)])
model, config = incompressible_case(dm, periodic_bcs; iterations)
residuals = run!(model, config; pref=0.0)

dux, duy, dp = field_errors(dm, model, Us_x, Us_y, ps)
@testset "psimple periodic channel (rank $rank)" begin
    @test dux < 1e-6
    @test duy < 1e-6
    @test dp < 1e-6
end
rank == 0 && println("PERIODIC channel n=$(MPI.Comm_size(comm)) dux=$dux duy=$duy dp=$dp")
