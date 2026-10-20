# The serial `preconditioner=AMG()` setup runs unchanged on a distributed mesh: PETSc's GAMG
# substitutes for it with one warning, giving the fields of an explicit `GAMG()` run.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

gmesh = rank == 0 ? cavity_mesh() : nothing
dm = distribute(gmesh; comm=comm)
function fields(p_precon)
    model, config = incompressible_case(dm, cavity_bcs; iterations=20, p_precon)
    run!(model, config; pref=0.0)
    n = dm.partition.n_owned
    Array(model.momentum.U.x.values)[1:n], Array(model.momentum.p.values)[1:n]
end

substituted = r"AMG\(\) has no PETSc equivalent; substituting PETSc's GAMG\(\)"
u_amg, p_amg = rank == 0 ? (@test_logs (:warn, substituted) match_mode=:any fields(AMG())) : fields(AMG())
u_gamg, p_gamg = fields(GAMG())
@testset "preconditioner=AMG() runs as GAMG (rank $rank)" begin
    @test u_amg == u_gamg
    @test p_amg == p_gamg
end
