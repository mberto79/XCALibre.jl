# `preconditioner=AMG(...)` runs on a distributed mesh as PETSc's GAMG with the same coarsening,
# smoother and sweeps: the fields equal a run that spells those GAMG options by hand.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

gmesh = rank == 0 ? cavity_mesh() : nothing
dm = distribute(gmesh; comm=comm)
function fields(p_precon; U_precon=Jacobi(), petsc_options="")
    model, config = incompressible_case(dm, cavity_bcs; iterations=20, p_precon, U_precon)
    run!(model, config; pref=0.0, petsc_options)
    n = dm.partition.n_owned
    Array(model.momentum.U.x.values)[1:n], Array(model.momentum.p.values)[1:n]
end

const GAMG_COMMON = "-pc_type gamg -pc_gamg_reuse_interpolation true -pc_mg_cycle_type v -pc_gamg_coarse_eq_limit 4096 " *
    "-pc_gamg_aggressive_coarsening 0"
by_hand(opts) = (p = "$GAMG_COMMON $opts",)

@testset "AMG($(nameof(typeof(c))); smoother=$(nameof(typeof(s)))) runs as GAMG (rank $rank)" for (c, s, sweeps, opts) ∈ (
        (SmoothedAggregation(), Jacobi(), 2, "-pc_gamg_agg_nsmooths 1 -pc_gamg_threshold 0.0 " *
            "-mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -mg_levels_ksp_richardson_scale 0.6666666666666666 " *
            "-mg_levels_ksp_max_it 2"),
        (Pairwise(), Chebyshev(), 1, "-pc_gamg_agg_nsmooths 0 -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi " *
            "-mg_levels_ksp_max_it 3 -mg_levels_ksp_chebyshev_esteig 0,0.11000000000000001,0,1.1"),
        (Pairwise(), SymGaussSeidel(), (0, 2), "-pc_gamg_agg_nsmooths 0 -mg_levels_ksp_type richardson -mg_levels_pc_type sor " *
            "-mg_levels_pc_sor_its 1 -mg_levels_pc_sor_local_symmetric true -pc_mg_distinct_smoothup true " *
            "-mg_levels_ksp_max_it 0 -mg_levels_up_ksp_max_it 2"))
    # refresh 1 on both, since a Jacobi() setup rebuilds every solve
    u_amg, p_amg = fields(AMG(c; smoother=s, sweeps, refresh=1))
    u_ref, p_ref = fields(Jacobi(); petsc_options=by_hand(opts))
    @test u_amg == u_ref
    @test p_amg == p_ref
end

# the translated settings reach PETSc: changing each one changes the fields
@testset "AMG settings take effect on a distributed mesh (rank $rank)" begin
    base = fields(AMG(SmoothedAggregation(); refresh=1))
    @test base != fields(AMG(SmoothedAggregation(); smoother=Chebyshev(), refresh=1))
    @test base != fields(AMG(; refresh=1))
    @test base != fields(AMG(SmoothedAggregation(); sweeps=(2, 3), refresh=1))
    @test base != fields(AMG(SmoothedAggregation()))
end

# momentum is non-symmetric: with Bicgstab() GAMG symmetrises its aggregation graph
@testset "AMG with Bicgstab() on a non-symmetric system (rank $rank)" begin
    u_amg, p_amg = fields(Jacobi(); U_precon=AMG(refresh=1))
    u_ref, p_ref = fields(Jacobi())
    @test all(isfinite, u_amg) && isapprox(u_amg, u_ref; rtol=1e-5) && isapprox(p_amg, p_ref; rtol=1e-5)
end

@testset "AMG(RugeStuben()) stops on a distributed mesh (rank $rank)" begin
    @test_throws r"use BoomerAMG\(\)" fields(AMG(RugeStuben()))
end
