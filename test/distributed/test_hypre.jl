# Phase 8 §3 gate: BoomerAMG (PETSc PCHYPRE) pressure preconditioner vs serial reference.
# Skips the solve comparison if the PETSc build lacks hypre; capability-error path always tested.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

ext = Base.get_extension(XCALibre, :XCALibrePETScExt)
petsclib = ext._petsclib(Float64)
PETSc.initialize(petsclib)
has_hypre = ext._petsc_has_pkg(petsclib, "hypre")

@testset "hypre capability guards (rank $rank)" begin
    # BoomerAMG is PETSc-only: serial constructor must refuse
    @test_throws ErrorException Preconditioner{BoomerAMG}(nothing)
end

gmesh = rank == 0 ? cavity_mesh() : nothing

if !has_hypre
    # build without hypre: BoomerAMG must fail loudly at solver construction, not mid-run
    dm = distribute(gmesh; comm=comm)
    model, config = incompressible_case(dm, cavity_bcs; iterations=1, p_precon=BoomerAMG())
    @testset "hypre missing errors early (rank $rank)" begin
        @test_throws ErrorException run!(model, config; pref=0.0)
    end
    rank == 0 && println("HYPRE gate SKIPPED (PETSc build lacks hypre); error path tested")
else
    iterations = 300
    ref = rank == 0 ? begin
        model, config = incompressible_case(gmesh, cavity_bcs; iterations)
        simple!(model, config; pref=0.0)
        (collect(model.momentum.U.x.values), collect(model.momentum.U.y.values),
         collect(model.momentum.p.values))
    end : nothing
    Us_x, Us_y, ps = MPI.bcast(ref, comm; root=0)

    dm = distribute(gmesh; comm=comm)
    model, config = incompressible_case(dm, cavity_bcs; iterations, p_precon=BoomerAMG())
    residuals = run!(model, config; pref=0.0)

    dux, duy, dp = field_errors(dm, model, Us_x, Us_y, ps)
    @testset "psimple cavity BoomerAMG (rank $rank)" begin
        @test dux < 1e-6
        @test duy < 1e-6
        @test dp < 1e-6
        @test maximum(residuals.p[iterations÷2:end]) < 1e-6
    end
    rank == 0 && println("HYPRE cavity n=$(MPI.Comm_size(comm)) dux=$dux duy=$duy dp=$dp")
end
