# `preconditioner=AMG(...)` on a distributed mesh: Pairwise() runs PETSc's PCMG over XCALibre's own pairwise
# aggregates (the serial levels on one rank); SmoothedAggregation() runs GAMG with the same smoother and
# sweeps, so its fields equal a run that spells those GAMG options by hand.
using XCALibre, PETSc, MPI, Test
using PETSc: LibPETSc
using SparseArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

gmesh = rank == 0 ? cavity_mesh() : nothing
dm = distribute(gmesh; comm=comm)
function fields(p_precon; U_precon=Jacobi(), petsc_options="", iterations=20)
    model, config = incompressible_case(dm, cavity_bcs; iterations, p_precon, U_precon)
    run!(model, config; pref=0.0, petsc_options)
    n = dm.partition.n_owned
    Array(model.momentum.U.x.values)[1:n], Array(model.momentum.p.values)[1:n]
end

const EXT = Base.get_extension(XCALibre, :XCALibrePETScExt)
live_pc(label) = (s = EXT._LIVE[label]; (s.petsclib, LibPETSc.KSPGetPC(s.petsclib, s.ksp)))
pc_type(label) = ((lib, pc) = live_pc(label); LibPETSc.PCGetType(lib, pc))

# global rows of each PCMG level, finest first
function mg_rows(label)
    lib, pc = live_pc(label)
    L = Int(LibPETSc.PCMGGetLevels(lib, pc))
    map(L-1:-1:0) do l
        A, _ = LibPETSc.KSPGetOperators(lib, LibPETSc.PCMGGetSmoother(lib, pc, l))
        Int(first(LibPETSc.MatGetSize(lib, A)))
    end
end

# the solver's matrix on one rank, as a Julia sparse matrix
function petsc_matrix(label)
    s = EXT._LIVE[label]
    lib = s.petsclib
    n = s.n_owned
    I, J, V = Int[], Int[], Float64[]
    for r ∈ 0:n-1
        nc, cols, vals = LibPETSc.MatGetRow(lib, s.A, r)
        append!(I, fill(r + 1, nc)); append!(J, cols .+ 1); append!(V, vals)
        LibPETSc.MatRestoreRow(lib, s.A, r, nc, cols, vals)
    end
    sparse(I, J, V, n, n)
end

const GAMG_COMMON = "-pc_type gamg -pc_gamg_reuse_interpolation true -pc_mg_cycle_type v -pc_gamg_coarse_eq_limit 4096 " *
    "-pc_gamg_aggressive_coarsening 0 -pc_gamg_agg_nsmooths 1 -pc_gamg_threshold 0.0"

@testset "AMG(SmoothedAggregation(); smoother=$(nameof(typeof(s)))) runs as GAMG (rank $rank)" for (s, sweeps, opts) ∈ (
        (Jacobi(), 2, "-mg_levels_ksp_type richardson -mg_levels_pc_type jacobi " *
            "-mg_levels_ksp_richardson_scale 0.6666666666666666 -mg_levels_ksp_max_it 2"),
        (Chebyshev(), 1, "-mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi " *
            "-mg_levels_ksp_max_it 3 -mg_levels_ksp_chebyshev_esteig 0,0.11000000000000001,0,1.1"),
        (SymGaussSeidel(), (0, 2), "-mg_levels_ksp_type richardson -mg_levels_pc_type sor " *
            "-mg_levels_pc_sor_its 1 -mg_levels_pc_sor_local_symmetric true -pc_mg_distinct_smoothup true " *
            "-mg_levels_ksp_max_it 0 -mg_levels_up_ksp_max_it 2"))
    # refresh 1 on both, since a Jacobi() setup rebuilds every solve
    u_amg, p_amg = fields(AMG(SmoothedAggregation(); smoother=s, sweeps, refresh=1))
    u_ref, p_ref = fields(Jacobi(); petsc_options=(p = "$GAMG_COMMON $opts",))
    @test u_amg == u_ref
    @test p_amg == p_ref
end

# the translated settings reach PETSc: changing each one changes the fields
@testset "AMG settings take effect on a distributed mesh (rank $rank)" begin
    base = fields(AMG(SmoothedAggregation(); refresh=1))
    @test base != fields(AMG(SmoothedAggregation(); smoother=Chebyshev(), refresh=1))
    @test base != fields(AMG(SmoothedAggregation(); sweeps=(2, 3), refresh=1))
    @test base != fields(AMG(SmoothedAggregation()))
end

@testset "AMG(Pairwise()) runs as PCMG over the native aggregates (rank $rank)" begin
    a = AMG(Pairwise(); coarsest=50, refresh=1)
    # one p solve: the levels come from the matrix PETSc holds
    fields(a; iterations=1)
    @test pc_type("p") == "mg"
    rows = mg_rows("p")
    @test length(rows) > 2 && rows[1] == MPI.Allreduce(dm.partition.n_owned, +, comm) && issorted(rows; rev=true)
    if nranks == 1
        Am = XCALibre.Solve._amg_matrix(petsc_matrix("p"), Int)
        h = XCALibre.Solve.setup_hierarchy(Am, a, CPU(); log_diagnostics=false)
        @test rows == [size(l.A, 1) for l ∈ h.host_levels]
    end
    u_ref, p_ref = fields(Jacobi())
    u_sa, p_sa = fields(AMG(SmoothedAggregation(); refresh=1))
    u_pw, p_pw = fields(AMG(Pairwise(); refresh=1))
    @test isapprox(u_pw, u_ref; rtol=1e-5) && isapprox(p_pw, p_ref; rtol=1e-5)
    @test u_pw != u_sa
    # merges = 2 pairs twice per level
    u_pw2, p_pw2 = fields(AMG(Pairwise(2); coarsest=50, refresh=1))
    @test isapprox(u_pw2, u_ref; rtol=1e-5) && isapprox(p_pw2, p_ref; rtol=1e-5)
    @test length(mg_rows("p")) < length(rows)
end

# momentum is non-symmetric: pairwise aggregation handles it as is, GAMG symmetrises its graph
@testset "AMG($(nameof(typeof(c)))) with Bicgstab() on a non-symmetric system (rank $rank)" for c ∈
        (Pairwise(), SmoothedAggregation())
    u_amg, p_amg = fields(Jacobi(); U_precon=AMG(c; refresh=1))
    u_ref, p_ref = fields(Jacobi())
    @test all(isfinite, u_amg) && isapprox(u_amg, u_ref; rtol=1e-5) && isapprox(p_amg, p_ref; rtol=1e-5)
end

@testset "AMG(RugeStuben()) stops on a distributed mesh (rank $rank)" begin
    @test_throws r"use BoomerAMG\(\)" fields(AMG(RugeStuben()))
end
