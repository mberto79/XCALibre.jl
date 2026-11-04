using XCALibre
using Accessors
using Test

# `solver` is the outer iteration and `preconditioner` the method on every path; each preconditioner
# runs as named, as a named substitute (one warning) or stops with guidance (PRECONDITIONER_PATHS).
# AMG runs natively on a serial mesh and as PETSc's GAMG on a distributed one; BoomerAMG only in PETSc.
const SV = XCALibre.Solve

mksetup(solver, pc, precision=FullPrecision(); kw...) =
    SolverSetup(; solver, preconditioner=pc, precision, convergence=1e-8, relax=1.0, kw...)

is_same(r) = r === :same

@testset "serial and PETSc column of each preconditioner" begin
    # (preconditioner, full, MixedF32, MixedF16, PETSc); :sub = named substitute, :err = error
    table = (
        (Jacobi(),       :same, :same, :same, :same),
        (NormDiagonal(), :same, :same, :err,  :sub),
        (DILU(),         :same, :same, :err,  :sub),
        (ILU0GPU(),      :same, :same, :err,  :sub),
        (IC0GPU(),       :same, :same, :err,  :sub),
        (AMG(),                      :same, :same, :err, :same),
        (AMG(SmoothedAggregation()), :same, :same, :err, :same),
        (AMG(RugeStuben()),          :same, :same, :err, :err),
        (BoomerAMG(),    :err,  :err,  :err,  :same))
    kind(r) = r === :same ? :same : r isa SV.Substitute ? :sub : :err
    for (pc, full, f32, f16, petsc) ∈ table
        @test kind(SV._pc_rule(pc, :full)) == full
        @test kind(SV._pc_rule(pc, :f32)) == f32
        @test kind(SV._pc_rule(pc, :f16)) == f16
        @test kind(SV._pc_rule(pc, :petsc)) == petsc
    end
    # a user-defined type runs as named serially and needs petsc_options under PETSc
    @test SV._pc_rule(SV.InertPreconditioner(), :full) === :same
    @test SV._pc_rule(SV.InertPreconditioner(), :petsc) isa SV.Unsupported
end

@testset "outer iterations of each path" begin
    for (solver, krylov, amg, mixed) ∈ (
            (Cg(), true, true, true), (Bicgstab(), true, true, true), (Cgs(), true, false, true),
            (Gmres(), true, false, true), (Richardson(), false, true, false))
        @test is_same(SV._solver_rule(solver, :krylov)) == krylov
        @test is_same(SV._solver_rule(solver, :amg)) == amg
        @test is_same(SV._solver_rule(solver, :mixed)) == mixed
        @test is_same(SV._solver_rule(solver, :petsc))
    end
    @test SV._solver_column(MixedF32(), AMG()) === :amg
end

@testset "SolverSetup refuses only what no path runs" begin
    @test_throws r"MixedF16 supports the Jacobi" mksetup(Cg(), DILU(), MixedF16())
    @test_throws r"MixedF16 does not apply to AMG" mksetup(Cg(), AMG(), MixedF16())
    @test_throws r"needs solver=Richardson" mksetup(Cg(), AMG(native=NativeAMG(coarse=Iterative())))
    # runs distributed, so the serial check stops it when the workspace is built
    for s ∈ (mksetup(Gmres(), AMG()), mksetup(Cg(), BoomerAMG()), mksetup(Richardson(), Jacobi()))
        @test_throws ArgumentError SV._check_serial(s)
    end
    @test mksetup(Richardson(), AMG(native=NativeAMG(coarse=Iterative()))).preconditioner isa AMG
    @test all(mksetup(s, pc).itmax == 1000 for (s, pc) ∈ ((Cg(), AMG()), (Cg(), BoomerAMG()), (Cg(), Jacobi())))
    @test_throws r"needs solver=Richardson" _workspace(AMG(native=NativeAMG(coarse=Iterative())), zeros(4))
end

@testset "AMG settings" begin
    a = AMG()
    @test a.coarsening isa Pairwise && a.smoother isa Jacobi && a.sweeps == Sweeps(2, 2) && a.refresh == 5
    @test AMG(smoother=Chebyshev()).sweeps == Sweeps(1, 1)
    @test AMG(sweeps=(0, 2)).sweeps == Sweeps(0, 2)
    @test SV._pre_sweeps(Sweeps(0, 2; grow=1, max=3), 1) == 0 && SV._pre_sweeps(Sweeps(0, 2; grow=1, max=3), 5) == 3
    @test SV._post_sweeps(Sweeps(0, 2; grow=1, max=3), 2) == 3
    @test_throws ArgumentError Sweeps(0, 0)
    @test_throws ArgumentError AMG(smoother=DILU())
    @test_throws ArgumentError Pairwise(0)
    @test_throws ArgumentError SOR(2.0)
    @test_throws r"Smoothed coarse solve takes" NativeAMG(coarse=Smoothed(SymGaussSeidel()))
    @test SV._amg_storage(SV._with_storage(a, Float32)) === Float32
end

@testset "AMG settings PETSc's GAMG cannot run" begin
    refused(a, solver=Cg()) = last(SV._lower_gamg(a, solver))
    options(a, solver=Cg()) = first(SV._lower_gamg(a, solver))
    @test isempty(refused(AMG()))
    @test isempty(refused(AMG(SmoothedAggregation(threshold=0.05f0))))
    @test isempty(refused(AMG(smoother=Chebyshev(), sweeps=1)))
    @test isempty(refused(AMG(smoother=GaussSeidel())))
    @test isempty(refused(AMG(sweeps=(1, 2))))
    for a ∈ (AMG(RugeStuben()), AMG(Pairwise(2)), AMG(SmoothedAggregation(weight=1.0)),
            AMG(smoother=Chebyshev(), sweeps=2), AMG(smoother=SSOR(1.2)), AMG(sweeps=Sweeps(1, 1; grow=1)))
        @test length(refused(a)) == 1
    end
    # aggressive coarsening off mirrors the native levels; non-symmetric systems symmetrise the graph
    @test options(AMG()).pc_gamg_aggressive_coarsening == 0 && options(AMG()).pc_gamg_agg_nsmooths == 0
    @test !haskey(options(AMG()), :pc_gamg_graph_symmetrize)
    @test options(AMG(), Bicgstab()).pc_gamg_graph_symmetrize
    @test options(AMG(coarsest=100)).pc_gamg_coarse_eq_limit == 100
    o = options(AMG(sweeps=(1, 3)))
    @test o.pc_mg_distinct_smoothup && o.mg_levels_ksp_max_it == 1 && o.mg_levels_up_ksp_max_it == 3
    @test options(AMG(smoother=SymGaussSeidel())).mg_levels_pc_sor_local_symmetric
    # runs serially, so SolverSetup accepts it; a distributed run stops when PETSc is set up
    @test mksetup(Cg(), AMG(RugeStuben())) isa SolverSetup
    @test occursin("use BoomerAMG()", SV._pc_rule(AMG(RugeStuben()), :petsc).msg)
    # NativeAMG settings changed from their defaults are reported, not refused
    @test SV._native_only(AMG()) == ""
    @test occursin("fuse", SV._native_only(AMG(native=NativeAMG(fuse=1))))
end

@testset "BoomerAMG" begin
    @test_throws r"distribute\(mesh\)" SV._check_serial(mksetup(Cg(), BoomerAMG()))
    h = SV._lower_hypre(BoomerAMG(coarsen=PMIS(), sweeps=(1, 2), smoother=GaussSeidel()))
    @test h.pc_hypre_boomeramg_coarsen_type == "PMIS" && h.pc_hypre_boomeramg_relax_type_all == "SOR/Jacobi"
    @test h.pc_hypre_boomeramg_grid_sweeps_down == 1 && h.pc_hypre_boomeramg_grid_sweeps_up == 2
    @test BoomerAMG(refresh=3).refresh == 3
    @test_throws r"petsc_options" BoomerAMG(strong_threshold=0.5)
    @test_throws ArgumentError BoomerAMG(smoother=Jacobi())
end

@testset "GAMG() is deprecated in favour of AMG" begin
    g = @test_logs (:warn, r"GAMG\(\) is deprecated; use AMG\(...\). This GAMG\(\) is AMG\(Pairwise\(\)") GAMG(agg_nsmooths=0, freeze=7)
    @test g isa AMG && g.coarsening isa Pairwise && g.refresh == 7
    @test GAMG().coarsening isa SmoothedAggregation
    @test GAMG(threshold=0.02).coarsening.threshold == 0.02
    old = GAMG()
    @test old.smoother isa Chebyshev && old.smoother.degree == 2 && old.coarsest == 50 && old.refresh == 25
    @test_throws r"petsc_options" GAMG(square_graph=1)
    @test_throws r"reuse_interpolation" GAMG(reuse_interpolation=false)
end

@testset "old spellings" begin
    old = @test_logs (:warn, r"use SolverSetup\(solver=Bicgstab\(\), preconditioner=AMG\(...\)\)") match_mode=:any mksetup(
        AMG(coarsening=Pairwise(), mode=Bicgstab()), Jacobi())
    @test old.solver isa Bicgstab && old.preconditioner isa AMG
    @test mksetup(AMG(), Jacobi()).solver isa Cg
    @test mksetup(AMG(mode=Richardson()), Jacobi()).solver isa Richardson
    @test (@test_logs (:warn, r"AMG\(mode=Bicgstab\(\)\) is deprecated and ignored") match_mode=:any mksetup(
        Cg(), AMG(mode=Bicgstab()))).preconditioner isa AMG
    @test AMGSolver === Richardson
    a = @test_logs (:warn, r"is deprecated") match_mode=:any AMG(coarsening=Pairwise(), pre_sweeps=1,
        coarse_refresh_interval=3, max_coarse_rows=100, fuse_levels=1, coarse_solve=CPU())
    @test a.sweeps == Sweeps(1, 2) && a.refresh == 3 && a.coarsest == 100 && a.native.fuse == 1 && a.native.coarse == Direct(max_rows=0)
    @test_throws r"precision=MixedF32" AMG(coarse_storage=Float32)
    @test (@test_logs (:warn, r"Geometric\(\) is deprecated") Geometric(merge_levels=2)) == Pairwise(2)
    @test (@test_logs (:warn,) AMGJacobi(omega=1.0)) == Jacobi(weight=1.0)
    @test (@test_logs (:warn,) AMGSOR(omega=1.2)) == SSOR(1.2)
    @test (@test_logs (:warn,) OnDeviceJacobi(iterations=4)) == Smoothed(Jacobi(); sweeps=4)
    # deprecation warnings print on the logging rank only (SolverSetup is built on every rank)
    SV._LOG_RANK[] = () -> false
    @test_logs mksetup(AMG(mode=Bicgstab()), Jacobi())
    @test_logs mksetup(Cg(), AMG(mode=Bicgstab()))
    SV._LOG_RANK[] = () -> true
end

# NEW SECTION: solves

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "finer_mesh_laplace.unv"))
hardware = Hardware(backend=CPU(), workgroup=1024)
model = Physics(time=Steady(), solid=Solid{Uniform}(k=1.0), energy=Energy{Conduction}(), domain=mesh)
BCs = assign(region=mesh, (T = [
    Dirichlet(:left_wall, 50.0), Zerogradient(:right_wall), Dirichlet(:bottom_wall, 10.0), Zerogradient(:upper_wall)],))
schemes = (T = Schemes(laplacian = Linear),)

function laplace_values(s)
    config = Configuration(solvers=(T=s,), schemes=schemes,
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1), hardware=hardware, boundaries=BCs)
    T = model.energy.T
    T_eqn = (- Laplacian{schemes.T.laplacian}(model.solid.rDf, T) == - Source(ScalarField(mesh))
        ) → ScalarEquation(T, config.boundaries.T)
    discretise!(T_eqn, T, config)
    apply_boundary_conditions!(T_eqn, config.boundaries.T, nothing, 0.0, config)
    @reset T_eqn.preconditioner = set_preconditioner(s, T_eqn)
    @reset T_eqn.solver = _workspace(s, T_eqn)
    update_preconditioner!(T_eqn.preconditioner, mesh, config)
    initialise!(T, 0.0)
    solve_system!(T_eqn, s, T, nothing, config)
    T_eqn, copy(T.values)
end

@testset "AMG as the preconditioner solves the system" begin
    _, x_ref = laplace_values(mksetup(Cg(), Jacobi(); rtol=1e-12, atol=0.0))
    for mode ∈ (Cg(), Bicgstab(), Richardson())
        eqn, x_new = laplace_values(mksetup(mode, AMG(SmoothedAggregation()); rtol=1e-10, atol=0.0))
        @test isapprox(x_new, x_ref; rtol=1e-6)
        # AMG owns its preconditioner: the equation's is inert, so nothing is built or refreshed
        @test eqn.preconditioner isa Preconditioner{SV.InertPreconditioner}
        @test eqn.solver isa AMGWorkspace && eqn.solver.converged
    end
    # MixedF32 stores the coarse levels in Float32 and keeps the solve in full precision
    eqn, x32 = laplace_values(mksetup(Cg(), AMG(), MixedF32(); rtol=1e-10, atol=0.0))
    @test isapprox(x32, x_ref; rtol=1e-6) && eqn.solver isa AMGWorkspace
    # the deprecated GAMG() is the native AMG with GAMG's default settings
    _, x_gamg = laplace_values(mksetup(Cg(), GAMG(); rtol=1e-10, atol=0.0))
    @test isapprox(x_gamg, x_ref; rtol=1e-6)
end

# momentum is a vector equation solved per component: AMG with Bicgstab() runs it like Jacobi()
@testset "AMG on a vector equation" begin
    bfs = UNV2D_mesh(pkgdir(XCALibre, "examples/0_GRIDS/backwardFacingStep_10mm.unv"), scale=0.001)
    run_bfs(Upc) = begin
        model = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=1e-3), turbulence=RANS{Laminar}(),
            energy=Energy{Isothermal}(), domain=bfs)
        bcs = assign(region=bfs, (
            U = [Dirichlet(:inlet, [0.5, 0.0, 0.0]), Extrapolated(:outlet), Wall(:wall, [0.0, 0.0, 0.0]), Symmetry(:top)],
            p = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0), Extrapolated(:wall), Symmetry(:top)]))
        solvers = (
            U = SolverSetup(solver=Bicgstab(), preconditioner=Upc, convergence=1e-10, relax=0.8, rtol=1e-6, atol=0.0),
            p = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-10, relax=0.2, rtol=1e-6, atol=0.0))
        config = Configuration(solvers=solvers, schemes=(U=Schemes(divergence=Linear), p=Schemes()), boundaries=bcs,
            runtime=Runtime(iterations=5, write_interval=-1, time_step=1), hardware=Hardware(backend=CPU(), workgroup=1024))
        initialise!(model.momentum.U, [0.5, 0.0, 0.0]); initialise!(model.momentum.p, 0.0)
        run!(model, config; progress=false)
        copy(model.momentum.U.x.values)
    end
    @test isapprox(run_bfs(AMG()), run_bfs(Jacobi()); rtol=1e-4)
end
