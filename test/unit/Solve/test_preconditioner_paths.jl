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
        (AMG(coarsening=SmoothAggregation()), :same, :err, :err, :same),
        (AMG(coarsening=Geometric()),         :same, :err, :err, :same),
        (AMG(coarsening=RugeStuben()),        :same, :err, :err, :err),
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
end

@testset "SolverSetup refuses only what no path runs" begin
    @test_throws r"MixedF16 supports the Jacobi" mksetup(Cg(), DILU(), MixedF16())
    @test_throws r"mixed precision does not apply to AMG" mksetup(Cg(), AMG(coarsening=SmoothAggregation()), MixedF16())
    @test_throws r"OnDeviceKrylov requires solver=Richardson" mksetup(Cg(), AMG(coarsening=SmoothAggregation(), coarse_solve=OnDeviceKrylov()))
    # runs distributed, so the serial check stops it when the workspace is built
    for s ∈ (mksetup(Gmres(), AMG(coarsening=SmoothAggregation())), mksetup(Cg(), BoomerAMG()), mksetup(Richardson(), Jacobi()))
        @test_throws ArgumentError SV._check_serial(s)
    end
    @test mksetup(Richardson(), AMG(coarsening=SmoothAggregation(), coarse_solve=OnDeviceKrylov())).preconditioner.mode isa Richardson
    @test all(mksetup(s, pc).itmax == 1000 for (s, pc) ∈ ((Cg(), AMG(coarsening=SmoothAggregation())), (Cg(), BoomerAMG()), (Cg(), Jacobi())))
    # unset mode is solved as CG, so the OnDeviceKrylov guard also runs when the workspace is built
    @test_throws r"OnDeviceKrylov requires solver=Richardson" _workspace(AMG(coarsening=SmoothAggregation(), coarse_solve=OnDeviceKrylov()), zeros(4))
    @test_throws r"AMG needs a coarsening" AMG()
    @test AMG(coarsening=Geometric()).coarse_refresh_interval == 5
end

@testset "AMG settings PETSc's GAMG cannot run" begin
    conflicts(; kw...) = SV._gamg_conflicts(AMG(; kw...))
    @test isempty(conflicts(coarsening=SmoothAggregation()))
    @test isempty(conflicts(coarsening=SmoothAggregation(strength_threshold=0.05f0)))
    @test isempty(conflicts(coarsening=Geometric(), smoother=AMGChebyshev()))
    @test isempty(conflicts(coarsening=Geometric(), smoother=AMGGaussSeidel(sweep=AMGForwardSweep())))
    for kw ∈ ((coarsening=RugeStuben(),), (coarsening=Geometric(merge_levels=2),),
            (coarsening=SmoothAggregation(smoother_weight=1.0),), (coarsening=Geometric(), pre_sweeps=1),
            (coarsening=Geometric(), smoother=AMGChebyshev(), pre_sweeps=2, post_sweeps=2),
            (coarsening=Geometric(), smoother=AMGSOR(omega=1.2)), (coarsening=Geometric(), max_levels=5))
        @test length(conflicts(; kw...)) == 1
    end
    # runs serially, so SolverSetup accepts it; a distributed run stops when PETSc is set up
    @test mksetup(Cg(), AMG(coarsening=RugeStuben())) isa SolverSetup
    @test occursin("use BoomerAMG()", SV._pc_rule(AMG(coarsening=RugeStuben()), :petsc).msg)
    # native-only settings changed from their defaults are reported, not refused
    @test SV._native_only_set(AMG(coarsening=Geometric())) == Symbol[]
    @test SV._native_only_set(AMG(coarsening=Geometric(), fuse_levels=1, scale_correction=false)) ==
        [:scale_correction, :fuse_levels]
end

@testset "BoomerAMG runs only through PETSc" begin
    @test_throws r"distribute\(mesh\)" SV._check_serial(mksetup(Cg(), BoomerAMG()))
end

@testset "GAMG() is deprecated in favour of AMG" begin
    g = @test_logs (:warn, r"GAMG\(\) is deprecated; it returns AMG\(coarsening=Geometric\(\)") GAMG(agg_nsmooths=0, freeze=7)
    @test g isa AMG && g.coarsening isa Geometric && g.coarse_refresh_interval == 7
    @test GAMG().coarsening isa SmoothAggregation
    @test GAMG(threshold=0.02).coarsening.strength_threshold == 0.02
    old = GAMG()
    @test old.smoother isa AMGChebyshev && old.smoother.degree == 2 && old.max_coarse_rows == 50 &&
        old.coarse_refresh_interval == 25
    @test_throws r"petsc_options" GAMG(square_graph=1)
    @test_throws r"reuse_interpolation" GAMG(reuse_interpolation=false)
end

@testset "old spellings" begin
    old = @test_logs (:warn, r"use SolverSetup\(solver=Bicgstab\(\), preconditioner=AMG\(...\)\)") mksetup(
        AMG(coarsening=SmoothAggregation(), mode=Bicgstab()), Jacobi())
    @test old.solver isa Bicgstab && old.preconditioner isa AMG && old.preconditioner.mode isa Bicgstab
    @test mksetup(AMG(coarsening=SmoothAggregation()), Jacobi()).solver isa Cg
    @test mksetup(AMG(coarsening=SmoothAggregation(), mode=Richardson()), Jacobi()).solver isa Richardson
    @test (@test_logs (:warn, r"AMG\(mode=Bicgstab\(\)\) is deprecated and ignored") mksetup(
        Cg(), AMG(coarsening=SmoothAggregation(), mode=Bicgstab()))).preconditioner.mode isa Cg
    @test AMGSolver === Richardson
    # deprecation warnings print on the logging rank only (SolverSetup is built on every rank)
    SV._LOG_RANK[] = () -> false
    @test_logs mksetup(AMG(coarsening=SmoothAggregation(), mode=Bicgstab()), Jacobi())
    @test_logs mksetup(Cg(), AMG(coarsening=SmoothAggregation(), mode=Bicgstab()))
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
        amg = AMG(coarsening=SmoothAggregation(), smoother=AMGJacobi())
        eqn, x_new = laplace_values(mksetup(mode, amg; rtol=1e-10, atol=0.0))
        @test isapprox(x_new, x_ref; rtol=1e-6)
        # AMG owns its preconditioner: the equation's is inert, so nothing is built or refreshed
        @test eqn.preconditioner isa Preconditioner{SV.InertPreconditioner}
        @test eqn.solver isa AMGWorkspace && eqn.solver.converged
    end
    # the deprecated GAMG() is the native AMG with GAMG's default settings
    _, x_gamg = laplace_values(mksetup(Cg(), GAMG(); rtol=1e-10, atol=0.0))
    @test isapprox(x_gamg, x_ref; rtol=1e-6)
end
