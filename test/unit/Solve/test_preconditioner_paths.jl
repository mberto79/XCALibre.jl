using XCALibre
using Accessors
using Test

# `solver` is the outer iteration and `preconditioner` the method on every path; each preconditioner
# runs as named, as a named substitute (one warning) or stops with guidance (PRECONDITIONER_PATHS).
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
        (AMG(),          :same, :err,  :err,  :sub),
        (GAMG(),         :sub,  :err,  :err,  :same),
        (BoomerAMG(),    :sub,  :err,  :err,  :same))
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
    @test_throws r"mixed precision does not apply to AMG" mksetup(Cg(), AMG(), MixedF16())
    @test_throws r"OnDeviceKrylov requires solver=Richardson" mksetup(Cg(), AMG(coarse_solve=OnDeviceKrylov()))
    # runs distributed, so the serial check stops it when the workspace is built
    for s ∈ (mksetup(Gmres(), AMG()), mksetup(Cg(), GAMG(), MixedF32()), mksetup(Richardson(), Jacobi()))
        @test_throws ArgumentError SV._check_serial(s)
    end
    @test mksetup(Richardson(), AMG(coarse_solve=OnDeviceKrylov())).preconditioner.mode isa Richardson
    @test mksetup(Cg(), AMG()).itmax == 200 && mksetup(Cg(), GAMG()).itmax == 1000
end

@testset "GAMG and BoomerAMG on a serial mesh run the native AMG" begin
    for pc ∈ (GAMG(), BoomerAMG()), solver ∈ (Cg(), Bicgstab(), Richardson())
        amg = SV._native_amg(mksetup(solver, pc))
        @test amg isa AMG && amg.mode isa typeof(solver)
    end
    @test_logs (:warn, r"GAMG\(\) is not available on a serial mesh; substituting XCALibre's native AMG") SV._check_serial(mksetup(Cg(), GAMG()))
end

@testset "old spellings" begin
    old = @test_logs (:warn, r"use SolverSetup\(solver=Bicgstab\(\), preconditioner=AMG\(...\)\)") mksetup(
        AMG(mode=Bicgstab()), Jacobi())
    @test old.solver isa Bicgstab && old.preconditioner isa AMG && old.preconditioner.mode isa Bicgstab
    @test mksetup(AMG(), Jacobi()).solver isa Cg
    @test mksetup(AMG(mode=Richardson()), Jacobi()).solver isa Richardson
    @test (@test_logs (:warn, r"AMG\(mode=Bicgstab\(\)\) is deprecated and ignored") mksetup(
        Cg(), AMG(mode=Bicgstab()))).preconditioner.mode isa Cg
    @test AMGSolver === Richardson
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

@testset "AMG as the preconditioner matches solver=AMG" begin
    for mode ∈ (Cg(), Bicgstab(), Richardson())
        amg = AMG(coarsening=SmoothAggregation(), smoother=AMGJacobi())
        new = mksetup(mode, amg; rtol=1e-10, atol=0.0)
        old = mksetup(SV._amg_with_mode(amg, mode), Jacobi(); rtol=1e-10, atol=0.0)
        eqn, x_new = laplace_values(new)
        _, x_old = laplace_values(old)
        @test x_new == x_old
        # AMG owns its preconditioner: the equation's is inert, so nothing is built or refreshed
        @test eqn.preconditioner isa Preconditioner{SV.InertPreconditioner}
        @test eqn.solver isa AMGWorkspace && eqn.solver.converged
    end
    # GAMG on a serial mesh solves with the native AMG
    _, x_gamg = laplace_values(mksetup(Cg(), GAMG(); rtol=1e-10, atol=0.0))
    _, x_amg = laplace_values(mksetup(Cg(), AMG(); rtol=1e-10, atol=0.0))
    @test x_gamg == x_amg
end
