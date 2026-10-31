# Distributed linear solves that stop without meeting atol/rtol warn once, from rank 0, with the
# equation label, the KSPConvergedReason, the iteration count and ‖D⁻¹(b - Ax)‖₂: plain PETSc solves
# (also after a BiCGStab retry and in the transpose solve) and MixedF32 corrections. PETSc stops on
# the serial rule, ‖D⁻¹(b - Ax)‖₂ ≤ atol + rtol·‖D⁻¹(b - Ax₀)‖₂, D = |diag(A)|.
using XCALibre, PETSc, MPI, Test
using Base.CoreLogging: Warn
import XCALibre.Distribute: _solve_owned!, passemble!, psolve_transpose!, _local_residual!

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "laplace_case.jl"))

gmesh = rank == 0 ? fine2d_mesh() : nothing
dm = distribute(gmesh; comm=comm)
model, config = laplace_case(dm, fine2d_bcs)
T = model.energy.T

function wrapped_eqn(setup; petsc_options="")
    (; schemes, boundaries) = config
    (; rhocp, rDf) = model.solid
    T_eqn = (
        Time{schemes.time}(rhocp, T) - Laplacian{schemes.laplacian}(rDf, T) == - Source(ScalarField(dm))
    ) → ScalarEquation(T, boundaries.T)
    deqn = wrap_eqn(T_eqn, dm, setup, config; label="T", petsc_options)
    discretise!(T_eqn, T, config)
    apply_boundary_conditions!(T_eqn, boundaries.T, nothing, 0.0, config)
    initialise!(T, 0.0)
    deqn
end

setup(; solver=Cg(), preconditioner=Jacobi(), itmax=2, rtol=0.0, atol=1e-14, precision=FullPrecision()) =
    SolverSetup(solver=solver, preconditioner=preconditioner, convergence=1e-10, relax=1.0, rtol=rtol,
        atol=atol, itmax=itmax, precision=precision)

# rank 0 emits exactly one warning per solve; the other ranks stay silent (the reason is global)
expect_warning(f, msg) = rank == 0 ?
    (@test_logs (:warn, msg) min_level=Warn f()) : (@test_logs min_level=Warn f())
expect_silence(f) = @test_logs min_level=Warn f()

const ITS = r"\[T\] did not converge in 2 iterations \(KSP_DIVERGED_ITS\): ‖D⁻¹\(b - Ax\)‖₂ = "

@testset "unconverged distributed solves warn (rank $rank)" begin
    deqn = wrapped_eqn(setup())
    expect_warning(() -> _solve_owned!(deqn, T, nothing), Regex("Cg solve " * ITS.pattern))
    passemble!(deqn.solver, deqn.eqn, deqn.partition; component=nothing)
    expect_warning(() -> psolve_transpose!(deqn.solver, T.values), Regex("Cg solve " * ITS.pattern))

    deqn = wrapped_eqn(setup(solver=Bicgstab()))
    expect_warning(() -> _solve_owned!(deqn, T, nothing), Regex("Bicgstab solve " * ITS.pattern))

    deqn = wrapped_eqn(setup(itmax=1000, rtol=1e-8))
    expect_silence(() -> _solve_owned!(deqn, T, nothing))
end

if any(l -> l.PetscScalar == Float32, PETSc.petsclibs)
    @testset "unconverged distributed MixedF32 solves warn (rank $rank)" begin
        # one correction (rtol above what Float32 attains): PETSc's own reason is reported
        deqn = wrapped_eqn(setup(rtol=1e-3, precision=MixedF32()))
        expect_warning(() -> _solve_owned!(deqn, T, nothing), Regex("Cg \\(MixedF32\\) solve " * ITS.pattern))
        # iterative refinement: the outer loop reports with the full-precision ‖D⁻¹(b - Ax)‖₂, the
        # corrections do not
        deqn = wrapped_eqn(setup(rtol=1e-12, precision=MixedF32()))
        expect_warning(() -> _solve_owned!(deqn, T, nothing), r"Cg \(MixedF32\) solve \[T\] did not " *
            r"converge in \d+ iterations \(iterative refinement stopped after 20 corrections\): ‖D⁻¹\(b - Ax\)‖₂ = ")

        deqn = wrapped_eqn(setup(itmax=1000, rtol=1e-8, precision=MixedF32()))
        expect_silence(() -> _solve_owned!(deqn, T, nothing))
    end
else
    rank == 0 && println("SKIP MixedF32 part of test_solve_warning: no Float32 PETSc library loaded")
end

# global ‖D⁻¹(b - Ax)‖₂ over owned rows, ghosts refreshed first
function true_residual(deqn)
    sync!(T, dm, config)
    _local_residual!(deqn, nothing, config)
    n = deqn.partition.n_owned
    (; A, R, diag_nz) = deqn.eqn.equation
    d = abs.(XCALibre.ModelFramework._nzval(A)[diag_nz[1:n]])
    sqrt(MPI.Allreduce(sum(abs2, R[1:n] ./ d), +, comm))
end

warm!() = (T.values .= 20 .+ 10 .* cos.(3 .* (1:length(T.values))); T.values)

ext = Base.get_extension(XCALibre, :XCALibrePETScExt)
const HYPRE = ext._petsc_has_pkg(ext._petsclib(Float64, 0), "hypre")

# warm start: rtol is relative to D⁻¹(b - Ax₀), not b. Jacobi monitors ‖D⁻¹r‖, block ILU solves the
# row-scaled system, CG with another PC and AMG PCs check ‖D⁻¹r‖ explicitly near the target
const CASES = ((Cg(), Jacobi()), (Cg(), DILU()), (Cg(), GAMG()),
    (Bicgstab(), Jacobi()), (Bicgstab(), DILU()), (Bicgstab(), GAMG()), (Cgs(), Jacobi()), (Cgs(), DILU()),
    (Gmres(), Jacobi()), (Gmres(), DILU()), (Gmres(), GAMG()),
    (HYPRE ? ((Cg(), BoomerAMG()), (Bicgstab(), BoomerAMG())) : ())...)
case_name((s, p)) = "$(nameof(typeof(s))) $(nameof(typeof(p)))"

function solve_warm(s, p; rtol, atol, precision=FullPrecision(), petsc_options="")
    deqn = wrapped_eqn(setup(solver=s, preconditioner=p, itmax=1000, rtol=rtol, atol=atol, precision=precision);
        petsc_options)
    warm!()
    r0 = true_residual(deqn)
    expect_silence(() -> _solve_owned!(deqn, T, nothing))
    r0, true_residual(deqn)
end

@testset "distributed $(case_name(c)) stops at the serial target (rank $rank)" for c ∈ CASES
    rtol = 1e-3
    r0, r = solve_warm(c...; rtol=rtol, atol=0.0)
    @test r ≤ rtol*r0
    @test r ≥ 1e-3*rtol*r0
end

const R0 = let deqn = wrapped_eqn(setup(itmax=1000))
    warm!()
    true_residual(deqn)
end

@testset "distributed $(case_name(c)) stops at atol (rank $rank)" for c ∈ CASES
    _, r = solve_warm(c...; rtol=0.0, atol=1e-4*R0)
    @test r ≤ 1e-4*R0
end

# atol = ρ·r0 with rtol = ρ is rtol = 2ρ alone (PETSc's own test would take the max, half the target)
@testset "distributed $(case_name(c)) adds atol and rtol (rank $rank)" for c ∈ CASES
    ρ = 1e-3
    _, r_sum = solve_warm(c...; rtol=ρ, atol=ρ*R0)
    _, r_ref = solve_warm(c...; rtol=2ρ, atol=0.0)
    @test r_sum ≈ r_ref rtol=1e-6
    @test r_sum ≤ 2ρ*R0
end

# KSPs outside the curated four keep PETSc's side and norm: no Jacobi side options forced on them
@testset "distributed passthrough -ksp_type $k stops at the serial target (rank $rank)" for k ∈ ("fgmres", "gcr")
    r0, r = solve_warm(Gmres(), Jacobi(); rtol=1e-3, atol=0.0, petsc_options="-ksp_type $k")
    @test r ≤ 1e-3*r0
    @test r ≥ 1e-6*r0
end

if any(l -> l.PetscScalar == Float32, PETSc.petsclibs)
    # the serial MixedF32 rule: corrections stop at the refinement's target, x₀ already converged skips the solve
    @testset "distributed MixedF32 $(case_name(c)) stops at the serial target (rank $rank)" for
            c ∈ ((Cg(), Jacobi()), (Cg(), GAMG()), (Bicgstab(), DILU()))
        r0, r = solve_warm(c...; rtol=1e-3, atol=0.0, precision=MixedF32())
        @test r ≤ 1.05e-3*r0
        @test r ≥ 1e-6*r0
        r0, r = solve_warm(c...; rtol=1e-8, atol=0.0, precision=MixedF32())
        @test r ≤ 1.05e-8*r0
        _, r_sum = solve_warm(c...; rtol=1e-3, atol=1e-3*R0, precision=MixedF32())
        _, r_ref = solve_warm(c...; rtol=2e-3, atol=0.0, precision=MixedF32())
        @test r_sum ≈ r_ref rtol=1e-6
    end

    @testset "distributed MixedF32 skips a converged x₀ (rank $rank)" begin
        deqn = wrapped_eqn(setup(itmax=1000, rtol=1e-3, atol=10*R0, precision=MixedF32()))
        x0 = copy(warm!())
        expect_silence(() -> _solve_owned!(deqn, T, nothing))
        n = deqn.partition.n_owned
        @test T.values[1:n] == x0[1:n]
    end
end
