# Distributed linear solves that stop without meeting atol/rtol warn once, from rank 0, with the
# equation label, the KSPConvergedReason, the iteration count and the residual norm: plain PETSc
# solves (also after a BiCGStab retry and in the transpose solve) and MixedF32 corrections.
using XCALibre, PETSc, MPI, Test
using Base.CoreLogging: Warn
import XCALibre.Distribute: _solve_owned!, passemble!, psolve_transpose!

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "laplace_case.jl"))

gmesh = rank == 0 ? fine2d_mesh() : nothing
dm = distribute(gmesh; comm=comm)
model, config = laplace_case(dm, fine2d_bcs)
T = model.energy.T

function wrapped_eqn(setup)
    (; schemes, boundaries) = config
    (; rhocp, rDf) = model.solid
    T_eqn = (
        Time{schemes.time}(rhocp, T) - Laplacian{schemes.laplacian}(rDf, T) == - Source(ScalarField(dm))
    ) → ScalarEquation(T, boundaries.T)
    deqn = wrap_eqn(T_eqn, dm, setup, config; label="T")
    discretise!(T_eqn, T, config)
    apply_boundary_conditions!(T_eqn, boundaries.T, nothing, 0.0, config)
    initialise!(T, 0.0)
    deqn
end

setup(; solver=Cg(), itmax=2, rtol=0.0, precision=FullPrecision()) = SolverSetup(solver=solver,
    preconditioner=Jacobi(), convergence=1e-10, relax=1.0, rtol=rtol, atol=1e-14, itmax=itmax,
    precision=precision)

# rank 0 emits exactly one warning per solve; the other ranks stay silent (the reason is global)
expect_warning(f, msg) = rank == 0 ?
    (@test_logs (:warn, msg) min_level=Warn f()) : (@test_logs min_level=Warn f())
expect_silence(f) = @test_logs min_level=Warn f()

const ITS = r"\[T\] did not converge in 2 iterations \(KSP_DIVERGED_ITS\): KSP residual norm = "

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
        # iterative refinement: the outer loop reports with the full-precision ‖b - Ax‖₂, the
        # corrections do not
        deqn = wrapped_eqn(setup(rtol=1e-12, precision=MixedF32()))
        expect_warning(() -> _solve_owned!(deqn, T, nothing), r"Cg \(MixedF32\) solve \[T\] did not " *
            r"converge in \d+ iterations \(iterative refinement stopped after 20 corrections\): ‖b - Ax‖₂ = ")

        deqn = wrapped_eqn(setup(itmax=1000, rtol=1e-8, precision=MixedF32()))
        expect_silence(() -> _solve_owned!(deqn, T, nothing))
    end
else
    rank == 0 && println("SKIP MixedF32 part of test_solve_warning: no Float32 PETSc library loaded")
end
