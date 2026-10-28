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

# warm start: rtol is relative to D⁻¹(b - Ax₀), not b. CG with a non-Jacobi PC keeps PETSc's ‖Br‖
@testset "distributed $(nameof(typeof(s))) $(nameof(typeof(p))) stops at the serial target (rank $rank)" for
        (s, p) ∈ ((Cg(), Jacobi()), (Bicgstab(), Jacobi()), (Bicgstab(), DILU()), (Gmres(), Jacobi()), (Gmres(), DILU()))
    rtol = 1e-3
    deqn = wrapped_eqn(setup(solver=s, preconditioner=p, itmax=1000, rtol=rtol, atol=0.0))
    T.values .= 20 .+ 10 .* cos.(3 .* (1:length(T.values)))
    r0 = true_residual(deqn)
    expect_silence(() -> _solve_owned!(deqn, T, nothing))
    r = true_residual(deqn)
    @test r ≤ rtol*r0
    @test r ≥ 1e-3*rtol*r0
end

