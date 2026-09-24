# Distributed Laplace gate; run per rank under mpiexec (see runtests_mpi.jl)
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "laplace_case.jl"))

gmesh = rank == 0 ? box_mesh() : nothing

# serial reference (full laplace! path) on rank 0
ref = if rank == 0
    model, config = laplace_case(gmesh, box_bcs)
    Rs = laplace!(model, config)
    Rs = Rs isa NamedTuple ? Rs.T : Rs
    (collect(model.energy.T.values), collect(Rs))
else
    nothing
end
Tserial, Rserial = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
model, config = laplace_case(dm, box_bcs)
residuals = run!(model, config)

n = dm.partition.n_owned
nloc = n + dm.partition.n_ghost
orig = dm.orig_cells
Tloc = model.energy.T.values
conv = config.solvers.convergence
kconv = findfirst(<=(conv), Rserial)

@testset "Phase 4 laplace (rank $rank)" begin
    # converged field matches serial per cell
    @test maximum(abs.(Tloc[1:n] .- Tserial[orig[1:n]]); init=0.0) < 1e-6

    # global relative L2 error
    num = MPI.Allreduce(sum(abs2, Tloc[1:n] .- Tserial[orig[1:n]]; init=0.0), +, comm)
    den = MPI.Allreduce(sum(abs2, Tserial[orig[1:n]]; init=0.0), +, comm)
    @test sqrt(num / den) < 1e-8

    # ghosts synced to the converged solution
    @test all(abs(Tloc[i] - Tserial[orig[i]]) < 1e-6 for i ∈ n+1:nloc)

    # residual history: same convergence iteration; sub-tolerance values are solver noise,
    # so only pre-convergence entries are compared (linear problem => usually none)
    @test findfirst(<=(conv), residuals.T) == kconv
    @test residuals.T[kconv] <= conv
    @test all(isapprox.(residuals.T[1:kconv-1], Rserial[1:kconv-1]; rtol=1e-2))
end

# a solve stopped at itmax reports it once, from rank 0 only
model1, config1 = laplace_case(dm, box_bcs; iterations=2, itmax=1)
@testset "PETSc reports an unconverged solve (rank $rank)" begin
    if rank == 0
        @test_logs (:warn, r"KSP_DIVERGED_ITS after 1 iterations") match_mode=:any run!(model1, config1)
    else
        @test_logs min_level=Base.CoreLogging.Warn run!(model1, config1)
    end
end

# a frozen hierarchy must stay one fixed SPD operator when the matrix grows past the one it was
# built from: stale coarse operators paired with the new fine matrix make the V-cycle indefinite
ext = Base.get_extension(XCALibre, :XCALibrePETScExt)
import XCALibre.ModelFramework: _A, _b, _rowptr, _colval, _nzval
model2, config2 = laplace_case(dm, box_bcs; precon=GAMG(freeze=5), itmax=200)
deqn = build_deqn(dm, model2, config2)
A = _A(deqn.eqn); rowptr, colval, nzval = _rowptr(A), _colval(A), _nzval(A)
function fill_spd!(shift)
    for r ∈ 1:n
        ks = rowptr[r]:rowptr[r+1]-1
        for k ∈ ks
            nzval[k] = colval[k] == r ? (length(ks) - 1)*(1 + shift) : -1.0
        end
    end
end
_b(deqn.eqn, nothing) .= 1.0
reasons = map((1e-3, 10.0, 10.0)) do shift
    fill_spd!(shift)
    XCALibre.Distribute.passemble!(deqn.solver, deqn.eqn, deqn.partition)
    XCALibre.Distribute.psolve!(deqn.solver, zeros(nloc))
    Int(ext.LibPETSc.KSPGetConvergedReason(deqn.solver.petsclib, deqn.solver.ksp))
end
@testset "frozen GAMG stays SPD across matrix changes (rank $rank)" begin
    @test all(>(0), reasons)
end
