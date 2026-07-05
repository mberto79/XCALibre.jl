# Phase 4 distributed Laplace gate; run per rank under mpiexec (see runtests_mpi.jl)
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
