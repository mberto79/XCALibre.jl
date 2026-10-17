# Distributed BiCGStab after a shadow-vector breakdown; run per rank under mpiexec (runtests_mpi.jl).
# Unit lower-triangular matrix in the global numbering and b = e₁: the residual is orthogonal to the
# shadow after one step, and the solve must still converge.
using XCALibre, PETSc, MPI, Test, LinearAlgebra, Accessors
import XCALibre.Solve: sync!

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "laplace_case.jl"))

gmesh = rank == 0 ? fine2d_mesh() : nothing
dm = distribute(gmesh; comm=comm)

model, config = laplace_case(dm, fine2d_bcs)
setup = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0,
    rtol=0.0, atol=1e-12, itmax=1000)
@reset config.solvers = setup
deqn = build_deqn(dm, model, config)
eqn = deqn.eqn
T = model.energy.T
discretise!(eqn, T, config)
apply_boundary_conditions!(eqn, config.boundaries.T, nothing, 0.0, config)

part = dm.partition
n, l2g = part.n_owned, part.local_to_global
first_row = MPI.Allreduce(minimum(l2g[1:n]), min, comm)
A = XCALibre.ModelFramework._A(eqn)
rowptr, colval, nzval = _rowptr(A), _colval(A), _nzval(A)
for i ∈ 1:n, nzi ∈ rowptr[i]:(rowptr[i+1] - 1)
    gi, gj = l2g[i], l2g[colval[nzi]]
    nzval[nzi] = gi == gj ? 1.0 : (gj < gi ? -0.25 : 0.0)
end
b = XCALibre.ModelFramework._b(eqn)
b .= 0.0
foreach(i -> l2g[i] == first_row && (b[i] = 1.0), 1:n)

initialise!(T, 0.0)
XCALibre.Distribute._solve_owned!(deqn, T, nothing)
sync!(T, dm, config)

x = T.values
r2 = sum(i -> abs2(b[i] - sum(nzval[k]*x[colval[k]] for k ∈ rowptr[i]:(rowptr[i+1] - 1))), 1:n)
rnorm = sqrt(MPI.Allreduce(r2, +, comm))
rank == 0 && println("distributed BiCGStab after breakdown: ‖b - Ax‖ = $rnorm")

@testset "distributed BiCGStab survives a shadow-vector breakdown (rank $rank)" begin
    @test all(isfinite, x[1:n])
    @test rnorm < 1e-10
end
