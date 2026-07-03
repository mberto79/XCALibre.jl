# Phase 4 perf gate: hot-path allocation budgets + type stability; run per rank under mpiexec.
# Budgets are ~2x the values measured at introduction (recorded in dev/STATE.md); a blown
# budget = an allocation regression on the per-iteration path — fix it, don't raise the budget.
using XCALibre, PETSc, MPI, Test
import XCALibre.Solve: residual, solve_equation!, solve_system!

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "laplace_case.jl"))

gmesh = rank == 0 ? box_mesh() : nothing
dm = distribute(gmesh; comm=comm)
model, config = laplace_case(dm, box_bcs)
deqn = build_deqn(dm, model, config)
T = model.energy.T
(; backend, workgroup) = config.hardware

solve_iter!() = solve_equation!(deqn, T, config.boundaries.T, config.solvers, config; time=1.0)

# warmup: JIT + PETSc setup + halo request allocation
solve_iter!(); solve_iter!()

a_halo = @allocated halo_exchange!(T, deqn.halo, backend, workgroup)
a_asm = @allocated passemble!(deqn.solver, deqn.eqn, deqn.partition; component=nothing)
a_slv = @allocated psolve!(deqn.solver, T.values)
a_res = @allocated residual(deqn, nothing, config)
a_eqn = @allocated solve_iter!()

t0 = MPI.Wtime(); solve_iter!(); t_iter = MPI.Wtime() - t0

println("PERF rank=$rank halo=$a_halo passemble=$a_asm psolve=$a_slv residual=$a_res " *
    "solve_eqn=$a_eqn t_iter_ms=$(round(t_iter*1000, digits=2))")

@testset "Phase 4 perf (rank $rank)" begin
    # allocation budgets (bytes) on the per-iteration hot path
    # halo cost is per neighbour patch (requests + pack/unpack launches), not per cell
    @test a_halo <= 4_096 + 4_096 * max(1, length(dm.procs))
    @test a_asm <= 16_384
    @test a_slv <= 8_192
    @test a_res <= 1_024
    @test a_eqn <= 65_536

    # type stability of new distributed entry points
    @test (@inferred residual(deqn, nothing, config)) isa Float64
    @test (@inferred solve_system!(deqn, config.solvers, T, nothing, config)) isa Float64

    # Phase 2 reductions stay inferred
    dphi = DistributedScalarField(dm, backend; comm=comm)
    dphi.field.values .= T.values
    @test (@inferred pnorm(dphi)) isa Float64
    @test (@inferred pdot(dphi, dphi)) isa Float64
    @test (@inferred pmean(dphi)) isa Float64
end
