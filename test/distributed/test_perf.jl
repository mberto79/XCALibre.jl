# Phase 4 perf gate: hot-path allocation budgets + type stability; run per rank under mpiexec.
# Budgets are ~2x the values measured at introduction (recorded in dev/STATE.md); a blown
# budget = an allocation regression on the per-iteration path — fix it, don't raise the budget.
using XCALibre, PETSc, MPI, Test, KernelAbstractions
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

# NEW SECTION: Phase 5 incompressible hot paths (BFS; capture deqns via the unified setup hook)

include(joinpath(@__DIR__, "psimple_case.jl"))
import XCALibre.Solvers: setup_incompressible_solvers, correct_mass_flux!, max_courant_number!
import XCALibre.Solve: make_symmetric!, unwrap_eqn, sync!

gm2 = rank == 0 ? bfs_mesh() : nothing
dm2 = distribute(gm2; comm=comm)
model2, config2 = incompressible_case(dm2, bfs_bcs; iterations=3)
# capture the wrapped eqns the same way the solver body receives them
captured = setup_incompressible_solvers(
    (args...; kwargs...) -> args, model2, config2)
_, _, ∇p, U_deqn, p_deqn, config2 = captured
(; U, p, Uf, pf) = model2.momentum
bcs2 = config2.boundaries
U_eqn, p_eqn = unwrap_eqn(U_deqn), unwrap_eqn(p_deqn)
mdotf = XCALibre.ModelFramework.get_flux(U_eqn, 2)
xdir, ydir, zdir = XCALibre.ModelFramework.XDir(), XCALibre.ModelFramework.YDir(), XCALibre.ModelFramework.ZDir()

# prime state the way the unified SIMPLE body does (self-syncing seams)
sync!(U, dm2, config2); sync!(p, dm2, config2)
interpolate!(Uf, U, config2)
correct_boundaries!(Uf, U, bcs2.U, 0.0, config2)
XCALibre.Solvers.flux!(mdotf, Uf, config2)
grad!(∇p, pf, p, bcs2.p, 0.0, config2)

usolve!() = solve_equation!(U_deqn, U, bcs2.U, config2.solvers.U, xdir, ydir, zdir, config2)
psolve_eqn!() = solve_equation!(p_deqn, p, bcs2.p, config2.solvers.p, config2; ref=nothing)

usolve!(); usolve!(); psolve_eqn!(); psolve_eqn!() # warmup
make_symmetric!(p_eqn, config2); correct_mass_flux!(mdotf, p_eqn, config2; time=1)
cCo = KernelAbstractions.zeros(backend, Float64, length(dm2.cells))
max_courant_number!(cCo, model2, config2)

a_ueqn = @allocated usolve!()
a_peqn = @allocated psolve_eqn!()
a_sym = @allocated make_symmetric!(p_eqn, config2)
a_cmf = @allocated correct_mass_flux!(mdotf, p_eqn, config2; time=1)
a_halo3 = @allocated sync!(U, dm2, config2)
a_cour = @allocated max_courant_number!(cCo, model2, config2)

println("PERF5 rank=$rank ueqn=$a_ueqn peqn=$a_peqn sym=$a_sym cmf=$a_cmf " *
    "halo3=$a_halo3 courant=$a_cour")

@testset "Phase 5 perf (rank $rank)" begin
    # ~2x measured at introduction (see dev/STATE.md); halo/solve terms scale per neighbour
    @test a_sym <= 512
    @test a_cmf <= 12_288
    @test a_halo3 <= 4_096 + 4_096 * max(1, length(dm2.procs))
    @test a_cour <= 4_096
    @test a_ueqn <= 98_304 + 8_192 * max(1, length(dm2.procs))
    @test a_peqn <= 40_960 + 4_096 * max(1, length(dm2.procs))

    @test (@inferred solve_equation!(
        U_deqn, U, bcs2.U, config2.solvers.U, xdir, ydir, zdir, config2)) isa
        Tuple{Float64,Float64,Float64}
    @test (@inferred solve_equation!(
        p_deqn, p, bcs2.p, config2.solvers.p, config2; ref=nothing)) isa Float64
    @test (@inferred max_courant_number!(cCo, model2, config2)) isa Float64
end
