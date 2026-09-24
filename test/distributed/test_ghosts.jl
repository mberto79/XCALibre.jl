# Ghost-consistency gate: after every self-syncing primitive of one SIMPLE iteration
# the ghost entries must equal their owners' values exactly; a nonzero names the primitive that
# lost its sync. Laminar BFS step by step, then two SST iterations through run!.
using XCALibre, PETSc, MPI, Test
import XCALibre.Solvers: setup_incompressible_solvers, correct_mass_flux!, inverse_diagonal!,
    remove_pressure_source!, H!, correct_velocity!, flux!, update_nueff!
import XCALibre.Solve: solve_equation!, explicit_relaxation!, unwrap_eqn, sync!
import XCALibre.Calculate: limit_gradient!, div!
import XCALibre.ModelFramework: get_flux, get_source, XDir, YDir, ZDir

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))
include(joinpath(@__DIR__, "sst_case.jl"))

# laminar case with a cell limiter on p so limit_gradient! is exercised
function limited_case(mesh; iterations)
    model, config = incompressible_case(mesh, bfs_bcs; iterations)
    schemes = (U=Schemes(divergence=Linear), p=Schemes(limiter=CellBased()))
    model, Configuration(solvers=config.solvers, schemes=schemes, runtime=config.runtime,
        hardware=config.hardware, boundaries=config.boundaries)
end

gmesh = rank == 0 ? bfs_mesh() : nothing
dm = distribute(gmesh; comm)
model, config = limited_case(dm; iterations=3)
captured = setup_incompressible_solvers((args...; kwargs...) -> args, model, config)
_, _, ∇p, U_deqn, p_deqn, config = captured
(; U, p, Uf, pf) = model.momentum
(; solvers, schemes, boundaries) = config
U_eqn, p_eqn = unwrap_eqn(U_deqn), unwrap_eqn(p_deqn)
mdotf = get_flux(U_eqn, 2)
nueff = get_flux(U_eqn, 3)
nu = model.fluid.nu
rDf = get_flux(p_eqn, 1)
divHv = get_source(p_eqn, 1)
rD = ScalarField(dm)
Hv = VectorField(dm)
prev = similar(p.values)
xdir, ydir, zdir = XDir(), YDir(), ZDir()
time = 1

mismatches = Pair{String,Float64}[]
chk(name, x) = push!(mismatches, name => check_ghosts(x, dm, config))

sync!(U, dm, config); sync!(p, dm, config)
interpolate!(Uf, U, config); correct_boundaries!(Uf, U, boundaries.U, 0.0, config)
flux!(mdotf, Uf, config)
grad!(∇p, pf, p, boundaries.p, 0.0, config); chk("grad! (setup)", ∇p.result)
limit_gradient!(schemes.p.limiter, ∇p, p, config); chk("limit_gradient! (setup)", ∇p.result)
update_nueff!(nueff, nu, model.turbulence, config)

for iteration ∈ 1:2
    solve_equation!(U_deqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config); chk("solve_system! U", U)
    inverse_diagonal!(rD, U_eqn, config); chk("inverse_diagonal!", rD)
    H!(Hv, U, U_eqn, config); chk("H!", Hv)
    inverse_diagonal!(rD, U_eqn, config; halo=false)
    remove_pressure_source!(U_eqn, ∇p, config)
    H!(Hv, U, U_eqn, config; halo=false)
    sync!((rD, Hv), dm, config); chk("sync! (rD, Hv) rD", rD); chk("sync! (rD, Hv) Hv", Hv)
    interpolate!(rDf, rD, config)
    interpolate!(Uf, Hv, config); correct_boundaries!(Uf, Hv, boundaries.U, time, config)
    flux!(mdotf, Uf, config); div!(divHv, mdotf, config)
    copyto!(prev, p.values)
    solve_equation!(p_deqn, p, boundaries.p, solvers.p, config; ref=nothing); chk("solve_system! p", p)
    correct_mass_flux!(mdotf, p_eqn, config; previous=prev, time=time)
    explicit_relaxation!(p, prev, solvers.p.relax, config); chk("explicit_relaxation!", p)
    grad!(∇p, pf, p, boundaries.p, time, config); chk("grad!", ∇p.result)
    limit_gradient!(schemes.p.limiter, ∇p, p, config); chk("limit_gradient!", ∇p.result)
    correct_velocity!(U, Hv, ∇p, rD, config); chk("correct_velocity!", U)
    update_nueff!(nueff, nu, model.turbulence, config)
end

# SST through run!: wall distance, k, omega and nut all cross the seam
gm2 = rank == 0 ? bfs_mesh() : nothing
dm2 = distribute(gm2; comm)
model2, config2 = sst_case(dm2, bfs_sst_bcs; iterations=2)
run!(model2, config2)
for (name, x) ∈ (("SST U", model2.momentum.U), ("SST p", model2.momentum.p),
        ("wall_distance! y", model2.turbulence.y), ("SST k", model2.turbulence.k),
        ("SST omega", model2.turbulence.omega), ("SST nut", model2.turbulence.nut))
    push!(mismatches, name => check_ghosts(x, dm2, config2))
end

rank == 0 && foreach(m -> println("GHOSTS n=$(MPI.Comm_size(comm)) ", m.first, " => ", m.second), mismatches)
@testset "ghost consistency (rank $rank)" begin
    for (name, d) ∈ mismatches
        @test (name, d) == (name, 0.0)
    end
end
MPI.Barrier(comm)
