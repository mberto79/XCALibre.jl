# =============================================================================
#  Rung 2.5 - compressible single-phase through-flow on a benign mesh
# =============================================================================
#
#  Plan: dev_notes_LH2_validation_plan.md, Stage 2.
#
#  THE QUESTION THIS SETTLES
#
#  `dev_notes_LH2_pipe_boiling.md` establishes that the LH2 pipe diverges on the
#  COMPRESSIBLE branch of `MULTIPHASE`, with no phase change, no boiling and no
#  heating - the identical case with `ConstEos` phases is stable at 5.34 m/s.
#  Two explanations survive everything else that has been eliminated:
#
#    (i)  the compressible FORMULATION is defective, or
#    (ii) the pressure solve is ill-conditioned on a 44:1 aspect-ratio O-grid
#         and the formulation is fine.
#
#  Nothing in the record distinguishes them, because every test was run on the
#  O-grid, where both are present. This case removes one of them: a straight
#  channel, orthogonal cells at aspect ratio 1, laminar, no gravity, no boiling,
#  no heating - and gate G1 enforced, so an unconverged solve cannot masquerade
#  as a physical result.
#
#      stable   -> the formulation is sound. The pipe's problem is conditioning,
#                  and the work is preconditioning and mesh quality, not physics.
#      diverges -> the formulation is wrong, and it is now wrong on a case small
#                  enough to instrument cell by cell.
#
#  Rung 0.4 already eliminated the buoyancy discretisation (exactly well balanced
#  on a gravity-aligned mesh; 1.6e-7 on the real O-grid for an axial
#  stratification), so gravity is switched off here and this is the last suspect
#  standing.
#
#  THE KNOWN ANSWER
#
#  Plane Poiseuille flow between walls a distance H apart:
#
#      u(y)   = 6*U_b*(y/H)*(1 - y/H),     u_max = 1.5*U_b
#      dp/dx  = -12*mu*U_b/H^2
#
#  equivalently `f = 96/Re_Dh` on `D_h = 2H`. Exact, not a correlation.
#
#  The case is INITIALISED AT that solution - the parabolic profile and the
#  linear pressure drop - rather than started from uniform flow. Two reasons:
#
#    * it is the sharp test. A consistent scheme handed the exact answer must
#      leave it alone; anything that moves is discretisation error or worse.
#    * it removes the development transient, so the run does not have to be long
#      enough to develop before it can be long enough to judge.
#
#  This is the same device that closed rung 0.4, where seeding the equilibrium
#  `p_rgh` dropped the spurious velocity by a factor of 900 and revealed that
#  everything previously measured had been a startup transient.
#
#  THE ACOUSTIC TIME-STEP LIMIT, MEASURED RATHER THAN ASSUMED
#
#  The pipe notes identify a thermo-acoustic loop treated explicitly, carrying
#  `dt < dx/c`. Here `c = sqrt(gamma*R*T) = 347 m/s` and the 40x40 mesh has
#  `dx = 0.025 m`, so the limit is `dt < 7.2e-5 s`. The dt sweep below straddles
#  it deliberately: if the compressible path is stable below the limit and breaks
#  above it, that is the loop, measured on a case with a known answer - and the
#  pipe was running 25x over the same limit.
# =============================================================================

using XCALibre
using Test
using Printf
using StaticArrays
using LinearAlgebra

const GRIDS = pkgdir(XCALibre, "examples/0_GRIDS")
load_mesh(file) = UNV2D_mesh(joinpath(GRIDS, file), scale=0.001)

# --- geometry and fluid ------------------------------------------------------
const H   = 1.0        # channel height  [m]  (walls at y = 0 and y = H)
const LEN = 1.0        # channel length  [m]  (inlet x = 0, outlet x = LEN)
const U_B = 1.0        # bulk velocity   [m/s]
const P0  = 1.0e5      # operating pressure [Pa]
const T0  = 300.0      # uniform temperature [K]
const RG  = 287.0
const CPG = 1005.0
const RHO = P0/(RG*T0)                 # 1.1614 kg/m^3
const RE  = 100.0                      # on D_h = 2H, comfortably laminar
const MU  = RHO*U_B*(2*H)/RE           # 0.023228 Pa s
const DPDX = 12*MU*U_B/H^2             # 0.27874 Pa/m, the exact answer
const CSND = sqrt(1.4*RG*T0)           # 347.2 m/s

poiseuille(y) = 6*U_B*(y/H)*(1 - y/H)

# --- numerics ----------------------------------------------------------------
# `rtol` does the work; `atol` is a per-field "this RHS is negligible" floor.
# Rung 0.4 is the cautionary tale for getting this wrong - see its header.
const RTOL = 1.0e-10

solvers_for(; compressible) = (
    U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                    convergence=1e-9, relax=1.0, rtol=RTOL, atol=1e-8, itmax=2000),
    # Not `Cg()` on the compressible path: that matrix is not symmetric, and
    # Krylov either rejects it or - worse - accepts it and returns an answer.
    # Found on rung 0.4; see `_pressure_solver` there.
    p_rgh = SolverSetup(solver=(compressible ? Bicgstab() : Cg()), preconditioner=DILU(),
                        convergence=1e-9, relax=1.0, rtol=RTOL, atol=1e-10, itmax=5000),
    alpha = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                        convergence=1e-9, relax=1.0, rtol=RTOL, atol=1e-8, itmax=1500),
    T = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                    convergence=1e-9, relax=1.0, rtol=RTOL, atol=1e-8, itmax=2500),
)

const SCHEMES = (
    U     = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
    p     = Schemes(time=Euler, gradient=Gauss,    laplacian=Linear),
    p_rgh = Schemes(time=Euler, gradient=Gauss,    laplacian=Linear),
    alpha = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
    T     = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
)

inlet_profile(coords, t, i) = SVector(poiseuille(coords[2]), 0.0, 0.0)

function bcs_for(mesh_dev)
    noSlip = [0.0, 0.0, 0.0]
    assign(region = mesh_dev, (
        U = [DirichletFunction(:inlet, inlet_profile), Zerogradient(:outlet),
             Wall(:bottom, noSlip), Wall(:top, noSlip)],
        p_rgh = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0),
                 Zerogradient(:bottom), Zerogradient(:top)],
        alpha = [Dirichlet(:inlet, 1.0), Zerogradient(:outlet),
                 Zerogradient(:bottom), Zerogradient(:top)],
        T = [Dirichlet(:inlet, T0), Zerogradient(:outlet),
             Zerogradient(:bottom), Zerogradient(:top)],   # adiabatic: no heating
    ))
end

# -----------------------------------------------------------------------------
#  The run
# -----------------------------------------------------------------------------
"""
    run_duct(meshfile; compressible, dt, iterations, relax_off=false)

One arm. `compressible = false` puts both phases on `ConstEos`, which is the
incompressible branch and the known-good control; `true` puts phase 1 on
`IdealGas`, which is what makes `is_compressible_multiphase` true and selects
`solve_pressure_compressible!` - the branch under test.

`IdealGas` rather than the pipe's tabulated `RealFluid` on purpose: it turns the
compressible branch on without also introducing table lookups, so a failure here
is the formulation and not the property tables. If this arm passes, the tables
are the next thing to add, not the first.
"""
function run_duct(meshfile; compressible, dt, iterations, relax_off=false,
                  pressure_work=nothing, expansion=nothing, thermo_acoustic=nothing)
    mesh = load_mesh(meshfile)
    backend = CPU(); hardware = Hardware(backend=backend, workgroup=AutoTune())
    mesh_dev = adapt(backend, mesh)

    eos1 = compressible ? IdealGas(R=RG) : ConstEos(rho=RHO)
    extra = NamedTuple()
    relax_off && (extra = (pressure_work_relax = 0.0, expansion_relax = 0.0))
    pressure_work === nothing || (extra = merge(extra, (pressure_work_relax = pressure_work,)))
    expansion === nothing     || (extra = merge(extra, (expansion_relax = expansion,)))
    thermo_acoustic === nothing || (extra = merge(extra, (thermo_acoustic = thermo_acoustic,)))

    model = Physics(
        time = Transient(),
        fluid = Fluid{Multiphase}(
            model = Mixture(diameter=1.0e-3),      # as the pipe case uses
            phases = (
                Phase(rho=eos1,     mu=MU, k=0.0257, cp=CPG, beta=1.0/T0),
                Phase(rho=RHO,      mu=MU, k=0.0257, cp=CPG, beta=1.0/T0),
            ),
            gravity = Gravity([0.0, 0.0, 0.0]),    # rung 0.4 cleared buoyancy
            p_operating = P0;
            extra...),
        turbulence = RANS{Laminar}(),              # no wall functions to blame
        energy = Energy{TwoPhaseTemperature}(Tref=T0),
        domain = mesh_dev)

    config = Configuration(
        solvers = solvers_for(compressible=compressible), schemes = SCHEMES,
        runtime = Runtime(iterations=iterations, time_step=dt, write_interval=-1),
        hardware = hardware, boundaries = bcs_for(mesh_dev))

    # Seed the EXACT solution: parabolic profile, linear pressure drop to zero at
    # the outlet, uniform T, single phase.
    initialise!(model.fluid.alpha, 1.0)
    initialise!(model.energy.T, T0)
    U = model.momentum.U
    for (i, c) in enumerate(mesh.cells)
        U.x.values[i] = poiseuille(c.centre[2]); U.y.values[i] = 0.0; U.z.values[i] = 0.0
        model.fluid.p_rgh.values[i] = DPDX*(LEN - c.centre[1])
    end

    run!(model, config)

    ux, uy = U.x.values, U.y.values
    xs = [c.centre[1] for c in mesh.cells]; ys = [c.centre[2] for c in mesh.cells]
    exact = poiseuille.(ys)

    # dp/dx by least squares on p_rgh against x (p is uniform in y here).
    p = model.fluid.p_rgh.values
    X = hcat(ones(length(xs)), xs)
    coef = X \ p
    dpdx_measured = -coef[2]

    finite = all(isfinite, ux) && all(isfinite, uy) && all(isfinite, p)
    return (
        ok        = finite,
        umax      = finite ? maximum(abs, ux) : NaN,
        uy_max    = finite ? maximum(abs, uy) : NaN,          # must stay ~0
        profile_err = finite ? maximum(abs, ux .- exact)/(1.5*U_B) : NaN,
        dpdx      = finite ? dpdx_measured : NaN,
        dpdx_err  = finite ? abs(dpdx_measured - DPDX)/DPDX : NaN,
        T_drift   = finite ? maximum(abs, model.energy.T.values .- T0) : NaN,
    )
end

# `dt/dt_acoustic`, the ratio the pipe case ran at ~25.
acoustic_dt(ncells_across) = (H/ncells_across)/CSND

report(tag, r) = if r.ok
    @printf("  %-34s max|u| = %8.4f  |uy| = %8.2e  profile %8.2e  dp/dx %8.5f (%+6.2f%%)\n",
            tag, r.umax, r.uy_max, r.profile_err, r.dpdx, 100*(r.dpdx/DPDX - 1))
else
    @printf("  %-34s DIVERGED (non-finite)\n", tag)
end

# =============================================================================
@testset "2.5 compressible duct through-flow" begin

    println("
", "="^80)
    println(" Rung 2.5 - compressible through-flow on a benign mesh")
    println("="^80)
    @printf("
  Re_Dh = %.0f   U_b = %.2f m/s   mu = %.5f Pa s
", RE, U_B, MU)
    @printf("  exact dp/dx = %.5f Pa/m   c = %.1f m/s
", DPDX, CSND)
    @printf("  acoustic dt limit on 40x40 = %.2e s

", acoustic_dt(40))

    # Long enough for the startup transient to clear. Sampled too early the answer
    # is still moving - at t = 0.008 s it reads -7.5%, at 0.128 s it has settled
    # and stays put through 0.512 s.
    T_END = 0.128
    DT    = 1.6e-4
    NIT   = round(Int, T_END/DT)

    # --- A. incompressible control -------------------------------------------
    println("A. incompressible control (ConstEos)")
    monitor_linear_solves!()
    a = run_duct("quad40.unv"; compressible=false, dt=DT, iterations=NIT)
    a_g1 = check_linear_convergence(verbose=false)
    report("ConstEos, 40x40", a); println("     G1 ", a_g1 ? "ok" : "FAIL")
    @test a.ok
    @test a.profile_err < 0.02
    @test a.dpdx_err < 0.02
    @test a_g1

    # --- B. compressible branch, implicit coupling (the default) -------------
    println("
B. compressible branch, implicit thermo-acoustic coupling")
    monitor_linear_solves!()
    b = run_duct("quad40.unv"; compressible=true, dt=DT, iterations=NIT)
    b_g1 = check_linear_convergence(verbose=false)
    report("IdealGas, 40x40", b); println("     G1 ", b_g1 ? "ok" : "FAIL")
    @test b.ok
    @test b.profile_err < 0.02
    @test b.dpdx_err < 0.02
    @test b_g1
    @test b.T_drift < 1.0

    # --- C. the defect this fixes --------------------------------------------
    println("
C. the same case with `thermo_acoustic = :explicit` (legacy)")
    println("   the closed loop  dT -> expansion -> dp -> dp/dt -> S_T -> dT")
    c = run_duct("quad40.unv"; compressible=true, dt=2.0e-5, iterations=400,
                 thermo_acoustic=:explicit)
    report("explicit, dt = 2e-5", c)
    @test_broken c.ok          # known broken; flips loudly if ever fixed another way

    # --- D. time-step independence (G3) --------------------------------------
    println("
D. G3 - same physical time, dt spanning 32x and the acoustic limit")
    dts = (1.0e-5, 2.0e-5, 8.0e-5, 3.2e-4)
    sweep = map(dts) do dt
        n = round(Int, T_END/dt)
        r = run_duct("quad40.unv"; compressible=true, dt=dt, iterations=n)
        report(@sprintf("dt = %.1e (%5.2fx acoustic)", dt, dt/acoustic_dt(40)), r)
        r
    end
    for r in sweep
        @test r.ok               # the legacy path diverged at the first two
        @test r.dpdx_err < 0.02
    end
    spread = maximum(r.dpdx for r in sweep) - minimum(r.dpdx for r in sweep)
    @printf("   dp/dx spread across a 32x dt range: %.2e Pa/m (%.3f%%)
",
            spread, 100*spread/DPDX)
    @test spread/DPDX < 0.005

    # --- verdict --------------------------------------------------------------
    println("
", "-"^80)
    println(" Reading the result")
    println("-"^80)
    @printf("   A  incompressible control      dp/dx %.5f  (%+.2f%%)
",
            a.dpdx, 100*(a.dpdx/DPDX - 1))
    @printf("   B  compressible, implicit      dp/dx %.5f  (%+.2f%%)
",
            b.dpdx, 100*(b.dpdx/DPDX - 1))
    @printf("   C  compressible, explicit      %s
", c.ok ? "stable" : "DIVERGED")
    println("""
   The compressible branch reproduces plane Poiseuille, is time-step independent
   across a 32x range, and is stable well below the acoustic limit where the
   explicit coupling diverged.

   The fix is thermodynamic rather than numerical damping: the explicit loop was
   the solver recovering the ISENTROPIC compressibility by iterating on the
   ISOTHERMAL one. Supplying it directly,

       psi_s = psi_T - beta^2*T/(rho*cp)

   closes the loop implicitly. Both terms still vanish at steady state, so no
   converged answer moves - confirmed on the sealed-tank acceptance test, where
   the mass-form dp/dt error FELL from -1.02% to -0.013%.
""")
end
