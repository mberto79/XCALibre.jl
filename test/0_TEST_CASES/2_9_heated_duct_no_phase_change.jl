# =============================================================================
#  Rung 2.9 - heated duct, single phase, NO phase change
# =============================================================================
#
#  Plan: dev_notes_LH2_validation_plan.md, Stage 2.
#
#  WHY THIS RUNG EXISTS
#
#  Rung 2.5 fixed the thermo-acoustic loop and rungs 2.6-2.8 carried the fix onto
#  real equations of state - but every one of those cases was ADIABATIC. With no
#  heating, `dT/dt` is order 1e-6 K and the `expansion` source is doing almost
#  nothing, so the correction
#
#      psi_s = psi_T - beta^2*T/(rho*cp)
#
#  passed a test that barely exercised it. This is the first rung where the
#  thermal term is genuinely driven.
#
#  It is also the last rung before vapour appears, and it has to pass first: every
#  phase-change model is driven by `T - T_sat`, so a temperature field that is
#  wrong by 5% makes every boiling rate downstream wrong by more, in a way that
#  will look like a boiling-model problem and will not be one.
#
#  Deliberately NO phase change and NO boiling: the wall flux is set so the fluid
#  never approaches saturation. What is being tested is whether heat can be put
#  into the fluid correctly, not what happens when it boils.
#
#  TWO EXACT ANSWERS, both from Shah & London for laminar plane-channel flow
#
#  1. AXIAL GRADIENT. Energy conservation alone, no development required:
#
#         dT/dx = 2*q_w/(rho*cp*U_b*H)
#
#     with `2*q_w` because both walls are heated. This is a statement about
#     whether the energy equation conserves energy in the presence of a wall
#     flux - the property gate G2 in the plan - and it holds whatever the
#     cross-stream profile is doing.
#
#  2. NUSSELT NUMBER, fully developed, uniform `q_w` on both walls:
#
#         Nu = h*D_h/k = 8.235...    on  D_h = 2H
#
#     This one DOES require thermal development, and the development length for
#     laminar flow is `0.05*Re*Pr*D_h` - far longer than this channel. So the
#     temperature is SEEDED with the developed profile and imposed at the inlet
#     through a `DirichletFunction`, exactly as the velocity is. Same device as
#     rungs 0.4 and 2.5: hand the scheme the answer and see whether it stays.
#
#  THE DEVELOPED PROFILE, derived rather than looked up
#
#  Fully developed, `u*dT/dx = alpha*d2T/dy2` with `u = 6*U_b*(y/H)(1 - y/H)`.
#  Writing `T(x,y) = T_in + (dT/dx)*x + theta(y)` and integrating twice with
#  `theta'(H/2) = 0` by symmetry:
#
#      theta(y) = (q_w/k)*(2*y^3/H^2 - y^4/H^3 - y) + C
#
#  `C` is fixed by requiring the bulk mean of `theta` to vanish, so `theta` is
#  measured from the bulk temperature. Carrying that integral through gives
#  `theta(0) = 0.242857*q_w*H/k`, hence
#
#      Nu = q_w*2H/(k*theta(0)) = 2/0.242857 = 8.2353
#
#  which recovers the tabulated value and confirms the seed is the right profile
#  rather than merely a plausible one.
#
#  AN ARTIFICIAL PRANDTL NUMBER
#
#  `mu` is set for `Re = 100` (laminar, with a closed-form answer) and `k` is then
#  set for `Pr = 1`. Both are artificial; the exact solutions above hold for any
#  `mu` and `k`, and what is under test is the energy equation and the isentropic
#  correction, not air. `q_w` is chosen to give a few K of wall-to-bulk difference
#  so the state stays well inside the ideal-gas range.
# =============================================================================

using XCALibre
using Test
using Printf
using StaticArrays
using LinearAlgebra

const GRIDS = pkgdir(XCALibre, "examples/0_GRIDS")
load_mesh(file) = UNV2D_mesh(joinpath(GRIDS, file), scale=0.001)

# --- geometry, fluid, heating ------------------------------------------------
const H   = 1.0
const LEN = 1.0
const U_B = 1.0
const P0  = 1.0e5
const T_IN = 300.0
const RG  = 287.0
const CP  = 1005.0
const RHO = P0/(RG*T_IN)
const RE  = 100.0
const MU  = RHO*U_B*(2*H)/RE      # 0.023228 Pa s
const PR  = 1.0
const KTH = MU*CP/PR              # 23.34 W/m/K
const QW  = 288.0                 # W/m^2 on each wall -> ~3 K wall-to-bulk

const DTDX   = 2*QW/(RHO*CP*U_B*H)          # exact axial gradient [K/m]
const NU_EXACT = 8.235294117647058          # 2/0.242857..., derived in the header

poiseuille(y) = 6*U_B*(y/H)*(1 - y/H)

# theta(y) before the bulk-mean shift.
theta_raw(y) = (QW/KTH)*(2*y^3/H^2 - y^4/H^3 - y)

# Bulk mean of `theta_raw`, weighted by the velocity profile, by quadrature on a
# fine grid - the same integral the header does analytically, evaluated here so
# the seed cannot drift from the derivation.
const THETA_BULK = let n = 20001
    ys = range(0, H, length=n)
    u  = poiseuille.(ys)
    sum(u .* theta_raw.(ys))/sum(u)
end
theta(y) = theta_raw(y) - THETA_BULK

# The seeded/imposed temperature field scales with the wall flux, so the arm that
# turns heating OFF must seed a uniform field too - otherwise it simply advects
# the heated profile it was given and reports its gradient as if it were physics.
# (It did, on the first run: 0.474 K/m against an expected 0.)
const QW_ACTIVE = Ref(QW)
qw_scale() = QW_ACTIVE[]/QW

developed_T(x, y) = T_IN + qw_scale()*(DTDX*x + theta(y))

inlet_U(coords, t, i) = SVector(poiseuille(coords[2]), 0.0, 0.0)
inlet_T(coords, t, i) = developed_T(0.0, coords[2])

const RTOL = 1.0e-10
solvers_for(compressible) = (
    U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                    convergence=1e-9, relax=1.0, rtol=RTOL, atol=1e-8, itmax=2000),
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

"""
    run_heated(kind; dt, iterations, qw)

`:const` incompressible control, `:ideal` the compressible branch. `qw = 0`
reproduces rung 2.5's adiabatic case, which is the natural control for the
heated one.
"""
function run_heated(kind; dt=1.6e-4, iterations=800, qw=QW)
    QW_ACTIVE[] = qw
    mesh = load_mesh("quad40.unv")
    backend = CPU(); hardware = Hardware(backend=backend, workgroup=AutoTune())
    mesh_dev = adapt(backend, mesh)
    compressible = kind !== :const
    eos = compressible ? IdealGas(R=RG) : ConstEos(rho=RHO)
    noSlip = [0.0, 0.0, 0.0]

    model = Physics(
        time = Transient(),
        fluid = Fluid{Multiphase}(
            model = Mixture(diameter=1.0e-3),
            phases = (Phase(rho=eos,             mu=MU, k=KTH, cp=CP, beta=1.0/T_IN),
                      Phase(rho=ConstEos(rho=RHO), mu=MU, k=KTH, cp=CP, beta=1.0/T_IN)),
            gravity = Gravity([0.0, 0.0, 0.0]),
            p_operating = P0),
        turbulence = RANS{Laminar}(),
        energy = Energy{TwoPhaseTemperature}(Tref=T_IN),
        domain = mesh_dev)

    config = Configuration(
        solvers = solvers_for(compressible), schemes = SCHEMES,
        runtime = Runtime(iterations=iterations, time_step=dt, write_interval=-1),
        hardware = hardware,
        boundaries = assign(region = mesh_dev, (
            U = [DirichletFunction(:inlet, inlet_U), Zerogradient(:outlet),
                 Wall(:bottom, noSlip), Wall(:top, noSlip)],
            p_rgh = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0),
                     Zerogradient(:bottom), Zerogradient(:top)],
            alpha = [Dirichlet(:inlet, 1.0), Zerogradient(:outlet),
                     Zerogradient(:bottom), Zerogradient(:top)],
            T = [DirichletFunction(:inlet, inlet_T), Zerogradient(:outlet),
                 FixedHeatFlux(:bottom, qw), FixedHeatFlux(:top, qw)],
        )))

    initialise!(model.fluid.alpha, 1.0)
    U = model.momentum.U; T = model.energy.T
    for (i, c) in enumerate(mesh.cells)
        x, y = c.centre[1], c.centre[2]
        U.x.values[i] = poiseuille(y); U.y.values[i] = 0.0; U.z.values[i] = 0.0
        T.values[i] = developed_T(x, y)
        model.fluid.p_rgh.values[i] = (12*MU*U_B/H^2)*(LEN - x)
    end

    run!(model, config)

    Tv = T.values; ux = U.x.values
    xs = [c.centre[1] for c in mesh.cells]; ys = [c.centre[2] for c in mesh.cells]
    all(isfinite, Tv) && all(isfinite, ux) ||
        return (ok=false, dtdx=NaN, dtdx_err=NaN, Nu=NaN, Nu_err=NaN, umax=NaN)

    # Measured axial gradient: a linear fit in x. The cross-stream variation is
    # x-independent once developed, so it does not bias the slope.
    dtdx = (hcat(ones(length(xs)), xs) \ Tv)[2]

    # Nusselt at mid-duct. Bulk temperature is the velocity-weighted mean over the
    # column; the wall value is reconstructed from the near-wall cell and the
    # imposed flux, `T_w = T_cell + q_w*delta/k`.
    xmid = LEN/2
    dx = LEN/40
    col = findall(x -> abs(x - xmid) < 0.51*dx, xs)
    yc = ys[col]; Tc = Tv[col]; uc = ux[col]
    order = sortperm(yc); yc, Tc, uc = yc[order], Tc[order], uc[order]
    T_bulk = sum(uc .* Tc)/sum(uc)
    delta = yc[1]                                  # first cell centre from the wall
    T_wall = Tc[1] + qw*delta/KTH
    Nu = qw > 0 ? qw*(2*H)/(KTH*(T_wall - T_bulk)) : NaN

    return (ok = true, dtdx = dtdx, dtdx_err = abs(dtdx - DTDX)/DTDX,
            Nu = Nu, Nu_err = qw > 0 ? abs(Nu - NU_EXACT)/NU_EXACT : NaN,
            umax = maximum(abs, ux), T_max = maximum(Tv), ncol = length(col))
end

report(tag, r) = if r.ok
    @printf("  %-28s dT/dx %8.5f (%+6.2f%%)   Nu %7.4f (%+6.2f%%)   max|u| %6.4f\n",
            tag, r.dtdx, 100*(r.dtdx/DTDX - 1), r.Nu, 100*(r.Nu/NU_EXACT - 1), r.umax)
else
    @printf("  %-28s DIVERGED\n", tag)
end

@testset "2.9 heated duct, no phase change" begin

    println("\n", "="^80)
    println(" Rung 2.9 - heated duct: the first rung where dT/dt is genuinely driven")
    println("="^80)
    @printf("\n  Re = %.0f   Pr = %.1f   q_w = %.0f W/m^2 on each wall\n", RE, PR, QW)
    @printf("  exact dT/dx = %.5f K/m      exact Nu = %.4f\n", DTDX, NU_EXACT)
    @printf("  seeded wall-to-bulk difference = %.3f K\n\n", theta(0.0))

    # --- A. incompressible control -------------------------------------------
    println("A. incompressible control (ConstEos)")
    monitor_linear_solves!()
    a = run_heated(:const)
    a_g1 = check_linear_convergence(verbose=false)
    report("ConstEos, heated", a); println("     G1 ", a_g1 ? "ok" : "FAIL")
    @test a.ok
    @test a_g1
    @test a.dtdx_err < 0.02
    @test a.Nu_err < 0.05

    # --- B. compressible branch - the real test of the fix -------------------
    println("\nB. compressible branch (IdealGas) - dT/dt now genuinely non-zero")
    monitor_linear_solves!()
    b = run_heated(:ideal)
    b_g1 = check_linear_convergence(verbose=false)
    report("IdealGas, heated", b); println("     G1 ", b_g1 ? "ok" : "FAIL")
    @test b.ok
    @test b_g1
    @test b.dtdx_err < 0.02
    @test b.Nu_err < 0.05

    # --- C. the same case with the legacy explicit coupling ------------------
    println("\nC. compressible, heated, `thermo_acoustic = :explicit` would go here -")
    println("   rung 2.5 already shows it diverges ADIABATICALLY, so heating cannot")
    println("   rescue it. Not re-run; see 2_5_compressible_duct_throughflow.jl.")

    # --- D. G3 ---------------------------------------------------------------
    println("\nD. G3 - same physical time, dt over 16x, compressible + heated")
    g3 = map((2.0e-5, 8.0e-5, 3.2e-4)) do dt
        n = round(Int, 0.128/dt)
        r = run_heated(:ideal; dt=dt, iterations=n)
        report(@sprintf("dt = %.1e (%5d steps)", dt, n), r)
        r
    end
    for r in g3
        @test r.ok
        @test r.dtdx_err < 0.02
        @test r.Nu_err < 0.05
    end
    spread = maximum(r.Nu for r in g3) - minimum(r.Nu for r in g3)
    @printf("   Nu spread across a 16x dt range: %.3e (%.3f%%)\n", spread, 100*spread/NU_EXACT)
    @test spread/NU_EXACT < 0.01

    # --- E. heating off, as the control for the control ----------------------
    println("\nE. q_w = 0 - must reproduce rung 2.5's adiabatic result exactly")
    e = run_heated(:ideal; qw=0.0)
    @printf("  %-28s dT/dx %10.3e (exact 0)   max|u| %6.4f\n", "IdealGas, adiabatic",
            e.dtdx, e.umax)
    @test e.ok
    # Judged against the HEATED gradient, not against an absolute number: what
    # matters is that the adiabatic residual cannot contaminate the signal. It
    # comes out at 1.3e-5 K/m, i.e. 2.7e-5 of the heated 0.493 K/m, on a 300 K
    # field - converged to zero at the level the discretisation supports.
    @printf("       residual is %.1e of the heated gradient
", abs(e.dtdx)/DTDX)
    @test abs(e.dtdx) < 1e-3*DTDX

    println("\n", "-"^80)
    println(" Reading the result")
    println("-"^80)
    @printf("   A  control      dT/dx %+6.2f%%   Nu %+6.2f%%\n",
            100*(a.dtdx/DTDX - 1), 100*(a.Nu/NU_EXACT - 1))
    @printf("   B  compressible dT/dx %+6.2f%%   Nu %+6.2f%%\n",
            100*(b.dtdx/DTDX - 1), 100*(b.Nu/NU_EXACT - 1))
    println("""
   The energy equation conserves energy under a wall flux (dT/dx), and the
   cross-stream temperature profile is right (Nu), on both branches and
   independently of the time step.

   That is the precondition for everything in Stages 3 and 4: every phase-change
   model is driven by T - T_sat, so this is the field they will all be reading.
""")
end
