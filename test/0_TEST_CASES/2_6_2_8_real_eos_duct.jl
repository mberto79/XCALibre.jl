# =============================================================================
#  Rungs 2.6 and 2.8 - Peng-Robinson and tabulated real-fluid EoS on the duct
# =============================================================================
#
#  Plan: dev_notes_LH2_validation_plan.md, Stage 2.
#
#  Rung 2.5 established that the compressible through-flow path works on a benign
#  mesh once the thermo-acoustic loop is closed implicitly - but it did so with
#  `IdealGas`, deliberately, so that a failure would be the FORMULATION and not
#  the property tables. This adds the real equations of state, one at a time.
#
#      2.6  PengRobinson   - rho, psi and beta from ONE analytic cubic
#      2.8  RealFluid      - rho, psi and beta from THREE separate tables
#
#  Rung 2.7 gates 2.8 and must pass first: it verifies that the three tables are
#  mutually consistent derivatives, which is what the isentropic correction
#  `psi_s = psi_T - beta^2*T/(rho*cp)` assumes. They are, to ~2nd order.
#
#  THE ANSWER IS INDEPENDENT OF THE EQUATION OF STATE - which is the point
#
#      dp/dx = 12*mu*U_b/H^2
#
#  contains no density. At Re = 100 inertia is negligible, so every arm must
#  return the SAME number, and any spread between them is attributable to the EoS
#  path alone rather than to a different physical problem. That makes this a
#  controlled comparison rather than four separate validations.
#
#  A REAL LH2 STATE, AN ARTIFICIAL VISCOSITY
#
#  The state is genuine: 0.4 MPa and 24 K, subcooled liquid hydrogen, comfortably
#  inside the tabulated range and well clear of `T_sat = 26.0 K`, so both the
#  cubic and the tables are evaluated ON their liquid branch. That is what is
#  being added here, and it is evaluated where it is defined.
#
#  The viscosity is not: real LH2 at this state gives `Re = 1.1e7` on a 1 m
#  channel, which is not laminar and has no closed-form answer. `mu` is therefore
#  set to hold `Re = 100`. This is a verification case, not an LH2 flow - the
#  Poiseuille solution is exact for any `mu`, and it is `rho`, `psi` and `beta`
#  that are under test, all of which come from the EoS at the real state.
#
#  PENG-ROBINSON IS A NUMERICAL STEPPING STONE, NOT A VALIDATION EoS
#
#  It is here because it is smooth, analytic and internally consistent - `rho`,
#  `psi` and `beta` all come from the same cubic, so it is the easy case for the
#  isentropic correction and a clean reference for what "consistent" looks like.
#
#  It is NOT suitable for quantitative hydrogen work, and that is not a caveat
#  invented here: `2_peng_robinson.jl` says so in its own header. H2 has a
#  NEGATIVE acentric factor (omega = -0.219), outside the range the alpha-function
#  correlation was fitted over, and liquid densities are wrong by tens of percent.
#  The check below prints that error rather than hiding it.
# =============================================================================

using XCALibre
using Test
using Printf
using StaticArrays
using LinearAlgebra

const GRIDS = pkgdir(XCALibre, "examples/0_GRIDS")
load_mesh(file) = UNV2D_mesh(joinpath(GRIDS, file), scale=0.001)

# --- state and geometry ------------------------------------------------------
const H   = 1.0          # channel height [m]
const LEN = 1.0
const U_B = 1.0          # bulk velocity [m/s]
const P0  = 0.4e6        # 0.4 MPa, the lowest of the Tatsumoto pressures
const T0  = 24.0         # subcooled liquid: T_sat(0.4 MPa) = 26.0 K
const RE  = 100.0

# Tabulated range, matching rung 2.7. The liquid branch stops below T_c = 33.145.
const P_TAB = (0.30e6, 1.25e6)
const T_TAB = (20.0, 32.5)

# Real LH2 transport and thermal properties at this state, except `mu` - see the
# header. `RHO_REF` only sets `mu` through the Reynolds number; each arm's actual
# density comes from its own EoS.
const RHO_REF = 70.0
const MU  = RHO_REF*U_B*(2*H)/RE
const CP  = 9800.0       # J/kg/K, LH2 near 0.4 MPa
const KTH = 0.100        # W/m/K
const DPDX = 12*MU*U_B/H^2      # the exact answer, EoS-independent

poiseuille(y) = 6*U_B*(y/H)*(1 - y/H)
inlet_profile(coords, t, i) = SVector(poiseuille(coords[2]), 0.0, 0.0)

const RTOL = 1.0e-10
solvers_for(compressible) = (
    U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                    convergence=1e-9, relax=1.0, rtol=RTOL, atol=1e-6, itmax=2000),
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

# Built once: walking the Helmholtz EOS over a 161x161 grid is not free.
const LH2_TABLE = RealFluid(H2(), :liquid; p=P_TAB, T=T_TAB, np=161, nT=161, verbose=false)
const PR_LIQUID = PengRobinson(H2(), branch=:liquid)

"""
    eos_for(kind) -> (rho_model, beta_model, label)

`:const` incompressible control, `:pr` Peng-Robinson, `:table` tabulated
Helmholtz. Each supplies its own `beta`, because the isentropic correction needs
`beta` and `psi` to be derivatives of the SAME density - pairing one model's rho
with another's beta is exactly the inconsistency rung 2.7 exists to detect.
"""
function eos_for(kind)
    if kind === :const
        return ConstEos(rho=RHO_REF), 0.0164, "ConstEos"
    elseif kind === :pr
        return PR_LIQUID, PengRobinsonBeta(PR_LIQUID), "PengRobinson"
    elseif kind === :table
        return LH2_TABLE.rho, LH2_TABLE.beta, "RealFluid"
    else
        error("unknown eos kind: $kind")
    end
end

function run_duct(kind; meshfile="quad40.unv", dt=1.6e-4, iterations=800)
    mesh = load_mesh(meshfile)
    backend = CPU(); hardware = Hardware(backend=backend, workgroup=AutoTune())
    mesh_dev = adapt(backend, mesh)
    rho_model, beta_model, _ = eos_for(kind)
    compressible = kind !== :const

    noSlip = [0.0, 0.0, 0.0]
    model = Physics(
        time = Transient(),
        fluid = Fluid{Multiphase}(
            model = Mixture(diameter=1.0e-3),
            phases = (
                Phase(rho=rho_model,          mu=MU, k=KTH, cp=CP, beta=beta_model),
                # Inactive at alpha = 1, but every phase is evaluated in every
                # cell, so it still has to be well defined.
                Phase(rho=ConstEos(rho=RHO_REF), mu=MU, k=KTH, cp=CP, beta=0.0164),
            ),
            gravity = Gravity([0.0, 0.0, 0.0]),
            p_operating = P0),
        turbulence = RANS{Laminar}(),
        energy = Energy{TwoPhaseTemperature}(Tref=T0),
        domain = mesh_dev)

    config = Configuration(
        solvers = solvers_for(compressible), schemes = SCHEMES,
        runtime = Runtime(iterations=iterations, time_step=dt, write_interval=-1),
        hardware = hardware,
        boundaries = assign(region = mesh_dev, (
            U = [DirichletFunction(:inlet, inlet_profile), Zerogradient(:outlet),
                 Wall(:bottom, noSlip), Wall(:top, noSlip)],
            p_rgh = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0),
                     Zerogradient(:bottom), Zerogradient(:top)],
            alpha = [Dirichlet(:inlet, 1.0), Zerogradient(:outlet),
                     Zerogradient(:bottom), Zerogradient(:top)],
            T = [Dirichlet(:inlet, T0), Zerogradient(:outlet),
                 Zerogradient(:bottom), Zerogradient(:top)],
        )))

    # Seeded at the exact solution, as rung 2.5 established: a consistent scheme
    # handed the answer must leave it alone.
    initialise!(model.fluid.alpha, 1.0)
    initialise!(model.energy.T, T0)
    U = model.momentum.U
    for (i, c) in enumerate(mesh.cells)
        U.x.values[i] = poiseuille(c.centre[2]); U.y.values[i] = 0.0; U.z.values[i] = 0.0
        model.fluid.p_rgh.values[i] = DPDX*(LEN - c.centre[1])
    end

    run!(model, config)

    ux = U.x.values; p = model.fluid.p_rgh.values
    xs = [c.centre[1] for c in mesh.cells]; ys = [c.centre[2] for c in mesh.cells]
    finite = all(isfinite, ux) && all(isfinite, p)
    finite || return (ok=false, umax=NaN, profile_err=NaN, dpdx=NaN, dpdx_err=NaN,
                      T_drift=NaN, rho=NaN)
    coef = hcat(ones(length(xs)), xs) \ p
    dpdx = -coef[2]
    return (ok = true,
            umax = maximum(abs, ux),
            profile_err = maximum(abs, ux .- poiseuille.(ys))/(1.5*U_B),
            dpdx = dpdx, dpdx_err = abs(dpdx - DPDX)/DPDX,
            T_drift = maximum(abs, model.energy.T.values .- T0),
            rho = model.fluid.rho.values[1])
end

@testset "2.6 / 2.8 real equations of state on the duct" begin

    println("\n", "="^80)
    println(" Rungs 2.6 / 2.8 - PengRobinson and RealFluid on the compressible duct")
    println("="^80)
    @printf("\n  state: %.2f MPa, %.1f K (subcooled liquid H2, T_sat = 26.0 K)\n", P0/1e6, T0)
    @printf("  Re_Dh = %.0f (artificial mu = %.4f Pa s - see header)\n", RE, MU)
    @printf("  exact dp/dx = %.5f Pa/m, independent of the EoS\n\n", DPDX)

    # What each EoS thinks the density is at this state. The Helmholtz value is
    # the reference; PR's error here is the documented H2 weakness, printed
    # rather than hidden.
    rho_tab = table_lookup(LH2_TABLE.rho.rho, P0, T0)
    rho_pr  = XCALibre.ModelPhysics.pr_density(PR_LIQUID, P0, T0)
    @printf("  density at this state:  Helmholtz %.3f   Peng-Robinson %.3f  (%+.1f%%)\n\n",
            rho_tab, rho_pr, 100*(rho_pr/rho_tab - 1))

    results = map((:const, :pr, :table)) do kind
        _, _, label = eos_for(kind)
        monitor_linear_solves!()
        r = run_duct(kind)
        g1 = check_linear_convergence(verbose=false)
        if r.ok
            @printf("  %-14s max|u| = %7.4f  profile %8.2e  dp/dx %9.5f (%+6.2f%%)  dT %7.1e  %s\n",
                    label, r.umax, r.profile_err, r.dpdx, 100*(r.dpdx/DPDX - 1),
                    r.T_drift, g1 ? "G1 ok" : "G1 FAIL")
        else
            @printf("  %-14s DIVERGED\n", label)
        end
        (kind = kind, label = label, g1 = g1, r...)
    end

    for r in results
        @test r.ok
        @test r.g1
        @test r.profile_err < 0.02
        @test r.dpdx_err < 0.02
        @test r.T_drift < 0.5
    end

    # The controlled comparison: dp/dx contains no density, so every arm must
    # agree. Any spread is the EoS path, not a different physical problem.
    spread = maximum(r.dpdx for r in results) - minimum(r.dpdx for r in results)
    @printf("\n  dp/dx spread across the three EoS: %.3e Pa/m (%.3f%% of exact)\n",
            spread, 100*spread/DPDX)
    @test spread/DPDX < 0.01

    # --- time-step independence on the hardest arm ---------------------------
    println("\n  G3 on the tabulated EoS - same physical time, dt over 16x:")
    g3 = map((2.0e-5, 8.0e-5, 3.2e-4)) do dt
        n = round(Int, 0.128/dt)
        r = run_duct(:table; dt=dt, iterations=n)
        r.ok ? @printf("    dt = %.1e (%5d steps)  dp/dx %9.5f (%+6.2f%%)\n",
                       dt, n, r.dpdx, 100*(r.dpdx/DPDX - 1)) :
               @printf("    dt = %.1e  DIVERGED\n", dt)
        r
    end
    for r in g3
        @test r.ok
        @test r.dpdx_err < 0.02
    end
    g3_spread = maximum(r.dpdx for r in g3) - minimum(r.dpdx for r in g3)
    @printf("    spread: %.3e Pa/m (%.3f%%)\n", g3_spread, 100*g3_spread/DPDX)
    @test g3_spread/DPDX < 0.01

    println("""

   Both real equations of state reproduce plane Poiseuille on the compressible
   path, agree with the incompressible control and with each other, and are
   time-step independent. The isentropic correction therefore survives contact
   with a real psi and beta - including a tabulated pair that are only consistent
   to interpolation order, which rung 2.7 measures directly.
""")
end
