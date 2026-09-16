# =============================================================================
#  NASA K-Site liquid hydrogen tank - self-pressurisation / boil-off
# =============================================================================
#
#  Reproduces cases K(i) and K(ii) of:
#     Fernandes, Korsukova, Ellis, Ambrose & Eastwick,
#     "A CFD comparison of interfacial phase change models for boil-off,
#      self-pressurisation and thermal stratification in liquid hydrogen
#      storage tanks", Int. J. Heat Mass Transfer 256 (2026) 128067.
#     https://doi.org/10.1016/j.ijheatmasstransfer.2025.128067
#
#  Experimental reference: Hasan, Lin & Van Dresar (1991); Van Dresar, Lin &
#  Hasan (1992).
#
# -----------------------------------------------------------------------------
#  Status
# -----------------------------------------------------------------------------
#  Runs end to end: the two-phase energy equation, `FixedHeatFlux`, the
#  compressible ideal-gas ullage and the phase change models are all in place.
#  Known deviations from the paper, each marked where it occurs below:
#    - isothermal start instead of the measured initial profile   # TO OBTAIN
#    - heat flux applied to the fluid, not through a conjugate wall # SIMPLIFICATION
#    - vapour transport properties (mu, k, cp) held constant, where the paper
#      uses temperature-dependent NIST values for both phases       # SIMPLIFICATION
#    - 17.5 hr at dt = 0.01 s is ~6.3M steps, impractical with explicit MULES
#  and no digitised experimental p(t) trace to compute the paper's MAPE against.
#
#  Mass conservation: `XCALIBRE_VOF_AUDIT=1` prints the phase-change mass budget
#  every `XCALIBRE_VOF_AUDIT_EVERY` steps (default 50). Measured over the first
#  10 s with Schrage, the tabulated liquid and dt = 0.01 s:
#    - a one-off +1.2e-5 kg of liquid in the first second (~5e-6 of the liquid
#      mass). The liquid density is seeded at a uniform p_operating and, being
#      compressible, then densifies under the ~700 Pa hydrostatic head.
#    - thereafter the liquid drifts by roughly -1e-7 to -2.5e-7 kg/s once the
#      interface condenses, comparable to the net transfer. Not yet attributed;
#      the likely candidate is the interface part of the liquid's thermal
#      expansion, which the alpha update does not cancel (see
#      `apply_phase_change_alpha!`). For comparison, the constant-density
#      liquid this replaced was off by +2.1e-7 kg over 10 s with beta = 0, and
#      by +6.2e-7 kg/s with beta = 0.0164.
#
#  KNOWN ISSUE: the vapour gains ~1.2e-7 kg/s beyond the transfer on this wedge
#  with the default `thermo_acoustic = :implicit` (2-3x less with `:explicit`);
#  cause not yet identified.
# =============================================================================

using XCALibre
using Test

# -----------------------------------------------------------------------------
# Case selection  (paper Table 1 and Table 3)
# -----------------------------------------------------------------------------
#   Case    Fill    Heat     q_w         P0        Duration
#   K(i)    50%     49.0 W   3.50 W/m^2  103 kPa   17.5 hr
#   K(ii)   50%     28.0 W   2.00 W/m^2  103 kPa   20.0 hr
#
CASE = :Ki

WALL_HEAT_FLUX, DURATION = CASE === :Ki ? (3.50, 17.5*3600) : (2.00, 20.0*3600)
FILL_LEVEL  = 0.50          # both K-Site cases; 50% => flat interface at z = 0
p_operating = 103.0e3       # [Pa] initial tank pressure (paper Table 3)

# -----------------------------------------------------------------------------
# Phase change model  (paper Sec. 3.4, Table 4 - baseline coefficients marked *)
# -----------------------------------------------------------------------------
#   Schrage   sigma = 1e-3*, 1e-4, 1e-5     accommodation coefficient [-]
#   MeJ       h     = 1.0*,  10.0, 100.0    liquid-vapour HTC [W/m^2/K]
#   Lee       sigma = 1e-6*, 1e-7, 1e-8     accommodation coefficient [-]
#
# Paper's finding: Schrage is most accurate and near-insensitive to sigma
# (max 3.0% MAPE); MeJ matches it when h is tuned (~10 W/m^2/K for this tank);
# Lee is worst (up to 11% MAPE) and diverges at sigma = 1e-6.
#
PHASE_CHANGE_MODEL = :schrage

phase_change = if PHASE_CHANGE_MODEL === :schrage
    # Paper Eq. (14), near-equilibrium form:
    #   mdot" = (2s/(2-s)) * sqrt(M/(2 pi R T_sat)) * (p_sat - p_v)
    Schrage(sigma=1.0e-3)
elseif PHASE_CHANGE_MODEL === :mej
    # Paper Eq. (9):  mdot" = h (T - T_sat) / L
    ModifiedEnergyJump(h=1.0)
elseif PHASE_CHANGE_MODEL === :lee
    # NOT the paper's parametrisation. The paper (Eqs. 11-12) derives its Lee
    # coefficient from an accommodation coefficient,
    #   beta = sigma * sqrt(M/(2 pi R T_sat)) * L rho_l/(rho_l - rho_v),
    # so the table's sigma = 1e-6 / 1e-7 / 1e-8 are not values of `r`.
    # XCALibre's `Lee` takes `r` directly as a volumetric relaxation rate [1/s];
    # `r = 100` is a numerical choice, not a value from the paper.
    #
    # KNOWN ISSUE: Lee does not conserve mass on the VOF path - drift 3.8e-3 over
    # 50 steps in `test/unit_test_phase_change.jl`, against 4.5e-6 for Schrage -
    # and the cause is not identified. Check the mass audit before trusting it.
    Lee(r=100.0)
else
    error("Unknown PHASE_CHANGE_MODEL: $PHASE_CHANGE_MODEL")
end

# -----------------------------------------------------------------------------
# Mesh
# -----------------------------------------------------------------------------
# Ellipsoid, major diameter 2.20 m, minor diameter 1.93 m (paper Sec. 2), giving
# V = 4.891 m^3 and S = 13.978 m^2 - both matching the paper's 4.89 m^3 and
# 13.98 m^2. Surface area matters because the case is specified by heat FLUX:
# 3.5 W/m^2 x 13.978 m^2 = 48.9 W against the paper's 49.0 W.
#
# The paper uses a 2D axisymmetric domain. XCALibre has no axisymmetric
# treatment, so this is a 5 degree wedge with Symmetry on the wedge planes -
# an equivalent discretisation.
#
# Generate with:
#     cd examples/0_GRIDS/ksite_wedge && julia make_ksite_wedge.jl
#     ./run_blockMesh.sh
# then validate with test/unit_test_ksite_wedge_mesh.jl

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "ksite_wedge", "constant", "polyMesh")
mesh = FOAM3D_mesh(mesh_file, scale=1.0, integer_type=Int64, float_type=Float64)

backend = CPU(); workgroup = AutoTune(); activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

a_semi = 1.100      # equatorial semi-axis [m]
b_semi = 0.965      # polar semi-axis [m]
z_fill = 0.0        # 50% fill of an ellipsoid => interface at the equator

# -----------------------------------------------------------------------------
# Operating point
# -----------------------------------------------------------------------------
# Saturation temperature at 103 kPa from the paper's Antoine fit, Eq. (15):
#     log10(p_sat [bar]) = A - B/(T_sat + C),  A=3.54314, B=99.395, C=7.726
# valid 21.01-32.27 K. At 103 kPa this gives T_sat ~ 20.43 K.
T_sat_0 = 20.43

# The paper applies the measured initial temperature profile (varying with
# height only) from Van Dresar et al. for the K-Site cases, rather than an
# isothermal start - Hasan notes this materially changes the early
# pressurisation rate. That profile is not in the repo; an isothermal start at
# T_sat is used here as a stand-in and is a known deviation.  # TO OBTAIN
T_liquid_0 = T_sat_0

noSlipVelocity = [0.0, 0.0, 0.0]
gravity = Gravity([0.0, 0.0, -9.81])   # tank axis is z

# -----------------------------------------------------------------------------
# Physics
# -----------------------------------------------------------------------------
# Phase 1 is the tracked phase (alpha = 1 -> liquid hydrogen).
#
# Paper Sec. 3.2: properties are temperature dependent from NIST, with the
# vapour treated as an ideal gas; flow is laminar ("laminar models were found to
# reproduce pressure rise and vapour stratification more accurately than
# commonly used RANS closures"); and surface tension is NEGLECTED, being
# several orders of magnitude smaller than the other terms.

# Liquid: tabulated from the Helmholtz EOS, so density, expansivity, cp, mu and k
# all follow the local (p, T). This is what gives the liquid BUOYANCY - the
# wall-heated boundary layer is lighter and rises towards the interface, the
# mechanism behind thermal stratification - and it lets the liquid expand
# without creating mass, because the density drop and the pressure equation's
# volume source now describe the same expansion. (With the previous
# constant-density liquid, `beta` had to be zero for exactly that reason.)
#
# Normal hydrogen, to be consistent with the saturation curve: the paper's
# Antoine fit (the default `saturation`) gives T_sat = 20.43 K at 103 kPa, which
# is normal hydrogen's value; parahydrogen's is 20.33 K (NIST).
#
# Range: p covers K(i)'s rise from 103 to ~207 kPa plus the ~700 Pa hydrostatic
# head. T_max must stay within 5 K of T_sat at p_min, or the liquid branch falls
# back to the saturation line part-way up a column; 25 K is inside that at 1 bar
# (T_sat ~ 20.3 K). Hotter cells - the ullage - read the boundary value, which is
# weighted by alpha ~ 0 there.
lh2 = RealFluid(H2(), :liquid; p=(1.0e5, 2.5e5), T=(19.0, 25.0), np=61, nT=61)

model = Physics(
    time = Transient(),
    fluid = Fluid{Multiphase}(
        model = VOF(cAlpha=1.0, sigma=0.0),   # surface tension neglected (Sec 3.2)

        phases = (
            # --- liquid hydrogen (alpha = 1) --------------------------------
            # Tabulated real fluid - see `lh2` above.
            Phase(lh2),
            # --- hydrogen vapour --------------------------------------------
            # Must be compressible for the tank to self-pressurise. The paper
            # (Sec. 3.2) treats the vapour as an ideal gas, so `IdealGas` rather
            # than the full Helmholtz EOS is the reference behaviour here.
            Phase(
                rho  = IdealGas(M=2.01588e-3),   # H2 molar mass -> R = 4124.2 J/kg/K
                mu   = 1.11e-6,     # [Pa s]
                k    = 0.0169,      # [W/m/K]
                cp   = 12200.0,     # [J/kg/K]
            ),
        ),

        phase_change = phase_change,
        h_fg = 446.0e3,             # [J/kg] latent heat (from EOS when available)
        p_operating = p_operating,

        # Local-density `p_rgh` split (the interFoam form), chosen EXPLICITLY.
        # This is a sharp VOF interface at a ~58:1 density ratio, where grad(rho)
        # is a clean jump that the face-flux buoyancy term balances; a reference
        # density would instead impose a body force (rho - rho_ref)*g that jumps
        # by ~670 N/m^3 across the interface. It is also the split this case has
        # always run with. Setting it explicitly also silences the "rho_ref is
        # unset" warning - see `multiphase_rho_ref`.
        rho_ref = nothing,

        gravity = gravity
    ),
    turbulence = RANS{Laminar}(),          # paper Sec. 3.2
    energy = Energy{TwoPhaseTemperature}(Tref=T_sat_0),
    domain = mesh_dev
)

# -----------------------------------------------------------------------------
# Boundary conditions
# -----------------------------------------------------------------------------
# Sealed tank: every boundary is a wall plus the two wedge symmetry planes and
# the degenerate axis faces (zero area, `Empty`).
#
# NOTE: the paper models conjugate heat transfer through the 2.08 mm Al 2219
# wall (6 cells, Al 5083 properties from NIST as a stand-in), applying the heat
# flux at the OUTER solid surface. That matters: Fig. 6(c) shows wall conduction
# redistributes heat towards the interface, so the flux into the fluid is
# markedly non-uniform even though the external flux is uniform. Applying the
# flux directly to the fluid, as below, is a simplification - XCALibre has a
# `Conduction` solid model but no fluid/solid coupling.        # SIMPLIFICATION

BCs = assign(
    region = mesh_dev,
    (
        U = [
            Wall(:tankWall, noSlipVelocity),
            Symmetry(:wedgeFront),
            Symmetry(:wedgeBack),
            Empty(:axis),
        ],
        p_rgh = [
            Zerogradient(:tankWall),
            Symmetry(:wedgeFront),
            Symmetry(:wedgeBack),
            Empty(:axis),
        ],
        alpha = [
            Zerogradient(:tankWall),
            Symmetry(:wedgeFront),
            Symmetry(:wedgeBack),
            Empty(:axis),
        ],
        T = [
            FixedHeatFlux(:tankWall, WALL_HEAT_FLUX),
            Symmetry(:wedgeFront),
            Symmetry(:wedgeBack),
            Empty(:axis),
        ],
    )
)

# -----------------------------------------------------------------------------
# Numerics  (paper Sec. 3.5: dt = 0.01 s, 5 inner iterations, 2nd order)
# -----------------------------------------------------------------------------
schemes = (
    U     = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
    p     = Schemes(time=Euler, gradient=Gauss,    laplacian=Linear),
    p_rgh = Schemes(time=Euler, gradient=Gauss,    laplacian=Linear),
    alpha = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
    T     = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
)

solvers = (
    U = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(),
        convergence=1e-7, relax=1.0, rtol=0.0, atol=1.0e-6),
    p_rgh = SolverSetup(
        solver = AMG(mode = Bicgstab(),
        smoother = AMGGaussSeidel(sweep = AMGForwardSweep())), 
        preconditioner=Jacobi(),
        convergence=1e-7, relax=1.0, rtol=0.0, atol=1.0e-8),
    alpha = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(),
        convergence=1e-7, relax=1.0, rtol=0.0, atol=1.0e-6),
    T = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(),
        convergence=1e-7, relax=1.0, rtol=0.0, atol=1.0e-6),
)

# 17.5 hr at dt = 0.01 s is 6.3 million steps. The paper used STAR-CCM+ with a
# 2nd-order implicit scheme and 5 inner iterations; XCALibre's alpha transport
# is explicit MULES and Courant limited, so this is the run-length problem
# flagged in the implementation plan.
dt = 0.02
runtime = Runtime(
    iterations = round(Int, DURATION/dt),
    time_step  = dt,
    write_interval = round(Int, 600/dt)
)

config = Configuration(
    solvers=solvers, schemes=schemes,
    runtime=runtime, hardware=hardware, boundaries=BCs)

# -----------------------------------------------------------------------------
# Initialisation
# -----------------------------------------------------------------------------
initialise!(model.momentum.U, noSlipVelocity)
initialise!(model.fluid.p_rgh, 0.0)
initialise!(model.energy.T, T_liquid_0)
initialise!(model.fluid.alpha, 0.0)                       # vapour everywhere
setField_Box!(
    mesh = mesh,
    field = model.fluid.alpha,
    value = 1.0,
    min_corner = [-a_semi, -a_semi, -b_semi],
    max_corner = [ a_semi,  a_semi,  z_fill+1e-6])

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

ENV["XCALIBRE_UFLUX_BLEND"] = "0.05"
ENV["XCALIBRE_GHF_MIDPOINT"] = "1"
residuals = run!(model, config, inner_loops=1)

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
# The paper's metric is MAPE on the pressurisation curve (Eq. 17):
#     Schrage <= 3.0%, MeJ comparable when tuned, Lee up to 11%.
# Reproducing that needs the digitised experimental p(t) trace (paper Fig. 7),
# which is not in the repo.                                    # TO OBTAIN
#
# K(i) rises from 103 kPa to roughly 207 kPa over 17.5 hr (Fig. 7a);
# K(ii) from 103 kPa to roughly 165 kPa over 20 hr (Fig. 7d).

p_ullage = ullage_average(model.momentum.p, model.fluid.alpha, threshold=0.5)

@test p_ullage > p_operating                          # must self-pressurise
@test all(0.0 .<= model.fluid.alpha.values .<= 1.0)   # alpha stays bounded
@test minimum(model.energy.T.values) > 19.0           # no unphysical undershoot
