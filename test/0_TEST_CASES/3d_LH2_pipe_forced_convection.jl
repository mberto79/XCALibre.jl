# =============================================================================
#  Forced convection nucleate boiling of saturated LH2 in a vertical heated pipe
# =============================================================================
#
#  Reproduces the steady-state nucleate boiling regime of:
#
#      Tatsumoto, Shirai, Shiotsu, Hata, Naruo, Kobayasi & Inatani,
#      "Forced convection heat transfer of saturated liquid hydrogen in
#       vertically-mounted heated pipes",
#      AIP Conf. Proc. 1573, 44-51 (2014).  doi:10.1063/1.4860681
#
#  Physics exercised, and why each piece is needed:
#
#    * REAL equation of state (`RealFluid`, Helmholtz H2). The experiment runs at
#      0.4-1.1 MPa against a critical pressure of 1.2964 MPa, so the vapour is at
#      up to 85% of p_crit. Ideal gas is not defensible there: at 1.1 MPa the
#      real vapour is ~4.4x more compressible than p/(RT) would suggest, and it
#      is exactly that compressibility that the pressure equation's psi term
#      carries. The liquid cp also roughly doubles across the pressure range, so
#      the properties are tabulated variable, not constant.
#
#    * `Mixture` multiphase model - the drift-flux formulation, appropriate for
#      the dispersed bubbly flow of subcooled/saturated nucleate boiling.
#
#    * `Lee` bulk phase change on the liquid/vapour interface.
#
#    * `RPI` wall nucleate boiling with `LemmertChawla` site density, which is
#      what actually generates vapour at the heated wall. Without it the Lee
#      model alone has no interface to act on in an initially all-liquid pipe.
#
#    * `RANS{KOmegaSST}` with wall functions on a y+ = 30-50 mesh.
#
# -----------------------------------------------------------------------------
#  !!! THIS CASE DOES NOT REACH A USABLE SOLUTION YET !!!
# -----------------------------------------------------------------------------
#  Everything constructs and the solver runs, but the compressible pressure path
#  goes unstable within ~100 steps at the time steps this case needs. The cause
#  has been isolated and is NOT in the boiling models - see below - so the case
#  is left faithful to the experiment rather than tuned into apparent stability.
#
#  WHAT WAS MEASURED (150 steps, this mesh, q_w = 3e4 W/m^2)
#
#    both phases RealFluid   + no phase change at all : max|U| = 5.2e5 m/s  DIVERGED
#    both phases RealFluid   + Lee only               : NaN                DIVERGED
#    both phases RealFluid   + RPI only               : NaN                DIVERGED
#    both phases ConstEos    + no phase change, q=0   : max|U| = 5.34 m/s  STABLE
#    both phases ConstEos    + no phase change, q=3e4 : max|U| = 5.34 m/s  STABLE
#
#  The last two are the same mesh, same turbulence model, same boundary
#  conditions and the same inlet velocity of 5.33 m/s, differing ONLY in whether
#  the phases carry a variable equation of state. So neither the through-flow
#  boundary conditions, the O-grid, the wall functions nor the boiling models are
#  responsible: the failure is specific to the compressible pressure equation.
#
#  WHAT HAS BEEN RULED OUT
#
#    the boiling models        - disabling both still diverges
#    heating                   - q = 0 still diverges
#    BCs / mesh / wall funcs   - identical setup with ConstEos phases is STABLE
#    velocity-in/pressure-out  - the repo's own validated subsonic compressible
#                                case (2D_cylinder_heated_unsteady, CPISO) uses
#                                exactly this arrangement
#    psi*dp/dt                 - `RealFluid(..., p_ref=p_sat)` sets psi to
#                                EXACTLY zero and it still diverges
#    startup shock             - initialising p_rgh to the developed frictional
#                                profile does not help (max|U| = 1.2e4 m/s)
#
#  Cutting dt by 200x to 1e-8 s largely suppresses it (max|U| = 13 m/s), which is
#  the signature of a stiff explicitly-treated source - but which source is not
#  yet established.
#
#  EVERY COMPRESSIBLE SOURCE TERM WAS DISABLED IN TURN - NONE IS THE CAUSE
#
#    psi*dp/dt          -> p_ref locking sets psi to 0    : still diverges
#    pressure work      -> update_pressure_work! zeroed   : still diverges
#    thermal expansion  -> update_expansion! zeroed       : still diverges
#    make_symmetric!    -> added to the compressible solve: NO CHANGE at all
#
#  LEADING SUSPECT BY ELIMINATION
#
#  With those zeroed the compressible branch is numerically almost identical to
#  the incompressible one, and the substantive difference left is that
#  `update_phase_state!` refreshes properties every step, so rho = rho(T) VARIES.
#  `phi_gf!` and `well_balanced_pressure_grad!` both build the buoyancy term from
#  snGrad(rho) and are well balanced by construction only when rho is piecewise
#  constant - exactly what section 5.4 of dev_notes_LH2_implementation_plan.md
#  predicted as "the item most likely to cost unplanned time".
#
#  Full evidence and the remaining decisive test in dev_notes_LH2_pipe_boiling.md.
#
#  WHAT IS USABLE TODAY
#
#  Setting both phases to `ConstEos` (constant density) makes the case stable
#  immediately, at the cost of the real-gas compressibility. That is enough to
#  do the single most valuable first validation - check (b) under "Validation"
#  below, the non-boiling Dittus-Boelter branch - which needs no compressibility
#  and isolates the mesh, wall functions and turbulence model.
#
#  Not validated against the paper's data either way; see "Validation" at the end.
# =============================================================================

using XCALibre
using Test

# -----------------------------------------------------------------------------
# Case selection
# -----------------------------------------------------------------------------
# The paper sweeps three saturation pressures and a range of flow velocities for
# each of four tube geometries. `CASE` must match the geometry the mesh was
# generated for (see examples/0_GRIDS/lh2_pipe_sector/make_lh2_pipe_sector.jl).
#
#   p_sat [MPa]   T_sat [K]   (paper, "Results and discussion")
#      0.4          26.0
#      0.7          29.0
#      1.1          31.9
#
CASE = :D6_L250
p_sat = 0.7e6            # [Pa]
U_inlet_mag = 5.33       # [m/s]   paper Figs. 3-4 span 1.5 - 11.6 m/s

# Wall heat flux. Fig. 3(a)/4(a) put the developed nucleate boiling regime for
# this geometry between roughly 1e4 and 1e5 W/m^2, with DNB near 6e4 W/m^2 at
# this velocity (Fig. 5b). 3e4 W/m^2 sits comfortably inside nucleate boiling,
# which is the regime this model is valid in - RPI says nothing about the
# post-DNB film boiling branch.
WALL_HEAT_FLUX = 3e4   # [W/m^2]

D, L_heated = if CASE === :D4_L100
    4.0e-3, 100.0e-3
elseif CASE === :D4_L167
    4.0e-3, 167.0e-3
elseif CASE === :D6_L150
    6.0e-3, 150.0e-3
elseif CASE === :D6_L250
    6.0e-3, 250.0e-3
else
    error("Unknown CASE: $CASE")
end

# -----------------------------------------------------------------------------
# Mesh
# -----------------------------------------------------------------------------
# 90 degree O-grid sector, meshed for y+ = 30-50. Generate with:
#     cd examples/0_GRIDS/lh2_pipe_sector
#     julia make_lh2_pipe_sector.jl && ./run_blockMesh.sh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "lh2_pipe_sector", "constant", "polyMesh")
mesh = FOAM3D_mesh(mesh_file, scale=1.0, integer_type=Int64, float_type=Float64)

backend = CPU(); workgroup = AutoTune(); activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# -----------------------------------------------------------------------------
# Real-fluid properties
# -----------------------------------------------------------------------------
# Tabulated once at setup from the Helmholtz H2 equation of state. The range must
# cover the whole solution: pressure spans the operating point plus the frictional
# and hydrostatic drop, temperature spans the inlet liquid to the superheated
# wall. `table_range_report` at the end checks whether the run stayed inside it.
p_table = (0.25e6, 1.25e6)
T_table = (19.0, 40.0)

# Saturation from the same EOS rather than an Antoine fit: h_fg falls by more
# than half between 0.4 MPa and the critical point, so a constant latent heat is
# not usable across the paper's pressure sweep.
saturation = build_saturation_curve(H2(), p=p_table, T=(19.0, 33.0), np=201, nT=201)

saturation = ConstantSaturation(saturation, p_sat)

T_sat = saturation_temperature(saturation, p_sat)
h_fg = latent_heat(saturation, p_sat, 0.0)
sigma_lv = calculate_surface_tension(H2(), T_sat)


# --- vapour: Peng-Robinson (ANALYTIC, no table) -----------------------------
# The vapour is where the real equation of state actually earns its keep. At
# these pressures it is at up to 85% of the critical pressure and its
# compressibility departs from ideal by a factor of 1.4 (0.4 MPa) to 4.4
# (1.1 MPa), so `IdealGas` is not defensible and the density must vary.
#
# WHY THE CUBIC RATHER THAN THE TABULATED HELMHOLTZ EOS
#
# The tabulated path was found to produce density tables that are discontinuous
# and non-monotonic INSIDE their declared range: a rectangular (p,T) grid
# necessarily includes states where the requested branch does not exist, and the
# fallback used there stepped the density back to its saturation value mid-column
# (see `validate_property_table`, which now rejects such a table at build time).
#
# Peng-Robinson has none of that. It is solved by Cardano in closed form, runs in
# the kernel at every cell, and so gives exactly the same smooth rho(p,T)
# everywhere - no interpolation, no off-branch patching. That is what makes it
# the right EOS for testing the SOLVER: any misbehaviour is then the solver's,
# not the property data's.
#
# ACCURACY: Peng-Robinson is a generic cubic and hydrogen has a negative acentric
# factor (omega = -0.219), outside the range its alpha-function was fitted over.
# At the operating point it gives rho_v = 9.01 against the Helmholtz 8.82
# (+2.2%), and rho_l = 61.77 against 56.75 (+8.8%). Fine for exercising the
# solver; NOT a substitute for the Helmholtz EOS in quantitative work.
# -----------------------------------------------------------------------------
# VAPOUR_EOS - switch the gas-phase equation of state
# -----------------------------------------------------------------------------
#   :ideal  - rho = p/(R*T). The simplest compressible EOS there is: analytic,
#             perfectly smooth and monotonic everywhere, with no branch structure
#             at all (no saturation line#   :const  - rho fixed at the saturated-vapour value. The mixture is then
#             INCOMPRESSIBLE (`is_compressible_multiphase` is false), which takes
#             the incompressible branch of the pressure equation - no psi term and
#             no implicit pressure-convection term. Useful as a control: it is the
#             only setting under which the compressible pressure path is entirely
#             out of the picture.
#
#   :ideal  - rho = p/(R*T). The simplest compressible EOS there is: analytic,
#             perfectly smooth and monotonic everywhere, with no branch structure
#             at all (no saturation line, no spinodal, no critical point). psi is
#             exactly 1/p and beta exactly 1/T. If a compressible case misbehaves
#             under THIS, the equation of state cannot be the reason - which is
#             what makes it the right control.
#
#             Not defensible as physics here: at 0.7 MPa the real vapour is
#             ~2.6x more compressible than p/(RT) says (psi = 2.63e-6 vs
#             1.43e-6), and worse approaching p_crit = 1.2964 MPa.
#
#   :pr     - Peng-Robinson cubic. Analytic and kernel-evaluated like :ideal, but
#             with real compressibility. Its vapour branch is only a distinct
#             root above T_sat (see the note below).
#
#   :table  - tabulated Helmholtz. Most accurate, but see `validate_property_table`
#             for why a rectangular (p,T) table is hazardous here.
# Saturated-vapour properties at the operating point. Needed before the switch
# because `:const` takes its density from here, and mu/k/cp come from here in
# every case - a cubic and an ideal gas both say nothing about transport
# properties.
gh2_sat = phase_properties_at(H2(), p_sat, T_sat, branch=:vapour)

VAPOUR_EOS = :const

gh2_eos = if VAPOUR_EOS === :const
    gh2_sat.rho
elseif VAPOUR_EOS === :ideal
    IdealGas(M = 2.01588e-3)
elseif VAPOUR_EOS === :pr
    PengRobinson(H2(), branch=:vapour)
elseif VAPOUR_EOS === :table
    RealFluid(H2(), :vapour, p=p_table, T=T_table, np=81, nT=81).rho
else
    error("Unknown VAPOUR_EOS: $VAPOUR_EOS")
end

# `IdealGas` carries its own expansivity exactly (`phase_betaT` returns 1, i.e.
# beta = 1/T), so it needs no `beta` model; Peng-Robinson supplies one derived
# from the same cubic as its density.
# `IdealGas` carries beta exactly (`phase_betaT` returns 1, i.e. beta = 1/T) so it
# needs none; Peng-Robinson derives one from its own cubic; a constant density
# takes the saturated value.
gh2_beta = VAPOUR_EOS === :pr    ? PengRobinsonBeta(gh2_eos) :
           VAPOUR_EOS === :const ? gh2_sat.beta :
           nothing

# The cubic supplies rho, psi and beta only - it says nothing about transport
# properties, and its cp needs an ideal-gas correlation it does not carry. Those
# three come from the Helmholtz EOS at the operating point, as the liquid's do.

# !!! OPERATING RANGE - READ BEFORE CHANGING T LIMITS !!!
#
# The vapour branch of a cubic is only a distinct root ABOVE T_sat. Below it the
# cubic has a single root and that root is the LIQUID one, so a vapour lookup at
# T < T_sat returns a liquid density - a 13.8x step measured over this envelope.
#
# This case is safe as configured because the inlet is saturated (T_inlet =
# T_sat) and the wall only heats, so T >= T_sat = 29.155 K throughout. If the
# temperature solver's `limit` is ever widened below T_sat, or inlet subcooling
# is introduced, check the branch first:
#
#     pr_table_report(gh2_eos, p=p_table, T=(T_sat, 40.0))
#
# `branch absent` counts the nodes where the vapour is not a distinct root, and
# `largest step` is the discontinuity the solver would meet there.

# --- liquid: CONSTANT at the saturation state -------------------------------
# The liquid does not need tabulating. Its properties are taken from the same
# Helmholtz EOS as the vapour, evaluated once at the operating point, so the two
# phases stay mutually consistent and there are no hardcoded magic numbers.
#
# Justified because the liquid is nearly incompressible over the pressure range
# the run actually spans (the frictional drop is ~618 Pa out of 0.7 MPa, giving
# drho/rho = psi*dp ~ 6e-5), and because it stays close to saturation - the wall
# superheat is a few kelvin. What it costs is the temperature dependence of cp
# and beta, which near the critical point is not negligible; that is a stated
# simplification, not a free lunch.
#
# It also removes `snGrad(rho)` from the mixture entirely while alpha ~ 1, which
# matters for the buoyancy discretisation (see the notes at the top of this
# file and section 5.4 of dev_notes_LH2_implementation_plan.md).


# Saturated-liquid properties at the operating point, straight from the EOS.
lh2_sat = phase_properties_at(H2(), p_sat, T_sat, branch=:liquid)

@info """Operating point
    p_sat  = $(p_sat/1e6) MPa
    T_sat  = $(round(T_sat, digits=3)) K
    h_fg   = $(round(h_fg/1e3, digits=2)) kJ/kg
    sigma  = $(round(sigma_lv*1e3, digits=4)) mN/m"""

# The liquid enters saturated: the paper's title case is *saturated* liquid
# hydrogen, so there is no inlet subcooling to speak of.
T_inlet = T_sat

# -----------------------------------------------------------------------------
# Turbulence inlet values
# -----------------------------------------------------------------------------
rho_l_ref = lh2_sat.rho
mu_l_ref = lh2_sat.mu
nu_l_ref = mu_l_ref/rho_l_ref
Re = rho_l_ref*U_inlet_mag*D/mu_l_ref

Tu = 0.05                                    # 5% inlet turbulence intensity
k_inlet = 1.5*(Tu*U_inlet_mag)^2
omega_inlet = sqrt(k_inlet)/(0.07*D*0.09^0.25)
nut_inlet = k_inlet/omega_inlet

@info """Flow
    Re     = $(round(Int, Re))
    rho_l  = $(round(rho_l_ref, digits=3)) kg/m^3
    mu_l   = $(round(mu_l_ref*1e6, digits=4)) uPa.s
    k_in   = $(round(k_inlet, digits=5)),  omega_in = $(round(omega_inlet, digits=1))"""

velocity = [0.0, 0.0, U_inlet_mag]           # upward flow, tube axis is +z
noSlip = [0.0, 0.0, 0.0]
gravity = Gravity([0.0, 0.0, -9.81])

# -----------------------------------------------------------------------------
# Physics
# -----------------------------------------------------------------------------
# Phase 1 is the tracked phase: alpha = 1 is liquid hydrogen.
model = Physics(
    time = Transient(),
    fluid = Fluid{Multiphase}(
        # Drift-flux mixture. `diameter` is the dispersed bubble diameter used
        # for the slip velocity; 0.5 mm is the order the RPI departure diameter
        # predicts for hydrogen at this pressure, and it should be revisited
        # alongside `TolubinskyKostanchuk`'s coefficients.
        # `alpha_transport` is a keyword of `Mixture`, NOT of `Fluid{Multiphase}`.
        # Placed on the fluid it lands in `physics_properties`, is never read, and
        # the model silently keeps its `:mules` default - the giveaway being an
        # alpha residual of exactly zero, since explicit MULES does no linear
        # solve and there is no residual to report.
        model = Mixture(diameter = 1e-9, alpha_transport = :implicit),
        dispersion_Sc = 0.9,     # turbulent Schmidt number
        p_abs_limit = (0.3e6, 1.2e6),   # [Pa] absolute
        rD_ref_density = lh2_sat.rho,   # 56.747 kg/m³

        # Phase 1 (tracked, alpha = 1) is the liquid: constant properties at the
        # saturation state. Phase 2 is the vapour: Peng-Robinson density and
        # expansivity (analytic, evaluated per cell), with transport properties
        # and cp held at the operating point.
        #
        # `PengRobinsonBeta` shares the same `gh2_eos` object as the density, so
        # the expansivity is a derivative of exactly the density field being
        # solved with and the two cannot drift apart.
        phases = (
            Phase(rho  = lh2_sat.rho,
                  mu   = lh2_sat.mu,
                  k    = lh2_sat.k,
                  cp   = lh2_sat.cp,
                  beta = lh2_sat.beta),
            Phase(rho  = gh2_eos,
                  mu   = gh2_sat.mu,
                  k    = gh2_sat.k,
                  cp   = gh2_sat.cp,
                  beta = gh2_beta)
        ),

        # --- bulk interfacial phase change -----------------------------------
        # `nothing` for now: RPI alone is the configuration that runs.
        #
        # !!! IF RE-ENABLING Lee, USE sigma = 1e-6, NOT 10 !!!
        # `sigma` is an ACCOMMODATION COEFFICIENT - a probability, physically
        # bounded by 1. The paper's parametric values are 1e-6 (baseline), 1e-7
        # and 1e-8, and `Lee`'s own default is 1e-6. A previous version of this
        # file carried `Lee(sigma = 10)`, which is 1e7x the baseline and 10x the
        # physical maximum; it enters the rate linearly through
        #     beta = sigma*sqrt(1/(2 pi R_sp T_sat))*L*rho_l/(rho_l - rho_v)
        # so the evaporation rate was seven orders of magnitude too large. Every
        # measurement taken with it is uninterpretable.
        #
        #   phase_change = Lee(sigma = 1e-6),
        #
        # `Lee` and `Schrage` need the vapour specific gas constant for their
        # kinetic prefactor. An EOS that carries one supplies it; a constant
        # density cannot, so pass it explicitly:
        #   phase_change = Lee(sigma = 1e-6, R = 4124.5),   # [J/kg/K]
        phase_change = nothing,

        # --- wall nucleate boiling -------------------------------------------
        # Kurul & Podowski heat flux partitioning, q_w = q_conv + q_quench + q_evap,
        # inverted for the wall temperature since this case prescribes the FLUX.
        #
        # Every empirical closure is swappable through the Physics API:
        #   site_density        = LemmertChawla() | HibikiIshii()
        #   departure_diameter  = TolubinskyKostanchuk() | KocamustafaogullariIshii()
        #   departure_frequency = Cole()
        #   influence_area      = DelValleKenning() | ConstantInfluenceArea()
        #
        # COEFFICIENTS ARE WATER FITS. LemmertChawla (m = 210, n = 1.805) and
        # TolubinskyKostanchuk (0.6 mm, 45 K) have no established cryogenic
        # values. They match STAR-CCM+ and Fluent because those codes use the
        # same water fits - that is evidence the implementation is faithful, not
        # that the coefficients suit hydrogen. Expect recalibration to be part of
        # validation.
        #
        # `start_iteration` holds the model off until the base flow is
        # established: h_c is built from the turbulence model's friction
        # velocity, so engaging RPI while k, omega and U are still at their
        # uniform initial values feeds N_a ~ dT_sup^1.805 a wall temperature
        # derived from a flow that does not yet exist. Zero engages immediately.
        wall_boiling = RPI(
            patches = (:pipeWall,),
            site_density = LemmertChawla(),
            departure_diameter = TolubinskyKostanchuk(),
            start_iteration = 0),

        # --- source under-relaxation ----------------------------------------
        # Independent temporal damping of the two vapour sources. Both are
        # stiff, for different reasons: the bulk models respond to (T - T_sat)
        # across the interface, the wall model to the wall superheat through
        # N_a ~ dT_sup^1.805, which is far steeper. Lee in particular is
        # reported by Fernandes et al. to diverge at sigma = 1e-6.
        #
        # These BLEND against the previous step rather than scaling the rate, so
        # the converged answer is unchanged - at steady state the relaxed and
        # unrelaxed solutions coincide. 1.0 is no relaxation.
        #
        # NOTE: they do NOT help with the divergence documented at the top of
        # this file. That was measured: relax = 0.01 on both (sources
        # effectively off) diverges identically to relax = 1.0, which is
        # consistent with the case also diverging with both models removed
        # entirely. They are here for the stiffness of the models themselves.
        phase_change_relax = 0.5,
        wall_boiling_relax = 0.5,

        # --- thermo-acoustic coupling: OFF for this case ---------------------
        # These two terms exist only on the compressible path and together form
        # a closed, explicitly-evaluated loop:
        #
        #   dT -> expansion = beta*dT/dt -> dp -> dp/dt
        #      -> S_T = beta*T*dp/dt -> dT      (closes)
        #
        # Each traversal divides by dt twice, so treating it explicitly carries
        # an ACOUSTIC CFL limit: dt < dx/c, with c = 1/sqrt(rho*psi) ~ 420 m/s
        # in liquid hydrogen here. Against the 34 um wall cells that is
        # dt < 8e-8 s, and this case needs dt ~ 2e-6 - about 25x over the limit.
        #
        # Damping is not enough; the loop has to be broken. Measured (q=0):
        #   pw=0.5,  exp=1.0  -> max|U| = 4.1e39   diverged
        #   pw=0.05, exp=1.0  -> max|U| = 1.9e37   diverged
        #   pw=0.0,  exp=0.0  -> max|U| = 7.67     STABLE, T in [28.97, 29.48]
        #
        # Setting both to zero is justified rather than a fudge: BOTH terms
        # vanish at steady state (dT/dt -> 0, dp/dt -> 0), so a steady-state
        # answer is unaffected. What is lost is the transient thermo-acoustic
        # response, which is not what this case is for. A self-pressurising
        # tank, where that response IS the answer, must leave them on.
        #
        # `expansion_relax` damps only the THERMAL part; the volume created by
        # phase change is added afterwards and always survives.
        pressure_work_relax = 0.0,
        expansion_relax     = 0.0,

        # -------------------------------------------------------------------
        # Pressure equation form: :volume (default) or :mass
        # -------------------------------------------------------------------
        # CURRENTLY :volume - DELIBERATELY, to isolate one change at a time.
        #
        # The Peng-Robinson vapour EOS and the mass-form pressure equation are
        # both new here. Running them together confounds the result: if the
        # pressure misbehaves there is no way to attribute it. The EOS is the
        # change being tested right now, so the pressure equation goes back to
        # the formulation every earlier measurement was made against.
        #
        # To test the mass form, flip this to :mass and change NOTHING else.
        #
        # Why :mass is worth returning to: momentum and energy convect with
        # `rhoPhi`, a mass flux, but the volume form never constrains it - and
        # with rho_l/rho_v ~ 57 the volumetric and mass fluxes are nowhere near
        # proportional. The measured discrete mass residual under the volume form
        # was rel = 1.0 in EVERY case run, including the constant-density
        # control. It is also what XCALibre's own single-phase compressible
        # solver (CPISO) has always done - `rhorDf = rhof*rD`, `mdotf` a mass
        # flux, `div(rho*u)` on the RHS - so the multiphase solver was the
        # outlier, not the innovation.
        #
        # On the sealed-ullage regression :mass cut the vapour mass drift by a
        # factor of 1670 (8.1e-6 -> 4.9e-9) while reproducing the exact
        # analytical dp/dt unchanged. On this case it made no difference: leg D
        # of the gravity study diverged with max|U| = 1.48e4 and a pressure
        # residual of 1.06e-16, i.e. converged to machine precision and diverging
        # anyway. See dev_notes_LH2_pipe_boiling.md.
        pressure_form = :volume,

        saturation  = saturation,
        h_fg        = h_fg,
        sigma       = sigma_lv,
        p_operating = p_sat,

        # --- buoyancy formulation: REFERENCE density, not local --------------
        # Selects the reference form of the buoyancy face flux in `phi_gf!`,
        #
        #     rDf*(rhof - rho_ref)*gn*area          instead of
        #     -ghf*snGrad(rho)*rDf                  (the default, rho_ref = nothing)
        #
        # The default form DIFFERENTIATES the mixture density. Once boiling makes
        # alpha - and therefore rho_m - carry any odd-even content, `snGrad`
        # amplifies it by 1/delta, which for the 34 um wall cells is ~3e4. That
        # contaminated flux is added straight to `mdotf`, drives div(u), hence p,
        # hence alpha: a closed loop that sustains a static checkerboard.
        #
        # The reference form uses `rhof` LINEARLY, so a checkerboard passes
        # through at amplitude instead of being multiplied by 3e4.
        #
        # This is also what STAR-CCM+ does - it solves a piezometric pressure
        # with a user-set reference density for exactly this reason.
        #
        # TRADE-OFF: the local form is well balanced BY CONSTRUCTION across a
        # sharp interface, and switching to the reference form regressed the VOF
        # hydrostatic test from <1e-7 to 7.4e-5. That does not apply here - this
        # is a dispersed bubbly flow at alpha ~ 0.998 with no sharp interface -
        # but a stratified tank case should keep the default.
        rho_ref = lh2_sat.rho,

        gravity = gravity
    ),
    turbulence = RANS{KOmegaSST}(walls=(:pipeWall, :wallUnheated)),
    energy = Energy{TwoPhaseTemperature}(Tref=0, Pr_t=0.85),#T_sat, Pr_t=0.85),
    domain = mesh_dev
)

# -----------------------------------------------------------------------------
# Boundary conditions
# -----------------------------------------------------------------------------
BCs = assign(
    region = mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Wall(:pipeWall, noSlip),
            Wall(:wallUnheated, noSlip),
            Symmetry(:symmetryX),
            Symmetry(:symmetryY),
        ],
        # The pressure LEVEL is set at the outlet. Unlike the sealed K-Site tank,
        # this is a flow-through domain, so the absolute pressure is imposed
        # rather than emerging from the compressibility term. `p_rgh = 0` at the
        # outlet plus `p_operating = p_sat` puts the outlet at the saturation
        # pressure, which is how the experiment is controlled.
        p_rgh = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Zerogradient(:pipeWall),
            Zerogradient(:wallUnheated),
            Symmetry(:symmetryX),
            Symmetry(:symmetryY),
        ],
        alpha = [
            Dirichlet(:inlet, 1.0),          # pure liquid entering
            Zerogradient(:outlet),
            Zerogradient(:pipeWall),
            Zerogradient(:wallUnheated),
            Symmetry(:symmetryX),
            Symmetry(:symmetryY),
        ],
        T = [
            Dirichlet(:inlet, T_inlet),
            Zerogradient(:outlet),
            FixedHeatFlux(:pipeWall, WALL_HEAT_FLUX),
            Zerogradient(:wallUnheated),     # adiabatic development section
            Symmetry(:symmetryX),
            Symmetry(:symmetryY),
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Zerogradient(:outlet),
            KWallFunction(:pipeWall),
            KWallFunction(:wallUnheated),
            Symmetry(:symmetryX),
            Symmetry(:symmetryY),
        ],
        omega = [
            Dirichlet(:inlet, omega_inlet),
            Zerogradient(:outlet),
            OmegaWallFunction(:pipeWall),
            OmegaWallFunction(:wallUnheated),
            Symmetry(:symmetryX),
            Symmetry(:symmetryY),
        ],
        nut = [
            Dirichlet(:inlet, nut_inlet),
            Zerogradient(:outlet),
            NutWallFunction(:pipeWall),
            NutWallFunction(:wallUnheated),
            Symmetry(:symmetryX),
            Symmetry(:symmetryY),
        ],
    )
)

# -----------------------------------------------------------------------------
# Numerics
# -----------------------------------------------------------------------------
schemes = (
    U     = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
    alpha = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
    p     = Schemes(time=Euler, divergence=Upwind, gradient=Gauss,    laplacian=Linear),
    p_rgh = Schemes(time=Euler, divergence=Upwind, gradient=Gauss,    laplacian=Linear),
    T     = Schemes(time=Euler, divergence=Upwind, gradient=Gauss,    laplacian=Linear),
    omega = Schemes(time=Euler, divergence=Upwind, gradient=Gauss,    laplacian=Linear),
    k = Schemes(time=Euler, divergence=Upwind, gradient=Gauss,    laplacian=Linear),
    y     = Schemes(gradient=Midpoint),
)

solvers = (
    U = SolverSetup(
        solver=Bicgstab(), preconditioner=DILU(),
        convergence=1e-7, relax=1, rtol=1e-2, atol=1e-10),
    p_rgh = SolverSetup(
        solver=AMG(), preconditioner=DILU(),
        convergence=1e-7, relax=0.9, rtol=1e-3, atol=1e-12, itmax=1000),
    alpha = SolverSetup(
        solver=Bicgstab(), preconditioner=DILU(),
        convergence=1e-7, relax=1, rtol=1e-2, atol=1e-10),
    T = SolverSetup(
        solver=Bicgstab(), preconditioner=DILU(),
        convergence=1e-7, relax=1, rtol=1e-2, atol=1e-10,
        limit=(25, T_table[2])),
    k = SolverSetup(
        solver=Bicgstab(), preconditioner=DILU(),
        convergence=1e-7, relax=1, rtol=1e-2, atol=1e-10),
    omega = SolverSetup(
        solver=Bicgstab(), preconditioner=DILU(),
        convergence=1e-7, relax=1, rtol=1e-2, atol=1e-10),
    # Wall distance, solved once at setup for the SST blending functions.
    # `KOmegaSST` requires it; without a `y` entry here the run fails inside
    # `wall_distance!` rather than at configuration time.
    y = SolverSetup(
        solver=Cg(), preconditioner=Jacobi(),
        convergence=1e-8, relax=1, rtol=1e-2),
)

# Run to steady state. The residence time is L_total/U ~ 0.06 s for this
# geometry, so a few hundred flow-throughs is ample for the thermal field and
# the wall vapour generation to settle.
dt = 2.0e-5
n_flow_throughs = 2
L_total = L_heated + 10*D
iterations = round(Int, n_flow_throughs*(L_total/U_inlet_mag)/dt)

runtime = Runtime(
    iterations = iterations,
    time_step = dt,
    write_interval = 100,#round(Int, iterations/50),
    adaptive = AdaptiveTimeStepping(maxCo=0.5, maxAlphaCo=0.25)
)

config = Configuration(
    solvers=solvers, schemes=schemes,
    runtime=runtime, hardware=hardware, boundaries=BCs)

initialise!(model.momentum.U, velocity)
initialise!(model.fluid.p_rgh, 0.0)
initialise!(model.energy.T, T_inlet)
initialise!(model.fluid.alpha, 1.0)          # pipe starts full of liquid
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, omega_inlet)
initialise!(model.turbulence.nut, nut_inlet)

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
residuals = run!(model, config, inner_loops=5)

# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------
# These are sanity conditions, not a validation. They catch the failure modes
# that would make any comparison meaningless.

@test all(0.0 .<= model.fluid.alpha.values .<= 1.0)      # bounded volume fraction
@test all(isfinite, model.energy.T.values)
@test minimum(model.energy.T.values) >= T_sat - 1e-9     # no unphysical undershoot
@test maximum(model.energy.T.values) < T_table[2]

# Boiling must actually have happened: with 3e4 W/m^2 on the wall the pipe
# cannot remain single phase.
@test minimum(model.fluid.alpha.values) < 1.0

# Equation-of-state coverage over the envelope the run ACTUALLY visited.
#
# `IdealGas` has nothing to check - rho = p/(R*T) is smooth and single-valued for
# any p, T > 0, which is exactly why it is the useful control. The cubic does:
# its vapour branch is only a distinct root above T_sat, and past that point a
# vapour lookup silently returns the LIQUID root. That is the analytic path's one
# silent failure mode, so it is asserted where it exists.
p_abs = ScalarField(mesh_dev)
@. p_abs.values = model.momentum.p.values
p_lo, p_hi = extrema(p_abs.values)
T_lo, T_hi = extrema(model.energy.T.values)

if VAPOUR_EOS === :pr
    report = pr_table_report(gh2_eos, p=(p_lo, p_hi), T=(T_lo, T_hi))
    @test report.n_missing == 0        # vapour was a distinct root everywhere
    @test report.worst_ratio < 1.5     # and rho varied smoothly across it
else
    @info "EOS envelope actually visited" VAPOUR_EOS p=(p_lo, p_hi) T=(T_lo, T_hi)
    @test p_lo > 0 && T_lo > 0         # IdealGas is valid for any positive p, T
end

# -----------------------------------------------------------------------------
# Validation - what is still needed
# -----------------------------------------------------------------------------
# The paper's quantitative results are:
#
#   (a) the nucleate boiling curve, q vs dT_sat = T_w - T_sat  (Figs. 3, 4).
#       The natural comparison is the RPI-solved wall temperature, available as
#       the `T_wall` face field of the wall boiling state. Reproducing the
#       measured curve is the real test of the LemmertChawla + Tolubinsky
#       coefficients, which were fitted to WATER and have no established values
#       for cryogens - they should be expected to need recalibration.
#
#   (b) the non-boiling branch agreeing with Dittus-Boelter (paper, Conclusion).
#       This is the cleanest first check and needs no boiling at all: run with
#       `wall_boiling = nothing` and a low q_w, and compare the wall heat
#       transfer coefficient against Nu = 0.023 Re^0.8 Pr^0.4. It isolates the
#       turbulence model, the wall functions and the mesh from the boiling
#       closures, and it should be done BEFORE any boiling comparison.
#
#   (c) the DNB heat flux correlation (Eqs. 1-5). Out of scope here: RPI models
#       nucleate boiling and has no DNB criterion. Predicting departure needs a
#       separate model, and the `alpha_min` ramp in `RPI` is a numerical
#       safeguard, NOT a dryout prediction.
#
# The measured data are not in the repository; they would have to be digitised
# from the paper's figures.                                          # TO OBTAIN
