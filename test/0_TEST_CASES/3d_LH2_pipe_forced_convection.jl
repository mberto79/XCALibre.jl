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
# 0.4 MPa / 5.53 m/s is the condition the digitised curve in
# `data/tatsumoto2014_D6_L250_0.4MPa_5.53ms.csv` was measured at. CASE, p_sat and
# U_inlet_mag must move together - a curve from a different geometry, pressure or
# velocity is not comparable.
p_sat = 0.4e6            # [Pa]   -> T_sat ~ 26.0 K
U_inlet_mag = 5.33       # [m/s]   paper Figs. 3-4 span 1.5 - 11.6 m/s

# Wall heat flux. Fig. 3(a)/4(a) put the developed nucleate boiling regime for
# this geometry between roughly 1e4 and 1e5 W/m^2, with DNB near 6e4 W/m^2 at
# this velocity (Fig. 5b), and the digitised curve ends at 64 kW/m^2.
#
# 6.6e4 is JUST ABOVE that - deliberately. The excess over CHF is what drives the
# excursion, so a small excess gives the gentlest traverse; going far above adds
# violence without adding information. Anything at or below 6.4e4 stays on the
# nucleate branch and never departs.
WALL_HEAT_FLUX = 6.6e4   # [W/m^2]

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# SITE_DENSITY_FIT - trading fit accuracy against numerical stiffness
# -----------------------------------------------------------------------------
# `N_a = (m*dT_sup)^n`. The measured branch above the knee is nearly VERTICAL, so
# a power law can only chase it with a large `n` - and `n` IS the stiffness of
# the whole wall closure, because q_evap ~ dT_sup^n is what the partition leans
# on once bubbles cover the wall. That makes the exponent a direct trade between
# how well the curve is reproduced and how violently the vapour source responds
# to cell-to-cell noise in the near-wall temperature.
#
# `m` and `n` are NOT independent: N_a = (m*dT_sup)^n means `m` only sets the
# amplitude, through m^n, while `n` sets the slope. Changing `n` alone would
# shift the whole curve, so each `m` below is REFITTED at its `n` by grid search
# against `data/tatsumoto2014_D6_L250_0.4MPa_5.53ms.csv`, with the exponential
# influence area in use. Measured trade:
#
#   fit      n      m      peak stiffness   RMS(log dT)      dT_sup at CHF
#                          d(ln q)/d(ln dT)  q > 20 kW/m^2   (measured 1.642 K)
#   :n21   21.17  1.085        16.6           0.0537            1.692  (+3.0%)
#   :n15   15.0   1.362        11.7           0.0664            1.731  (+5.4%)
#   :n12   12.0   1.653         9.3           0.0772            1.766  (+7.6%)
#   :n10   10.0   2.006         7.7           0.0877            1.803  (+9.8%)
#   :n8     8.0   2.691         6.2           0.1025            1.852 (+12.8%)
#   :n6     6.0   4.364         4.6           0.1269            1.947 (+18.6%)
#
# The RMS column is over q > 20 kW/m^2 only. Across the FULL curve every entry
# sits near 0.31, because all of them are ~22% low at 15 kW/m^2 - the low-flux
# window cannot be fitted by ANY of these. That is a limitation of the power-law
# form, not of a particular exponent, and it is why the whole-curve residual is
# useless for choosing between them.
#
# HOW TO USE THIS. Do not tune - BISECT. Run `:n6` first, purely as a diagnostic:
# its stiffness is under a third of `:n21`. If the case is stable there, the
# exponent really is driving the instability and `:n12`/`:n10` is the operating
# compromise. If `:n6` still breaks up, the exponent is NOT the cause, and
# lowering it further costs accuracy for nothing.
# =============================================================================
#  RESOLVED 2026-08-19 - why the low-flux window "could not be fitted"
# =============================================================================
#
#  The paragraph above records that every entry in the Pareto front is ~22% low
#  at 15 kW/m^2 and concludes this is "a limitation of the power-law form, not of
#  a particular exponent". That conclusion was wrong, and the cause was not in the
#  site density at all: it was `h_c`.
#
#  The whole front was fitted with `h_c` from Petukhov at y+ = 40, which gives
#  14199 W/m^2/K. But the LOW-flux end of a boiling curve IS single-phase
#  convection - the wall is barely superheated and nucleation does nothing - so
#  the data determines `h_c` directly there:
#
#      lowest measured point:  q = 4.32 kW/m^2 at dT_sup = 0.663 K
#      => h_c = q/dT_sup = 6516 W/m^2/K;  least squares on the lowest 3 points
#         gives 7045 W/m^2/K.
#
#  Petukhov is therefore 2x HIGH. With it, pure convection alone already puts the
#  lowest point at dT = 4320/14199 = 0.30 K against a measured 0.663 K - and no
#  site density can repair that, because nucleation only ADDS flux. The low-flux
#  window was unfittable because the convective coefficient was wrong, not because
#  the power law was inadequate.
#
#  (The same gap is recorded elsewhere in this file from the other direction: a
#  measured h_conv of 3692-5388 W/m^2/K against a Dittus-Boelter ~10,200.)
#
#  Refitting the WHOLE nucleate branch with h_c = 7045:
#
#      RMS(log dT)   shipped water fit  0.8875
#                    old front (best)   ~0.31 across the full curve
#                    NEW :lh2 fit       0.0673        <- 13x better than water
#
#  and the optimum is now INTERIOR to the search grid. The old front's `n = 21.17`
#  came from `calibrate_rpi_highflux.jl`, which fits only 21-59 kW/m^2 and so
#  never saw the constraint the low-flux points impose; with the correct `h_c` the
#  whole branch prefers n ~ 7.8. That reconciles the two numbers - they are fits
#  to different windows with different `h_c`, not a disagreement about physics.
#
#  CONSEQUENCE FOR THE CFD. `h_c` here comes from the WALL FUNCTION, not from this
#  calibration, and the fit moves with it. So the wall function has a target now:
#  if the converged `h_conv` on the heated patch is not near 7045 W/m^2/K, the
#  convection/evaporation split is wrong and the site density is compensating for
#  it. Rung 0.6 of the validation plan is the test for that, and it is worth doing
#  before reading anything into a predicted boiling curve.
#
#  Reproduce with:  julia --project=. test/0_TEST_CASES/calibrate_rpi_lh2.jl
# =============================================================================
# -----------------------------------------------------------------------------
# Pressure equation form - and the SOLVER MUST FOLLOW IT
# -----------------------------------------------------------------------------
# `:mass` scales the pressure Laplacian by `rho_f`, and the resulting matrix is
# NOT SYMMETRIC. `AMG()` defaults to `mode = Cg()`, which requires symmetry and
# throws when it is not there.
#
# The failure is delayed and therefore confusing: with no vapour, `rho_f` is
# nearly uniform, the matrix is nearly symmetric and the check passes. It fires
# the moment boiling produces a real void fraction - which on the staircase means
# the run survives initialisation and the first level and then dies at the second,
# looking like an instability at that heat flux rather than a solver mismatch.
#
# Measured: `3d_LH2_pipe_boiling_curve.jl` completes level 1 at 10 kW/m^2
# (alpha_min = 7.8e-60, i.e. no vapour) and raises
# `ArgumentError: AMG(mode=Cg()) requires a symmetric matrix` at level 2.
#
# `PRESSURE_SOLVER` below is derived from this rather than set independently, so
# the two cannot drift apart.
# DEFAULT :volume, and this is a pragmatic choice, not a preference.
#
# `:mass` is more accurate where it works - on the sealed-tank acceptance test its
# dp/dt error is -0.013% against `:volume`'s +1.27%, and vapour mass drift is 1670x
# lower. But on THIS mesh no available solver converges on its matrix:
#
#   AMG(mode = Cg())        throws - the mass matrix is not symmetric
#   AMG(mode = AMGSolver()) hits itmax, crawls, and produces T_max = 54.7 K with
#                           T_min pinned on the 25 K floor. Classical AMG
#                           coarsening assumes an M-matrix; this is not one.
#   Bicgstab() + DILU       measured with the G1 monitor over 12 steps at
#                           20 kW/m^2: 36 of 72 pressure solves ran the full 1000
#                           iterations and stopped 4000x short of tolerance, with
#                           the residual (~6e-5) LARGER than |b| (1.55e-5). That
#                           is Bicgstab breaking down, not stalling. ~3 s/step.
#   Gmres() + DILU          ~16 s/step - converges monotonically by construction
#                           but far too slow to use. See `PRESSURE_SOLVER`.
#
# So `:volume` is what runs. It restores a symmetric matrix, which makes AMG's
# Cg mode valid and fast again (~0.3 s/step).
#
# TO REVISIT: the mass form needs a solver that tolerates a non-symmetric,
# badly-conditioned system on a 44:1 aspect-ratio O-grid. GMRES has been tried and
# is too slow (~16 s/step). The remaining candidate is a stronger PRECONDITIONER
# rather than a different Krylov method - `ILU0GPU` on a GPU backend. See the note
# at `PRESSURE_SOLVER`. Until then the accuracy gain is not reachable on this case.
# VERDICT (2026-08-26): :volume. Not a pragmatic compromise any more - :mass is
# unusable on this case for a reason that is not the linear solver.
#
# `alpha` reaches the pressure operator's COEFFICIENT through
#     alpha -> rho_m -> a_P -> rD -> rDf -> Laplacian coefficient -> p -> flux
# so a checkerboard in `alpha` makes the checkerboarded pressure the operator's
# GENUINE SOLUTION, not a mode the discretisation failed to damp. `:mass` adds two
# further alpha->pressure paths, and one - `div(rho_f u*)` - cannot be closed
# because it IS the mass flux. `rD_ref_density` / `mass_mobility_ref` close the
# others and were on throughout; they help without curing it.
#
# Measured on the ladder with `:mass`: every configuration lost the solution as
# near-wall void crossed ~0.70, invariant to the dryout ramp (position AND width),
# both AIAD thresholds, d_bubble over 4x, wall lubrication, the median filter,
# momentum buoyancy, pressure tolerance 1e-4 -> 1e-6, and dt (1e-5 and 5e-6 broke
# at the SAME PHYSICAL TIME). Signature: p_rgh lost its 10x azimuthal smoothing
# advantage over alpha while rho tracked alpha exactly, and alphaCourant/Courant
# went well past its definitional 2.0 - face fluxes alternating in sign, which
# Courant cannot see because it uses the cell-centre velocity.
#
# With `:volume`: full ladder 5e3 -> 1.2e5, fixed dt, smooth fields, closure = 1 at
# every level, 8.0% mean error vs Tatsumoto on the nucleate branch.
#
# The accuracy argument for `:mass` below was measured on a SEALED TANK, where
# mass-conservation drift accumulates. This is a through-flow pipe at fixed outlet
# pressure, so that advantage largely does not apply. Full analysis in
# dev_notes_LH2_pipe_boiling.md, "Mass-form pressure equation".
PRESSURE_FORM = :volume  # :mass is NOT viable here - see above

# AMG is NOT the answer for the mass form, and `mode = AMGSolver()` is not a fix.
# Classical AMG coarsening assumes an M-matrix; the mass form's `rho_f`-scaled
# Laplacian is not one, and the result is not a symmetry error but slow or absent
# convergence. Measured: AMG hits its iteration limit, the run crawls, and level 1
# returns T_max = 54.7 K with T_min pinned on the 25 K solver floor before level 2
# goes NaN. Both AMG modes are therefore ruled out under `:mass`.
#
# MEASURED on this mesh, `pressure_form = :mass`, 20 kW/m^2:
#
#   Bicgstab() + DILU   ~3 s/step.  G1 over 12 steps: 36 of 72 pressure solves ran
#                       the full 1000 iterations and stopped 4000x short, with the
#                       returned residual (~6e-5) LARGER than |b| (1.55e-5) - i.e.
#                       breaking down, not merely stalling.
#   Gmres() + DILU      ~16 s/step. Cannot increase the residual by construction,
#                       but far too slow to be usable here.
#
# Bicgstab is back because 3 s/step at least leaves room to work with. Note this
# is a choice between two solvers that BOTH fail to converge on the mass-form
# matrix - it is not a fix.
#
# WHICH PRECONDITIONER. The short answer is that DILU is already the best CPU
# option available, so the preconditioner is not the remaining lever:
#
#   Jacobi         weakest. dev_notes_LH2_pipe_boiling.md records it stalling 3-4
#                  orders short on this mesh, with 45 of 45 solves at itmax. The
#                  wall cells are 34 um x 1.5 mm x 0.4 mm, ~44:1, and a pressure
#                  Poisson system on that stretching is beyond a diagonal scaling.
#   NormDiagonal   row-norm scaling. A different diagonal, not a structurally
#                  stronger one - no reason to expect it to succeed where Jacobi
#                  fails.
#   DILU           CURRENT. Correct family for this matrix: the factorisation
#                  uses `D[j] -= A[i,j]*A[j,i]/D[i]`, the ASYMMETRIC form (this is
#                  OpenFOAM's DILU, not DIC), so it is valid for a non-symmetric
#                  system rather than merely tolerated by one.
#
#   IC0GPU         incomplete CHOLESKY - symmetric only. Invalid for `:mass`.
#   ILU0GPU        a genuine step up: full incomplete LU rather than a diagonal
#                  approximation to it, and correct for non-symmetric systems.
#                  GPU ONLY (CUDA/AMD/oneAPI extensions); this case runs on CPU().
#
# A CPU `ILU0` exists in `preconditioners_0_types.jl` but is COMMENTED OUT, so it
# is not currently selectable and its state is unknown.
#
# So the three real options, in order of effort:
#   1. `PRESSURE_FORM = :volume` - symmetric matrix, Cg-mode AMG valid, ~0.3 s/step.
#      Costs the mass form's accuracy. This is what currently completes levels.
#   2. Run on a GPU backend and use `ILU0GPU` with Bicgstab. The most likely route
#      to making `:mass` actually work.
#   3. Enable and test the CPU ILU0.
#
# TO TRY: set `PRESSURE_FORM = :mass` above. With `:volume` this branch is unused.
# MEASURED, finally, instead of assumed. The `:mass` pressure matrix is
# GENUINELY non-symmetric - not marginally, and not a scale artefact:
#
#   worst |A[i,j] - A[j,i]|      = 1.40e-8
#   relative to local diagonal   = 0.0954        <- 9.5%
#   diagonal magnitude           ~ 1.3e-7 (max 1.04e-6)
#   structurally missing entries = 0
#
# The asymmetry is `Divergence(pconv, p_rgh)`, the implicit pressure convection,
# which is upwinded and therefore non-symmetric by construction. It scales as
# `psi*|U.Sf|`, and under `:mass` `psi = alpha*psi_v*rho_l` is LINEAR in void, so
# it grows with heat flux.
#
# `Cg()` was passing the old ABSOLUTE 1e-10 test at low flux only because the
# matrix itself is small (diagonal ~1e-7) - not because it was symmetric. So CG
# has been running on a ~10% asymmetric operator and converging by luck, with no
# guarantee, and it finally threw once the void raised `psi` enough.
#
# `AMGSolver()` mode does not require symmetry. `AMGGaussSeidel` because the
# default `AMGJacobi` was measured hitting itmax and crawling to NaN on this
# matrix; Gauss-Seidel is the smoother for a non-symmetric operator and is the
# one option never tried.
#
# FALLBACKS if this is too slow or stalls (both previously measured on `:mass`):
#   Bicgstab() + DILU   diverged, ~3 s/step
#   Gmres()   + DILU    converged but ~16 s/step
PRESSURE_SOLVER = PRESSURE_FORM === :mass ?
    #
    # FORWARD sweep, not the default SYMMETRIC one. A symmetric sweep runs
    # forward AND backward, i.e. twice the work, and it exists so the smoother is
    # a SYMMETRIC OPERATOR - which only matters when AMG preconditions CG. Under
    # `mode = AMGSolver()` this is a standalone V-cycle with no such requirement,
    # so the backward pass buys nothing here and costs 2x.
    # BiCGStab, not the unaccelerated AMGSolver fixed-point iteration.
    #
    # SYNTHETIC BENCHMARK (1D convection-diffusion, n=4000, 3 asymmetry levels):
    #
    #   rel asym   mode           iters   V-cycles   time
    #   0.091      AMG+BiCGStab       7      14       1.63 ms
    #   0.091      AMGSolver         15      15       1.66 ms
    #   0.500      AMG+BiCGStab       5      10       1.11 ms
    #   0.500      AMGSolver         11      11       1.25 ms
    #
    # BiCGStab HALVES the iteration count but costs TWO preconditioner applies per
    # iteration, so on THAT problem it is a wash - and the benchmark was too easy
    # to discriminate: a 2-level hierarchy converging in ~15 cycles is nothing
    # like a 3D pressure system on 25k cells.
    #
    # MEASURED ON THIS CASE: 0.43 s/iteration. For comparison -
    #
    #   AMG + Cg          ~0.3  s/step   INVALID (matrix is 9.5% asymmetric)
    #   AMG + BiCGStab     0.43 s/it     valid, and the one to use
    #   AMG + AMGSolver   ~5x Jacobi     valid, no Krylov acceleration
    #   Bicgstab + DILU   ~3    s/step   valid, weak preconditioner
    #   Gmres + DILU      ~16   s/step   valid, unusable
    #
    # So the acceleration DOES pay on the real matrix, roughly matching the CG
    # mode that was never entitled to run on it. The synthetic benchmark
    # under-predicted, exactly as its own caveat warned.
    #
    # Fall back to `AMGSolver()` if it does not help; both are valid for a
    # non-symmetric matrix, which `Cg()` is not.
    AMG(mode = Bicgstab(),
        smoother = AMGGaussSeidel(sweep = AMGForwardSweep())) :
    AMG(mode = Bicgstab(),
        smoother = AMGGaussSeidel(sweep = AMGForwardSweep()))          # :volume also carries pconv, but small enough that Cg copes

# THE SITE DENSITY AND THE DEPARTURE DIAMETER ARE A PAIR. `q_evap ~ N_a*f*D_d^3`,
# so changing one without refitting the other changes the evaporative term by
# orders of magnitude. Each entry below names the departure model it was fitted
# with; do not mix them.
SITE_DENSITY_FIT = :lh2_t4_mmp

SITE_DENSITY = (
    # Whole-branch fit with h_c = 7045 W/m^2/K taken from the data. RMS(log dT)
    # = 0.0673 over the full nucleate range; 11 of 12100 grid points within 10%
    # of best, so (m, n) are far better determined than in the front below -
    # though still traded off against each other and not physical values.
    # Fitted WITH `KocamustafaogullariIshii`. RMS(log dT) = 0.0766 over the full
    # nucleate branch - essentially the same fit quality as `:lh2` below, because
    # the site density absorbs the departure diameter (they are degenerate against
    # the boiling curve). The curve therefore CANNOT choose between them; the mesh
    # can, and does - see the departure-diameter note further down.
    # --- K-I FRONT. `m` is REFITTED at each `n`; they are not independent. ------
    #
    # RETRACTED - the "measured failure" recorded here was an ARTEFACT of the
    # instrumentation, not a property of the case. It read:
    #
    #   step 540   alpha_max 5.68e-4   T_max 26.434   max|U| 5.494
    #   step 560   alpha_max 4.52e-3   T_max 26.704   max|U| 5.798
    #   step 580   alpha_max 1.000     T_max 71.06    max|U| 9.7e8   <- "gone"
    #
    # and was attributed to an N_a ~ dT_sup^11 feedback loop. That trace was
    # produced by calling `run!` in chunks of 20 steps so the state could be
    # sampled between calls. `multiphase!` allocates `mdot_lagged` (and
    # `mdot_wall_prev`, `mdot_bulk_prev`, `p_abs_prev`, `T_prev`) OUTSIDE its
    # time loop, so every `run!` call resets them. For `Mixture` with
    # `implicit_alpha` the phase-change rate is carried in `mdot_lagged` to be
    # consumed by the alpha equation on the NEXT step - so step 1 of every chunk
    # silently drops it.
    #
    # Proof: with chunks of 1 step the source NEVER reaches alpha and
    # `alpha_max` is identically 0.0 at steps 20/40/60/80/100, against a
    # chunk-of-20 reference of 7.5e-8 ... 2.2e-6.
    #
    # WHAT IS AND IS NOT RETRACTED. The case DOES diverge - the retraction is of
    # the numbers and of the mechanism assigned to them, not of the failure.
    # Measured with ONE continuous `run!` (per-step logging via the
    # `postprocess` hook, which runs every iteration and is NOT gated by
    # `write_interval`), same :ki fit, same q_w = 1e4:
    #
    #   harness          dt      last good        outcome
    #   chunks of 20   1e-5   step 560 / 5.6 ms   DIVERGED step 580 / 5.8 ms
    #   continuous     1e-5   step 700 / 7.0 ms   clean, still growing smoothly
    #   continuous     2e-5   step 530 / 10.6 ms  DIVERGED step 550 / 11.0 ms
    #
    # So the chunked harness failed at 5.8 ms while the continuous run was
    # healthy at 7.0 ms: the "8x in 20 steps" jump and the exact-zero
    # `evap_frac` were harness artefacts. But a continuous run still dies, and
    # the 7.0 ms clean run proves nothing beyond 7.0 ms - it had not yet reached
    # the time at which the dt = 2e-5 run failed.
    #
    # The continuous failure has a different signature from the retracted one:
    # vapour first spreads to ~900 cells and PLATEAUS there for ~100 steps
    # (n>1e-3: 578 at step 390, 912 by 460, still 912 at 500) with `max|U|`
    # tracking the 5.53 bulk to within 1%, and only then runs away:
    #
    #   step 520  alpha_max 2.78e-3  n>1e-3 1740  max|U| 5.590  T resid 3.0e-12
    #   step 530  alpha_max 3.04e-3  n>1e-3 1764  max|U| 5.666  T resid 1.0e-10
    #   step 540  alpha_max 1.71e-2  n>1e-3 3860  max|U| 6.629  T resid 1.7e-8
    #   step 550  alpha_max 1.000    n>1e-3 13849 max|U| 8.2e25 T resid 1.2e-5
    #
    # The ENERGY residual leads by ~20 steps, rising 4 orders while alpha is
    # still at 3e-3 and the velocity field is still healthy. Whether the trigger
    # is physical (a real void-driven transition at ~11 ms) or numerical (set by
    # dt) is UNRESOLVED - it needs a dt = 1e-5 run taken past step 1100 to reach
    # the same physical time. Do not size `n` against any of this until that is
    # settled.
    #
    # So bisect on `n`, as the note above says. The fit barely cares:
    #
    #     n       m        RMS(log dT)   local slope d(ln q)/d(ln dT) at CHF
    #     6.0     27.5651  0.1005        4.08
    #     8.0     10.8083  0.0819        5.43
    #     10.0     6.1759  0.0768        6.78
    #     11.018   5.0024  0.0765        7.46   <- :ki, diverges as above
    #     14.0     3.2369  0.0787        9.48
    #
    # 0.0765 -> 0.1005 in RMS buys a halving of the stiffness. Start at :ki_n6; if
    # that is stable the exponent is confirmed as the driver and :ki_n8/:ki_n10 is
    # the operating compromise. If :ki_n6 still breaks, the exponent is NOT the
    # cause and lowering it further costs accuracy for nothing.
    ki_n6  = LemmertChawla(m = 27.5651, n =  6.0),
    ki_n8  = LemmertChawla(m = 10.8083, n =  8.0),
    ki_n10 = LemmertChawla(m =  6.1759, n = 10.0),
    ki     = LemmertChawla(m =  5.0024, n = 11.018),   # whole-branch best fit
    ki_n14 = LemmertChawla(m =  3.2369, n = 14.0),

    # Fitted WITH `TolubinskyKostanchuk(d_ref = 1.110e-3)`, the Fritz value at the
    # OLD theta = 41.37 deg. RMS(log dT) = 0.0673. Only valid paired with that
    # departure model AND that angle.
    lh2 = LemmertChawla(m = 3.0, n = 7.798),

    # CURRENT. Fitted with `TolubinskyKostanchuk(d_ref = 1.0734e-4)`, i.e. Fritz
    # at the MEASURED theta = 4 deg. RMS(log dT) = 0.0670 - the best of any fit
    # here - with n = 7.798 against :ki's 11.018, so it is also the least stiff.
    #
    # Same `n` as `lh2` above because `n` is set by the SHAPE of the branch and
    # the angle only rescales `D_d`; `m` absorbs that through m ~ D_d^(-2.5/n),
    # which predicts a ratio of 2.12 for the 10.34x change in D_d against 2.06
    # measured (6.176/3.0). That consistency is a check on the fit, not a
    # coincidence.
    lh2_t4 = LemmertChawla(m = 6.176, n = 7.798),

    # PAIRED WITH `partition = :mmp`. Refitted because the partition changes what
    # is being inverted: with q_conv no longer weighted by (1 - A_b) the same wall
    # temperature delivers ~1.8x the flux, so a :kurul_podowski fit run under :mmp
    # under-predicts superheat for a bookkeeping reason, not a physical one.
    #
    #   partition          m       n       RMS(log dT)
    #   :kurul_podowski    6.176   7.798   0.0670
    #   :mmp               3.615   9.945   0.0767
    #
    # The 0D fit mildly PREFERS Kurul-Podowski - but it evaluates a fully wetted
    # wall (alpha_l = 1, K_dry = 0) at every point, so it cannot see the regime
    # :mmp exists to fix, where A_b -> 1 drives q_c to zero and the wall loses its
    # convective path entirely. The accuracy cost is on the branch we can measure;
    # the benefit is in the one we cannot.
    lh2_t4_mmp = LemmertChawla(m = 3.615, n = 9.945),

    # The older Pareto front, fitted with the Petukhov h_c over q > 20 kW/m^2
    # only. Retained for the stiffness bisection described above - `n` is the
    # knob that controls how vertical the branch is, and lowering it remains the
    # diagnostic for whether the exponent is driving an instability.
    n21 = LemmertChawla(m = 1.085, n = 21.17),
    n15 = LemmertChawla(m = 1.362, n = 15.0),
    n12 = LemmertChawla(m = 1.653, n = 12.0),
    n10 = LemmertChawla(m = 2.006, n = 10.0),
    n8  = LemmertChawla(m = 2.691, n =  8.0),
    n6  = LemmertChawla(m = 4.364, n =  6.0),
    n = LemmertChawla()      # shipped WATER fit: m = 210, n = 1.805
)[SITE_DENSITY_FIT]

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
T_table = (19.0, 120.0)

# Saturation from the same EOS rather than an Antoine fit: h_fg falls by more
# than half between 0.4 MPa and the critical point, so a constant latent heat is
# not usable across the paper's pressure sweep.
saturation = build_saturation_curve(H2(), p=p_table, T=(19.0, 120.0), np=201, nT=201)

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

VAPOUR_EOS = :pr

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

#= `IdealGas` carries its own expansivity exactly (`phase_betaT` returns 1, i.e.
beta = 1/T), so it needs no `beta` model; Peng-Robinson supplies one derived
from the same cubic as its density.
`IdealGas` carries beta exactly (`phase_betaT` returns 1, i.e. beta = 1/T) so it
needs none; Peng-Robinson derives one from its own cubic; a constant density
takes the saturated value. =#
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
        # DRIFT BUBBLE DIAMETER - must track the DEPARTURE diameter, and did not.
        #
        # 1.25e-5 was set to match the OLD KocamustafaogullariIshii value at
        # theta = 41.37 deg (12.474 um - they agreed to 3 s.f., which was not a
        # coincidence). Moving the departure model to Fritz at the measured 4 deg
        # took D_d to 107.34 um and left this behind, so the wall was generating
        # 107 um bubbles while the mixture transported 12.5 um ones. `tau_d` goes
        # as d^2, making the drift velocity ~74x too small - vapour was created at
        # the wall far faster than it could be carried away, the near-wall cell
        # filled to alpha = 0.9926 at 3e4, and `wall_boiling_liquid_factor` then
        # cut the source to zero and back as cells crossed void 0.8-0.9. That
        # relaxation cycle is the oscillation, and it caps alpha long before any
        # departure criterion can be reached.
        #
        # MEASURED drift velocity (Manninen + Schiller-Naumann, solved not lagged):
        #
        #   d = 12.5 um  (was)     tau_d = 4.78e-6 s   Ur = 4.31e-5 m/s     1.0x
        #   d = 107.3 um (D_d)     tau_d = 3.52e-4 s   Ur = 2.58e-3 m/s    59.7x
        #   d = 429.4 um (d_max)   tau_d = 5.64e-3 s   Ur = 1.61e-2 m/s   372.6x
        #
        # Note Ur grows far more slowly than tau_d: drag is NOT in the Stokes
        # limit at these sizes (Re_p ~ 28 at d_max), so a 1180x increase in tau_d
        # buys only 373x in drift velocity.
        #
        # PROBE: set to `d_max` (4.2936e-4), the TolubinskyKostanchuk upper bound
        # and 4x the departure diameter - a deliberate OVER-estimate to test
        # whether wall-normal transport is the binding constraint. If the cap
        # clears, it is; if alpha still saturates, transport is not the cause and
        # this should come back down to D_d.
        #
        # The two diameters are the SAME PHYSICAL BUBBLE and should be derived
        # from one source rather than hard-coded independently - that is exactly
        # how they drifted apart here.
        model = Mixture(diameter = 1.0734e-4, alpha_transport = :implicit),
        # AIAD: morphology-blended drag (STAR Eqn 1965) AND the interfacial area
        # density that `ModifiedEnergyJump` below consumes. d_droplet enters the
        # drag as 1/d^2 and the area as 6*alpha_l/d, so it matters twice; 1e-4 is
        # the Weber-limited droplet in a 5.33 m/s vapour stream.
        interfacial_area = AIAD(d_bubble = 1.0734e-4, d_droplet = 1.0e-4,
                                alpha_bubbly = 0.5, alpha_droplet = 0.1,
                                sharpness = 70.0),
        # TURBULENT DISPERSION - the ROUTE matters more than the coefficient.
        #
        # MEASURED at q_w = 3e4 on the Laplacian route, Sc_t = 0.9: the vapour
        # INVENTORY is right - domain-mean void 0.1680 against 0.1685 from thermal
        # equilibrium, 0.3% - but the DISTRIBUTION is not. alpha_max = 1.0 with 23%
        # of cells above 0.3, because radial spreading across the 3 mm pipe takes
        # ~1.02 s against a 69 ms residence: 15x too slow. Buoyancy drift cannot
        # help - gravity is along the tube axis, so that drift is purely AXIAL and
        # has no radial component at all.
        #
        # Sc_t = 0.09 did fix it (smooth, uniform exit void) but is 10x below any
        # physical turbulent Schmidt number - evidence the ROUTE was wrong, not the
        # coefficient.
        #
        # The drift-flux route carries a 1/(alpha_c*alpha_d) denominator the plain
        # Laplacian does not, so it disperses far more strongly where the void is
        # SMALL - the edge of the vapour region, which is where spreading is
        # needed. `:auto` picks the Laplacian here because the transport is
        # implicit; `:drift_flux` forces the other route. EXACTLY ONE is ever
        # active - `Dtf` is left at zero on the drift-flux route so the equation's
        # Laplacian contributes nothing.
        #
        # NOT lift. Tomiyama C_L changes sign at Eo_d ~ 4, i.e. d = 2.58 mm for
        # LH2; these bubbles are 107-429 um, 24x smaller, so C_L is POSITIVE and
        # lift would push vapour TOWARD the wall - the wrong way. Wall lubrication
        # (Antal/Tomiyama/Frank) is the force that pushes off the wall, and is the
        # next thing to add if this is not enough.
        dispersion_Sc = 0.9,              # physical range is 0.7-1.0
        # :laplacian, NOT :drift_flux. The drift-flux route was tried and it
        # CHECKERBOARDS alpha in the wall-normal direction, for a structural
        # reason: it forms `Ur += (D_t/(alpha*(1-alpha)))*grad(alpha)` at CELL
        # CENTRES, interpolates to faces, and the drift flux then multiplies by
        # `alpha_f*(1-alpha_f)`. Those factors CANCEL, so the net face flux is
        # just `-D_t*grad(alpha).Sf` - the SAME physical term the Laplacian
        # carries, but assembled from a cell-centred gradient on a wide stencil
        # that is blind to odd-even oscillation. Same defect as computing a
        # pressure gradient without Rhie-Chow: it can advect a sawtooth but not
        # damp one.
        #
        # So the 1/(alpha_c*alpha_d) denominator is NOT an enhancement - it exists
        # to undo the alpha*(1-alpha) the drift flux applies. Both routes are the
        # same physics at the same Sc_t; only the Laplacian discretises diffusion
        # AS diffusion.
        #
        # CONSEQUENCE: dispersion at a physical Sc_t cannot fix the wall-normal
        # transport deficit by either route - Sc_t = 0.09 only worked by brute
        # force. The missing mechanism is a separate lateral FORCE, and lift is
        # the wrong one (positive C_L at these bubble sizes pushes vapour TOWARD
        # the wall). Wall lubrication is the candidate:
        #
        #   Antal cutoff  ~1.41d = 148 um  = 2.7 cells
        #   Frank range   ~10d   = 1073 um = 19 cells = 36% of pipe radius
        #   U_normal at first cell centre = 0.033 m/s, against 0.0094 m/s needed
        #     to clear the wall cell within its 5.9 ms fill time
        dispersion_route = :laplacian,    # :auto | :laplacian | :drift_flux

        # WALL LUBRICATION - the lateral force the model was missing.
        #
        # Nothing else moves vapour off the wall here. Buoyancy drift is AXIAL
        # (gravity is along the tube), turbulent dispersion at a physical Sc_t is
        # ~15x too slow to cross the 3 mm radius in the 69 ms residence, and lift
        # is the WRONG SIGN at these bubble sizes - Tomiyama C_L changes sign at
        # Eo_d ~ 4, i.e. d ~ 2.58 mm in LH2, and D_d here is 107 um, so lift would
        # push vapour TOWARD the wall.
        #
        # Wall lubrication is orientation-independent (wall normal and
        # wall-PARALLEL relative velocity, no gravity) and points the right way.
        # It also feeds on the axial drift that could not help directly:
        # F ~ |U_r,par|^2, and |U_r,par| IS that axial drift, 0.0412 m/s here.
        #
        # MEASURED coefficients on this mesh (d = 107.3 um, |U_r,par| = 0.0412):
        #
        #   y [um]   y/d    C_w Antal   U_wl Antal    needed
        #    27.8    0.26        4286     0.0331      0.0094   <- 3.5x margin
        #    55.7    0.52        1647     0.0127      0.0094
        #   111.4    1.04       327.7     0.0025
        #   200.0    1.86           0          0               <- cut off
        #
        # where "needed" is the velocity to clear the wall cell within its 5.9 ms
        # fill time at q_w = 3e4.
        #
        # `Antal` is SHORT RANGE (2.7 cells) and that is the intent, not a
        # limitation: the job is to stop the wall cell saturating at alpha = 1,
        # not to flatten the profile - bubbly upflow genuinely IS wall peaked, and
        # the experiment only says the wall does not dry out below CHF.
        #
        # ESCALATION if the peak still reaches too far: `Frank(Cwc = 10.0)`,
        # ~19 cells / 36% of the radius, and 3x stronger at the first cell.
        # OFF for this test - control against the Antal run.
        #   nothing        no lateral wall force (default)
        #   Antal()        short range, ~1.38*d_b cutoff
        #   Antal(Cw2=0.110)  cutoff pulled in to ~1.0*d_b
        #   Frank(Cwc=10.0)   long range, ~10*d_b
        # OFF. Measured at 7e4 (above the 6.4e4 CHF): with Antal ON near-wall
        # void pinned at 0.50 and K_dry never fired; OFF it reached 0.82 and
        # K_dry hit 0.69. A term calibrated to keep void off the wall sub-CHF
        # structurally forbids the vapour blanket that CHF IS. Dispersion was
        # ruled out separately (Sc 0.7->3.0 moved void only 0.496->0.555).
        # NOTE: Antal ON also breaks the LADDER at 1e4 (NaN at level 2) while
        # working in single-flux runs - an unexplained bug, not a physics limit.
        wall_lubrication = Antal(),
        lift = TomiyamaLift(C_max = 0.05),
        p_abs_limit = (0.1e6, 1.2e6),   # [Pa] absolute

        # KEEP THIS ON with `pressure_form = :mass`. It is not optional there -
        # it is what makes the mass form usable at all.
        #
        # Measured, 200 steps, this case:
        #
        #   mass   + rD_ref_density : STABLE
        #   mass   - rD_ref_density : NaN
        #   volume + rD_ref_density : STABLE
        #
        # A plausible-sounding argument says the opposite - that a_P ~ rho*V/dt
        # makes rD ~ dt/(rho*V), so the mass form's `rho_f*rDf` already has the
        # density cancelled and freezing it on top reintroduces `alpha`. That
        # argument is WRONG, as the table above shows; the momentum diagonal does
        # not carry the density the way it assumes. Recorded because it is
        # convincing enough to be worth not re-deriving.
        rD_ref_density = lh2_sat.rho,   # 56.747 kg/m³
        mass_mobility_ref = lh2_sat.rho,

        # PHASE ORDER: VAPOUR FIRST. `alpha` always tracks phase 1 (the solver
        # hardcodes `volume_fraction = 1`), so putting the vapour first makes
        # `alpha` the VOID FRACTION.
        #
        # This is deliberate and it is the standard choice for a dispersed phase
        # (Fluent, STAR-CCM+ and driftFluxFoam all transport the dispersed
        # fraction and infer the continuous one). The solver conserves `alpha` and
        # the mixture mass; whichever phase is NOT tracked is recovered by
        # subtraction, and that subtraction is amplified by roughly
        # `rho_m/rho_tracked` times `(tracked fraction)/(inferred fraction)`.
        # Tracking the liquid at 3% void that factor is ~450, so a 1% flux
        # inconsistency wipes the vapour out entirely - measured on this case, the
        # void collapsed from 0.23 to 0.05 downstream of the plate with no
        # condensation model present. Tracking the vapour it is ~1.
        #
        # `liquid_phase = 2` below tells the solver which phase is physically the
        # liquid, independently of which one `alpha` measures. Everything that
        # genuinely cares - the phase-change sink and its sign, the drift
        # weighting and sign, the continuous/dispersed roles in the slip closure,
        # and every RPI wall-boiling closure - keys off that, not off `alpha`.
        #
        # `PengRobinsonBeta` shares the same `gh2_eos` object as the density, so
        # the expansivity is a derivative of exactly the density field being
        # solved with and the two cannot drift apart.
        phases = (
            Phase(rho  = gh2_eos,           # phase 1 = VAPOUR = tracked by alpha
                  mu   = gh2_sat.mu,
                  k    = gh2_sat.k,
                  cp   = gh2_sat.cp,
                  beta = gh2_beta),
            Phase(rho  = lh2_sat.rho,       # phase 2 = LIQUID
                  mu   = lh2_sat.mu,
                  k    = lh2_sat.k,
                  cp   = lh2_sat.cp,
                  beta = 0.0)
        ),

        # Which phase is the liquid. NOT the same question as which phase `alpha`
        # tracks - see the note above.
        liquid_phase = 2,

        # --- bulk interfacial phase change -----------------------------------
        # `nothing` for now: RPI alone is the configuration that runs.
        #
        # API CHANGED 2026-08-19. `Lee(sigma = ...)` has been REMOVED and now
        # throws. `sigma` was an accommodation coefficient the relaxation
        # parameter was DERIVED from,
        #
        #     beta = sigma*sqrt(1/(2 pi R_sp T_sat))*L*rho_l/(rho_l - rho_v)
        #
        # and the result was then multiplied by the interfacial area density, so
        # the effective coefficient carried a hidden alpha*(1 - alpha) factor and
        # vanished in a nearly pure cell whatever the superheat. It was neither
        # the `r` of the published Lee model nor in its units.
        #
        # Now the coefficient is prescribed directly, in the standard form:
        #
        #     mdot = r*alpha_l*rho_l*(T - T_sat)/T_sat      [kg/m^3/s]
        #
        #   phase_change = Lee(r = 1.0),      # [1/s], same value both branches
        #
        # `r` is volumetric, so there is no area scaling and no kinetic
        # prefactor - it needs no gas constant and works with any EOS. Verified
        # on rung 3.1: the measured relaxation time is +0.46% of analytic.
        #
        # CHOOSING `r`. It is a numerical relaxation rate: large enough to hold
        # the interface near saturation, not so large that the source is stiff.
        # `r = 100` at this case's dt was measured stiff enough to overshoot in
        # the unit-test box; start around 1 and raise it while watching the
        # superheat, and note the verification statement is that the answer must
        # CONVERGE as `r` grows (rung 3.2).
        # ON for the film-boiling ladder. NOT optional once the void gets high:
        # `wall_boiling_liquid_factor` ramps the WALL vapour source to zero as the
        # near-wall liquid runs out (void 0.8 -> 0.9), after which the wall flux
        # enters the vapour as SENSIBLE heat through the unchanged
        # `FixedHeatFlux` condition. Evaporation at the film-liquid interface then
        # becomes this model's job - the wall-boiling source comment says it
        # outright: "a case run with wall boiling as the only phase change source
        # will heat the film without ever consuming the latent heat." Expect that
        # to bite at 6.4e4 and 7e4, not at the lower levels.
        #
        # `r = 1.0` is the value this file already recommended above, and the
        # starting point that note prescribes. IT IS NOT CALIBRATED for this case.
        # Symptom of too LOW: vapour superheating on the film branch (T_max
        # climbing while alpha stalls) - raise it. Symptom of too HIGH: stiffness
        # and overshoot, as measured at r = 100. The verification statement is
        # that the answer must CONVERGE as `r` grows (rung 3.2), which has not
        # been done on this case.
        # TEMPORARILY OFF for the same reason - matches the validated baseline so
        # the calibration refit is the ONLY variable. Needed again above ~0.9 void.
        # REQUIRED for the film branch. Above ~0.9 void
        # `wall_boiling_liquid_factor` zeroes the wall vapour source and the wall
        # flux enters the vapour as SENSIBLE heat; the wall-boiling source note
        # is explicit that "a case run with wall boiling as the only phase change
        # source will heat the film without ever consuming the latent heat".
        # r = 1.0 is this file's own suggested starting point and is NOT
        # calibrated - see the note above.
        # BULK PHASE CHANGE: interfacial heat transfer, NOT Lee relaxation.
        #
        #     Lee: mdot = r*alpha_l*rho_l*(T - T_sat)/T_sat
        #     MEJ: mdot = a_i * h * (T - T_sat)/h_fg
        #
        # Lee weights evaporation by the LIQUID fraction, so in a film-boiling
        # cell the rate goes to zero however superheated the vapour is - the
        # wrong limit for the regime that matters here. Measured on the ladder_B
        # 1e5 field: going from the 0.5-0.9 void band to 0.9-0.99, superheat
        # RISES 1.7x (0.849 -> 1.426 K) while the Lee rate FALLS 2.3x
        # (0.651 -> 0.278 kg/m^3/s), because alpha_l collapses 0.35 -> 0.081.
        # The hottest, most film-like cells produce the LEAST bulk boiling.
        #
        # `ModifiedEnergyJump` has no alpha_l weighting and IS multiplied by the
        # interfacial area (`uses_interfacial_area` is false only for `Lee`), so
        # it is driven by superheat and area - the film-boiling mechanism. With
        # the AIAD droplet area 6*alpha_l/d_droplet ~ 4900 m^2/m^3 at that band,
        # h = 1000 gives ~17.8 kg/m^3/s against Lee's 0.28: ~60x more evaporation
        # WHERE IT MATTERS, while staying comparable at low void where a_i is
        # small - so the validated nucleate branch should be largely unaffected.
        #
        # `h` [W/m^2/K] is the interfacial HTC. Ranz-Marshall
        # Nu = 2 + 0.6 Re^0.5 Pr^0.33 with the conduction floor Nu = 2 gives
        # h = 2*k_v/d = 2*0.02/1e-4 = 400, so 400-2000 is the physical range.
        phase_change = Lee(r=100.0),#ModifiedEnergyJump(h = 1000.0),
        # phase_change = nothing,
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
        # `friction_velocity` selects how u_tau - and hence h_c, and hence the
        # convective share of the partition - is obtained:
        #
        #   :k       u_tau = Cmu^0.25*sqrt(k)   assumes LOCAL EQUILIBRIUM in the
        #                                       near-wall cell (the default)
        #   :loglaw  Newton solve of the log law from the VELOCITY, the same
        #                                       equation the momentum wall
        #                                       treatment uses
        #
        # At equilibrium the two agree exactly. Away from it `:k` is LOW: with
        # near-wall k still at its inlet value here, it under-predicts u_tau by
        # 23%, and `h_c = rho*cp*u_tau/T+` is linear in u_tau, so q_conv is short
        # by the same factor and the partition makes up the difference through
        # evaporation - inflating the wall superheat.
        wall_boiling = RPI(
            patches = (:pipeWall,),

            # --- departure diameter: FRITZ value for LH2, not the water fit ---
            #   d_ref = 0.0208*theta_deg*sqrt(sigma/(g*(rho_l - rho_v)))
            # gives 1110 um at 0.4 MPa against the shipped 600 um water value.
            # It enters q_evap CUBED, so this alone is a ~6x change in
            # evaporation per site. Physically derived rather than fitted, which
            # is why it goes in ahead of the site density.
            # KOCAMUSTAFAOGULLARI-ISHII, not Tolubinsky-Kostanchuk with the Fritz
            # value. RPI deposits a departing bubble's whole volume into the
            # FIRST CELL, so `D_d` relative to the cell height decides whether
            # that is physical. Measured on this O-grid (first cell 0.0557 mm):
            #
            #   TolubinskyKostanchuk(d_ref = Fritz)  D_d = 1.110  mm   D_d/dy = 19.9
            #   KocamustafaogullariIshii             D_d = 0.0125 mm   D_d/dy = 0.22
            #
            # At D_d/dy = 20 the bubble volume lands in a cell one twentieth its
            # size. Measured consequence on the staircase at only 10 kW/m^2:
            # alpha_max = 1.0 (the cell dries out completely), T_max = 47.7 K,
            # evap_frac collapsing to 5e-6, then NaN. Spreading the same vapour
            # over one departure diameter would give alpha ~ 0.03 instead of 0.63.
            #
            # It is also the wrong PHYSICS. Fritz balances buoyancy against
            # surface tension - a POOL boiling correlation. At 5.53 m/s the bubble
            # is sheared off long before buoyancy detaches it. K-I is a force
            # balance carrying the density ratio, and unlike Tolubinsky it
            # responds to pressure (12.5 -> 1.0 um over 0.4 -> 1.1 MPa), which
            # matters because the paper sweeps pressure.
            #
            # SUPERSEDED - the argument above was built on theta = 41.37 deg,
            # which is the K-I library DEFAULT and a WATER value. The MEASURED
            # contact angle for LH2 on this surface is 4 deg, and D_d is LINEAR
            # in it:
            #
            #   theta      Fritz [um]   K-I [um]   Fritz/cell   K-I/cell
            #   41.37        1110.17     12.474        19.93      0.2239
            #    4.00         107.34      1.206         1.93      0.0217
            #
            # Fritz was rejected for giving 1.11 mm ("bubble 20x the cell") and
            # K-I adopted to fix it - but that 1.11 mm was an artefact of the
            # wrong angle. At 4 deg Fritz gives 107 um, i.e. 1.93 cells and
            # squarely in the 0.1-0.5 mm measured for cryogen departure, while
            # K-I gives 1.2 um, BELOW typical surface cavity scale. K-I's
            # density-ratio factor 0.0012*(drho/rho_v)^0.9 is 0.0112 here against
            # 0.92 at water conditions - 89x of work outside its calibration
            # range - so its pressure response is extrapolation, not physics.
            #
            # REFITTED at theta = 4 deg, h_c basis :data, whole nucleate branch:
            #
            #   departure   m        n         RMS(log dT)
            #   K-I         2.116    22.468    0.0946
            #   Fritz/TK    6.176     7.798    0.0670   <- best fit AND gentlest
            #
            # so Fritz/TK wins on accuracy and on stiffness at the same time:
            # n = 7.798 against the 11.018 that was driving the runaway and the
            # near-wall checkerboarding.
            #
            # THE OLD OBJECTION STILL STANDS, and is not resolved by this: Fritz
            # is a POOL boiling balance, and at 5.53 m/s the bubble is sheared
            # off before buoyancy detaches it. Neither model carries shear. The
            # right fix is a shear-based departure correlation; until then this
            # is the better of two imperfect options, chosen because K-I at the
            # correct angle is no longer physically admissible.
            #
            # `TolubinskyKostanchuk` at ZERO subcooling (this case is saturated)
            # returns `d_ref`, so this IS the Fritz diameter at 4 deg.
            #
            # PAIRED with `SITE_DENSITY_FIT = :lh2_t4`. Changing one without the
            # other changes q_evap by orders of magnitude.
            departure_diameter = TolubinskyKostanchuk(d_ref = 1.0734e-4,
                                                      d_max = 4.2936e-4),

            # --- site density: PROVISIONAL, from the HIGH-flux window only ----
            # `calibrate_rpi_highflux.jl` fits 21-59 kW/m^2 to RMS(log dT) = 0.056
            # and is well determined (6 of 4900 grid points within 10%). The
            # LOW-flux window could not be fitted at all - see the caveat block
            # below - so this pair is calibrated where the case actually runs,
            # not across the whole curve.
            #
            # n = 21 is extreme. The measured branch above the knee is nearly
            # VERTICAL (local exponent 15.3), and a power law can only chase that
            # with a large exponent. It is a fitting artefact, not a physical
            # site density - do not quote it as one. It is also stiff, which is
            # part of why the transient wall below matters.
            # SITE_DENSITY_FIT (set at the top of the file) selects from the
            # (n, m) Pareto front below. `m` is REFITTED at each `n` by grid
            # search against the digitised curve - the two are not independent,
            # since N_a = (m*dT_sup)^n means `m` only sets the amplitude through
            # m^n while `n` sets the slope, so changing `n` alone would just move
            # the whole curve.
            site_density = SITE_DENSITY,

            # --- influence area: SMOOTH saturation, not the hard clamp --------
            # The classical form is A_b = min(1, K*N_a*pi*D_d^2/4). With the
            # n = 21.17 site density above, that cap BINDS between 36 and
            # 38 kW/m^2 on this case, and the moment it does,
            #
            #     q_conv = h_c*(T_w - T_l)*(1 - A_b) = 0
            #
            # identically, and stays zero at every higher flux. That removes the
            # only term in the partition that responds smoothly and linearly to
            # the local liquid temperature, leaving the entire wall flux to
            # q_evap ~ dT_sup^21. Measured consequence: the local slope
            # d(ln q)/d(ln dT_sup) jumps from 7.9 to 18.1 across that step, and
            # each wall cell's vapour source becomes an almost vertical function
            # of its own superheat with nothing coupling it to its neighbours.
            #
            # `:exponential` uses A_b = 1 - exp(-x) instead - the Poisson void
            # probability for influence zones placed at random, which is what
            # nucleation cavities actually are. The linear form is the special
            # case that assumes they tile without overlapping.
            #
            # It is not a fitting change: over the measured curve the RMS(log dT)
            # residual moves from 0.31122 to 0.31117 and the predicted superheat
            # at CHF from 1.666 to 1.662 K. What changes is that q_conv decays
            # smoothly to 6% at CHF instead of switching off at 38 kW/m^2, and
            # the slope discontinuity disappears.
            influence_area = DelValleKenning(saturation = :exponential),

            # NOTE: with the hard clamp, everything h_c depends on - the log-law
            # friction velocity below, the T+ branch selection - has NO effect
            # above 38 kW/m^2, because q_conv is zero there. The smooth form is
            # what makes the setting below matter across the whole range.
            friction_velocity = :loglaw,

            # --- transient wall: REQUIRED for anything approaching DNB --------
            # C = rho_w*cp_w*thickness [J/m^2/K]. Non-zero switches the solve
            # from the algebraic inversion to a lumped wall energy balance,
            #
            #     C dT_w/dt = q_gen - [q_conv + q_quench + q_evap]
            #
            # The algebraic form CANNOT represent DNB even in principle: past CHF
            # the boiling curve is non-monotone, so a prescribed flux has up to
            # THREE roots and bisection picks one arbitrarily. A transient wall
            # inverts nothing - it follows the trajectory, and the excursion at
            # departure falls out. It also damps the n = 21 stiffness above.
            #
            # The heater from the paper: SS308, 0.5 mm wall.
            #
            #   C = rho_w*cp_w*t = 8070 * 6.0 * 0.5e-3 = 24.21 J/m^2/K
            #
            # `cp_w = 6 J/kg/K` is the CRYOGENIC value at ~26 K, not the ~500
            # J/kg/K a room-temperature table gives - metal specific heat follows
            # the Debye T^3 law and collapses by roughly two orders of magnitude
            # on the way down. Using a handbook value here would make the wall
            # ~80x more sluggish than it is.
            #
            # Time constant: tau = C/h_total ~ 24.21/14000 ~ 1.7 ms. At
            # dt = 2e-6 s that is ~850 steps per time constant, so the wall
            # transient is well resolved; against a 58 ms flow-through it is
            # fast, so the wall is quasi-steady on the flow timescale while still
            # being able to follow an excursion at departure.
            wall_capacity = 0,# 24.21,

            # WALL DRYOUT - STAR-CCM+ User Guide Eqns (2112)-(2115).
            #
            #   K_dry = 0            alpha_delta <= alpha_dry
            #         = f(beta)      otherwise,  f(beta) = beta^2*(3 - 2*beta)
            #   beta  = (alpha_delta - alpha_dry)/(1 - alpha_dry)
            #
            # so the ramp runs from `dryout_start` to 1.0, NOT to a second free
            # threshold. This smoothstep owns the nucleate-to-dryout transition
            # outright; the film-boiling blend it was shared with is gone.
            #
            # alpha_dry = 0.5. STAR defaults to 0.9 and notes that value "prevents
            # instabilities... due to unintentional dryout"; they also record
            # Weisman & Pei's geometric 0.82 as the figure to use when studying
            # DNB. 0.5 is BELOW both, so the nucleate terms start retreating much
            # earlier than either reference - deliberate here, to see whether an
            # earlier, gentler retreat gives a coherent front rather than the
            # isolated fully-transitioned faces a late narrow ramp produced.
            dryout_start = 0.82,
            # 0.645, NOT STAR's 0.9. On a y+ ~ 60 mesh the first cell is 55.8 um,
            # so the cell-averaged void the solver sees is a dry vapour film
            # DILUTED by the liquid filling the rest of the cell:
            #
            #   alpha_cell ~ t/dy1 + (1 - t/dy1)*alpha_bulk
            #
            # Reaching 0.9 would need t ~ 45 um - 80% of the cell - which is
            # developed film boiling, not DNB onset. 0.645 corresponds to a film
            # of ~16-23 um, which is a sensible inception thickness, and it is
            # where alpha_delta actually sits at the measured CHF of 6.4e4.
            #
            # So this is a MESH-SPECIFIC threshold, not a fitted constant. It
            # should be re-derived, not re-fitted, if the near-wall spacing
            # changes. PROVISIONAL pending a literature film thickness (which is
            # itself heat-flux dependent) - see the validation plan.
            # 0.9 - back to a sane ramp width now that PlayHysteresis below
            # handles the loop gain. 0.645 was narrowed specifically to force
            # K_dry up, and that raised the gain to 1.5/0.145 = 10.3, which
            # clamped the void at 0.62 and made it chatter. The play band opens
            # the loop instead, so the ramp no longer has to fight it.
            dryout_end   = 0.95,
            dryout_snap = 1,

            # BUBBLY LAYER for alpha_delta: one departure diameter.
            #
            # With D_d = 107.3 um and a first cell centre at ~27.9 um, this gives
            # delta/2 - y_c = +25.8 um, so Eqn (2112) extrapolates OUTWARD from
            # the cell centre. Measured profiles rise away from the wall there
            # (peak at 148-220 um), so alpha_delta > alpha_cell and the criterion
            # fires somewhat EARLIER than the raw cell value would.
            #
            # `WallCellLayer()` reduces exactly to the cell value (delta/2 = y_c,
            # no extrapolation) and is the mesh-dependent choice.
            # `YPlusLayer(y_plus)` is independent of both mesh and D_d, which is
            # worth trying given D_d is still the least certain input here.
            # PAIRED WITH dryout_end ABOVE - do not change one without the other.
            # `WallSurface` evaluates the void AT the wall (y = 0) instead of at a
            # layer midpoint. On this profile - which decays monotonically from
            # the wall, alpha' = -894 /m measured - that ADDS ~0.048, where every
            # finite-thickness layer subtracts. STAR's recommended 5.5*D_d would
            # subtract 0.239 and put K_dry at zero for every flux in the ladder.
            bubbly_layer = WallCellLayer(),

            # WALL-TANGENTIAL SMOOTHING of alpha_delta before the dryout ramp.
            #
            # Every other coupling in the wall closure is wall-NORMAL - an
            # independent T_wall bisection per face, a normal extrapolation for
            # alpha_delta, a pointwise K_dry. Nothing ties a face to its
            # neighbours ALONG the heater, and that gap has produced the same
            # symptom three times here: 54% azimuthal scatter in the first-cell
            # void at fixed (z, r), isolated faces sitting at w_film = 1 beside
            # neighbours at 0, and a jagged K_dry front.
            #
            # Note the layer measures do NOT address this: `DiameterLayer` and
            # friends smooth along the WALL NORMAL, which is the direction the
            # jaggedness is not in - which is why widening the layer never
            # helped.
            #
            # Two Laplacian passes at w = 0.5 over the edge-adjacency graph of
            # the patch. That is enough to kill the face-to-face odd-even mode
            # (a single pass leaves ~1/4 of it, two leave ~1/16) while barely
            # touching the along-the-heater variation, whose wavelength is tens
            # of faces. `dryout_smoothing = 0` restores the previous behaviour
            # bit-for-bit.
            # OFF. Measured harm: at the step where the void ran away this
            # suppressed the alpha_delta PEAK by 0.35 (raw 1.00 -> 0.65) while
            # leaving the mean untouched (0.1418 -> 0.1416), which halved K_dry
            # (0.594 -> 0.277) exactly when dryout needed to fire. A localised dry
            # patch IS a peak, and Laplacian smoothing is a peak-killer - the
            # wrong filter for a threshold closure. Harmless in quiet states
            # (suppression 0.002 at step 1500), which is why it looked fine.
            # ON, one MEDIAN pass. Removes isolated face-to-face outliers in
            # alpha_delta while leaving a coherent dryout front untouched.
            #
            # Verified on a 12-face chain: a lone face at 0.95 among 0.2
            # neighbours is returned to 0.20 exactly, a 0.2->0.95 STEP is
            # reproduced exactly, and odd-even scatter on a ramp drops 7.3x with
            # the ramp intact.
            #
            # The `:laplacian` filter is what must NOT be used here: on the same
            # lone-outlier test it cuts the peak to 0.575 AND smears it across
            # three faces, which is how it halved K_dry (0.594 -> 0.277) at the
            # exact step the void ran away. One pass is deliberate - median
            # passes erode genuine features if stacked.
            dryout_smoothing = 3,
            dryout_filter = :median,
            dryout_smoothing_weight = 0.5,   # unused while smoothing is 0

            # PLAY (backlash) HYSTERESIS on the dryout criterion.
            #
            # K_dry closes a negative feedback loop - void up, dryout up,
            # evaporation down, void down - and that ONE loop causes both symptoms
            # seen on this case: it clamps the void, and above a gain it
            # oscillates. Measured: a 0.5-0.9 ramp (gain 3.75) was stable but
            # capped K_dry at 0.48, while 0.5-0.645 (gain 10.3) clamped the void
            # LOWER, at 0.62, and chattered. Sharpening the trigger and reaching
            # it turned out to be the same knob pulling opposite ways.
            #
            # Inside the play band dK_dry/d(alpha) is EXACTLY ZERO, so any
            # oscillation below 2r produces no change in K_dry at all - the loop
            # is OPENED rather than damped - while the ramp keeps its slope. r
            # does not trade against sharpness, which is what separates this from
            # simply widening the ramp.
            #
            # r = 0.05 sits above the measured face-to-face scatter in
            # alpha_delta (0.008-0.077) and well below any plausible physical
            # hysteresis width. Verified offline: loop area 0.154, peak width
            # 0.367 at alpha = 0.70; with `hysteresis = nothing` the loop area is
            # EXACTLY zero, so the default path is unchanged.
            #
            # NOTE alpha_delta and alpha_delta_raw now differ by up to r - one is
            # the play state, the other the instantaneous value. That gap is the
            # operator working, not an error.
            hysteresis = nothing,#PlayHysteresis(r = 0.05),

            # TEMPORAL RELAXATION of alpha_delta - the dryout loop gain limiter.
            #
            # Measured failure this run: the wall balance held every face to
            # within 0.03% of the applied 7e4 W/m^2 for 1500 steps at
            # alpha_delta ~ 0.58, then between t = 0.015 s and 0.020 s the field
            # swept through 0.75 - where dK_dry/d(alpha_delta) PEAKS at 3.0 - and
            # the balance collapsed to a 3%..211% spread with the mass source
            # spanning 90x. Both Courant numbers ran away with it.
            #
            # K_dry closes a negative feedback loop (more void -> more dryout ->
            # less evaporation -> less void), which self-corrects at low gain but
            # oscillates once the gain exceeds 1 with a step of delay. Note the
            # wall barely moved (T_wall 27.80..27.99), so `wall_capacity` does NOT
            # damp this - the oscillation is in the MASS source and bypasses the
            # wall's thermal inertia entirely.
            #
            # r multiplies the high-frequency loop gain, so r < (de-ds)/1.5 = 1/3
            # brings it under unity; 0.2 leaves margin (effective gain 0.6). The
            # filter time constant is dt/r = 50 us, far below the 630 us wall
            # constant and the 58 ms flow-through, so it cannot distort the
            # physics - and it is exactly the identity once converged.
            #
            # The alternative is widening the ramp (gain is 1.5/(de-ds)), but
            # that changes the dryout curve itself; this leaves it alone.
            # OFF (1.0 = no relaxation). Same failure mode as the smoothing:
            # a 5-step low-pass lags a fast-rising alpha_delta, so K_dry
            # under-responds during the runaway it is meant to arrest.
            dryout_relaxation = 0.25,
            start_iteration = 0,

            # WALL HEAT PARTITION - :kurul_podowski (default) or :mmp
            #
            # :mmp replicates STAR-CCM+'s Mixture Multiphase wall boiling,
            # User Guide Eqn (2944):
            #
            #     q_w = q_conv + (q_evap + q_quench)(1 - K_dry)
            #
            # Two changes from Kurul-Podowski:
            #
            #   1. q_conv uses MIXTURE properties and is NOT weighted by (1 - A_b).
            #      STAR: "there [are] convection contributions from vapor and
            #      liquid, always the mixture in contact with the wall".
            #   2. K_dry (= 1 - wall_boiling_liquid_factor) scales q_evap and
            #      q_quench but NOT q_conv, and sits INSIDE the wall-temperature
            #      inversion, so dryout RAISES T_w as STAR describes.
            #
            # WHY: under Kurul-Podowski q_c uses LIQUID properties however dry the
            # wall gets, and A_b -> 1 squeezes it to ZERO - measured at exactly 0.0
            # at dT_sup = 1.52 K with this (m, n), matching q_conv collapsing
            # 4980 -> 1883 across the ladder. The wall then has no valid convective
            # path and the flux is dumped through FixedHeatFlux as sensible heat.
            # Under :mmp convection degrades continuously into vapour convection
            # instead, which is the mechanism the near-wall void cap has been
            # missing.
            partition = :mmp
        ),
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
        # SUPERSEDED 2026-08-19 - the loop is now closed IMPLICITLY, so zeroing
        # these is no longer necessary and is no longer done.
        #
        # The analysis above is right that the loop must be broken and that
        # damping does not do it. It is wrong about the mechanism: the scaling is
        # INVERTED relative to an acoustic CFL. Measured on rung 2.5 (plane
        # Poiseuille, laminar, adiabatic, started from the exact solution):
        #
        #   dt = 1e-5  (0.1x  dx/c)  DIVERGED
        #   dt = 5e-5  (0.7x)        survives, dp/dx +12844% wrong
        #   dt = 2e-4  (2.8x)        stable, +3.97%
        #   dt = 1e-3  (13.9x)       stable, -0.88%
        #
        # Smaller dt is WORSE. So `dt < dx/c` was the wrong diagnosis, and cutting
        # dt to 1e-8 helping on this case was a coincidence of this case.
        #
        # What the loop actually is: the solver recovering the ISENTROPIC
        # compressibility by iterating on the ISOTHERMAL one. Substituting the
        # pressure-work temperature response back into the expansion source and
        # moving it onto the left-hand side gives the exact identity
        #
        #     psi_s = psi_T - beta^2*T/(rho*cp)
        #
        # which for an ideal gas turns 1/p into 1/(gamma*p) = 1/(rho*c^2) - the
        # coefficient that carries the acoustic wave speed. `thermo_acoustic =
        # :implicit` (the default) supplies it directly instead of iterating
        # towards it, and the `expansion` source drops exactly the increment that
        # `S_T` produced.
        #
        # Measured after the fix, same case: dp/dx to -0.06% of exact, stable at
        # every dt tried, and time-step independent to 0.039% across a 32x range.
        # Disabling a leg instead gave -4.17%, so the implicit coupling is ~50x
        # more accurate than switching the term off - it corrects the physics
        # rather than removing it. The sealed-tank dp/dt acceptance test also
        # IMPROVED, from -1.02% to -0.013% under `pressure_form = :mass`.
        #
        # To reproduce the old behaviour for comparison:
        #     thermo_acoustic = :explicit,  pressure_work_relax = 0.0,
        #     expansion_relax = 0.0,

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
        pressure_form = PRESSURE_FORM,

        # dp/dt FILTER TIME CONSTANT [s]. This is what makes the case survive
        # past ~11 ms at q_w = 1e4; without it the run dies and, worse, dies
        # SOONER the finer dt gets.
        #
        # `pressure_work_relax` alone is a per-step blend, so its filter time
        # constant is ~dt/r - it smooths over one or two STEPS whatever dt is.
        # The quantity it filters, (p - p_prev)/dt, carries a noise floor that
        # grows as 1/dt. So refining dt strengthens the noise and weakens the
        # filter at the same time. MEASURED here, one continuous `run!`:
        #
        #   dt = 2e-5, per-step r = 0.5     DIVERGED at 11.0 ms
        #   dt = 1e-5, per-step r = 0.5     DIVERGED at  7.5 ms  <- half dt, sooner
        #   dt = 1e-5, pressure work OFF    clean to 12.0 ms
        #
        # With `pressure_work_tau` the factor becomes dt/(tau + dt), a first-
        # order low-pass of FIXED time constant, so damping stops depending on
        # dt. Both time steps then survive AND agree - at 12.0 ms, dt = 1e-5 vs
        # 2e-5 gives max|U| 5.623/5.620, alpha_max 4.2547e-3/4.2183e-3 (0.86%),
        # and an IDENTICAL 2616 cells above alpha = 1e-3. That is gate G3.
        #
        # It is a BLEND, not a scale: at steady state field == prev, so this
        # bounds how fast Dp/Dt may move and leaves its converged value alone.
        # `pressure_work_relax = 0.0` would instead delete the term, which costs
        # -4.17% on dp/dx (see `multiphase_pressure_work_relax`).
        #
        # VALUE: the acoustic transit time of the domain. 7e-4 s is the figure
        # quoted in `multiphase_pressure_work_relax`; L_total/c for this domain
        # (0.370 m, c ~ 1100 m/s in LH2) gives 3.4e-4 s. Both are the same order
        # and the results above were taken at 7e-4. NOT yet swept - the filter
        # does not bias the converged answer, so this is a stability knob rather
        # than a calibration, but it has not been shown insensitive either.
        pressure_work_tau = 7.0e-4,

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
            Dirichlet(:inlet, 0.0),          # pure liquid entering = zero void
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
        # NOT a bare `AMG()`: see `PRESSURE_SOLVER`. With `pressure_form = :mass`
        # the matrix is not symmetric and AMG's default Cg mode rejects it.
        solver=PRESSURE_SOLVER, preconditioner=DILU(),
        convergence=1e-7, relax=0.9, rtol=1e-4, atol=1e-14, itmax=1000),
    alpha = SolverSetup(
        solver=Bicgstab(), preconditioner=DILU(),
        convergence=1e-7, relax=0.9, rtol=1e-4, atol=1e-10),
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
# MEASURED 2026-08-19. dt matters, but the HEAT FLUX matters far more, and an
# earlier version of this note had the attribution backwards.
#
# Cold start, 60 steps at dt = 2e-5, wall boiling on, sweeping only q_w:
#
#   q_w = 10 kW/m^2   Courant 0.29     AlphaCourant 0      p_rgh 3.5e-11   HEALTHY
#   q_w = 20 kW/m^2   Courant 11.8     AlphaCourant 4.19   p_rgh 1.0e-9    marginal
#   q_w = 66 kW/m^2   Courant 6.9e31   AlphaCourant 1.6e51 p_rgh 1.8e5     DIVERGED
#
# Three regimes, and they are about physics rather than numerics:
#
#   * BELOW vapour generation the case is rock solid - Courant 0.29 and a
#     pressure residual at 1e-11 is a converged, well-behaved run.
#   * ONCE VAPOUR APPEARS the ALPHA Courant number goes over 1 (4.19 at 20 kW/m^2)
#     while the pressure residual stays at 1e-9. That is the explicit MULES alpha
#     update over its own stability limit, not a pressure problem, and the remedy
#     is a smaller dt *once boiling starts* - or `AdaptiveTimeStepping(maxAlphaCo
#     = ...)`, which is what it is for. Note the curve driver does NOT currently
#     use it.
#   * AT 66 kW/m^2 nothing helps, and it should not: `WALL_HEAT_FLUX = 6.6e4` is
#     deliberately ABOVE the measured CHF of 64 kW/m^2 (see its own comment), and
#     `FILM_BOILING = false` leaves RPI with no departure criterion. The header
#     says as much - RPI "will keep predicting nucleate boiling at any flux it is
#     given". Starting cold at a post-CHF flux asks the model for a state it
#     cannot represent.
#
# WHY THE BOILING CURVE SCRIPT WORKS AND THIS ONE DOES NOT. It RAMPS - 10, 20, 30,
# 40, 50, 66 kW/m^2, letting each level settle before the next - so the wall
# approaches CHF along the boiling curve instead of being dropped past it from a
# cold start. That is the right way to drive this case, and it is why
# `3d_LH2_pipe_boiling_curve.jl` completes its levels while a single-point run at
# the same final flux does not.
#
# For a single-point run: use a flux comfortably below CHF, or set
# `FILM_BOILING = true` (which also needs `wall_capacity > 0`), or ramp.
#
# Still open regardless of flux: the `T >= T_sat` assertion at the end of this
# file trips even at 10 kW/m^2 - the near-wall cooling recorded in
# dev_notes_LH2_pipe_boiling.md is NOT fixed by the thermo-acoustic or
# calibration work.
dt = 2e-5

# THREE flow-throughs, not one, and the reason is specific to a VOID-driven
# transition. The criterion fires on near-wall vapour fraction, and that field
# does not exist until the flow has carried vapour the length of the plate - the
# wall boiling dumps measured earlier showed the plate average settling at about
# ONE flow-through. So departure cannot even be assessed until then, and the
# excursion itself needs room after it: the wall time constant is ~2 ms and the
# DNB traverse ~10-20 ms, against a 69 ms flow-through.
#
# One flow-through would show the void still developing and no departure, which
# is indistinguishable from a criterion that never fires.
n_flow_throughs = 1

# 20 D of UNHEATED pipe, not 10: `dev_length_factor = 10` upstream AND
# `exit_length_factor = 10` downstream. This was still counting the inlet
# development only, which made `iterations` 16% short of the requested flow-
# throughs - see `make_lh2_pipe_sector.jl`.
L_total = L_heated + 20*D
iterations = round(Int, n_flow_throughs*(L_total/U_inlet_mag)/dt)

# runtime = Runtime(
#     iterations = iterations,
#     time_step = dt,
#     write_interval = 100,#round(Int, iterations/50),
#     adaptive = AdaptiveTimeStepping(maxCo=0.5, maxAlphaCo=0.25)
# )

runtime = Runtime(
    iterations = iterations,
    time_step = dt,
    write_interval = 100,#round(Int, iterations/50)
)

config = Configuration(
    solvers=solvers, schemes=schemes,
    runtime=runtime, hardware=hardware, boundaries=BCs)

initialise!(model.momentum.U, velocity)
initialise!(model.fluid.p_rgh, 0.0)
initialise!(model.energy.T, T_inlet)
initialise!(model.fluid.alpha, 0.0)          # pipe starts full of liquid, so ZERO
                                             # void: alpha now tracks the VAPOUR
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, omega_inlet)
initialise!(model.turbulence.nut, nut_inlet)

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
# `3d_LH2_pipe_boiling_curve.jl` includes this file purely for the setup - mesh,
# fluid, models, schemes, solvers - and then drives its own heat-flux staircase.
# It sets this flag first so everything above is built and nothing below runs.
if @isdefined(BOILING_CURVE_SETUP_ONLY)
    @info "Setup only: skipping the single-point run (BOILING_CURVE_SETUP_ONLY is set)"
else

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
# cannot remain single phase. `alpha` is the VOID fraction, so this is a maximum
# above zero rather than a minimum below one.
@test maximum(model.fluid.alpha.values) > 0.0

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

end  # BOILING_CURVE_SETUP_ONLY
