#  ADIABATIC AIR-WATER BUBBLY UPFLOW - interfacial force validation
#  ============================================================================
#
#  WHAT THIS CASE IS FOR
#
#  The radial void distribution in a vertical bubbly pipe is set by exactly
#  three forces: lift (toward the wall for small bubbles), wall lubrication
#  (away from it, short range), and turbulent dispersion (down the gradient).
#  Nothing else. There is no heat, no phase change and no wall boiling anywhere
#  in this case.
#
#  That isolation is the point. On the LH2 pipe a near-wall void ceiling was
#  attributed in turn to turbulent dispersion, to mesh resolution, to the dryout
#  band and to the K_dry feedback loop before it turned out to be a missing
#  force. Every one of those hypotheses is eliminated by a single run of this
#  case, because none of those mechanisms exists here.
#
#  WHAT IT VALIDATES, IN TWO TIERS
#
#  Tier A needs NO experimental data:
#
#    A1  Single phase (alpha_in = 0) must recover the log law and a friction
#        factor matching Blasius, f = 0.316*Re^-0.25.
#    A2  Sign check. Run D_BUBBLE = 3 mm and 7 mm with everything else
#        identical: the first must be WALL peaked, the second CORE peaked.
#        This is VERIFICATION of the lift implementation against Tomiyama's
#        correlation (reversal at Eo_d = 6.06, a ~5.9 mm bubble in air-water),
#        NOT a comparison with Liu & Bankoff - their bubbles are 2-4 mm, so
#        every one of their 42 conditions is on the wall-peaking branch. See
#        `test/unit_test_lift_sign.jl` for the coefficient-level test.
#    A3  Area-averaged void must satisfy the drift-flux relation
#        <alpha> = j_g/(C0*j + u_gj) with C0 ~ 1.2 (Zuber-Findlay).
#
#  Tier B needs digitised profiles from the source papers:
#
#    B1  Liu & Bankoff (1993) Part II, doi 10.1016/S0017-9310(05)80290-X.
#        D = 38.1 mm, measuring station L/D = 36, dual-sensor resistivity probe.
#        Fig. 3 = radial void fraction; Fig. 12 = local mean bubble diameter.
#        42 conditions: J_f 0.376-1.391 m/s, J_g 0.027-0.347 m/s, local void to
#        50%, mean bubble size 2-4 mm.
#
#        WHAT IT TESTS. Every condition is wall peaked with a flat core for
#        r/R <= 0.8, so the target is PEAK SHAPE - height, radial position, and
#        the sharpness trend the paper reports: "wall peaking became more
#        pronounced for high liquid flows... at lower liquid flow rates, a more
#        uniform void distribution with a relatively lower peak". Reproducing
#        that trend across J_f is a stronger test than matching any one profile.
#
#        Fig. 3 and Fig. 12 do NOT share conditions, so d_b must be interpolated
#        from Fig. 12 (2.2 mm at J_g = 0.027 rising to ~4.0 mm at J_g = 0.347).
#        Lift scales as d^2, so that interpolation is a stated uncertainty, not
#        a detail.
#    B2  Serizawa, Kataoka & Michiyoshi (1975), Int. J. Multiphase Flow 2:221.
#        The canonical wall-peaked profile.
#    B3  MTLOOP / TOPFLOW wire-mesh data (Lucas, Krepper & Prasser, FZD). The
#        dataset most CFD codes validate this force set against.
#
#  Digitised data is NOT included here and must not be invented - drop a CSV of
#  (r/R, alpha) into `data/` and point `REFERENCE_DATA` at it.
#
#  MESH
#      cd examples/0_GRIDS/bubbly_pipe_sector
#      julia --project=. make_bubbly_pipe_sector.jl && ./run_blockMesh.sh
#
#  ============================================================================

using XCALibre
using Printf
using Statistics: mean
using StaticArrays: SVector

# -----------------------------------------------------------------------------
# Case selection
# -----------------------------------------------------------------------------
# Local mean bubble diameter, interpolated from Liu & Bankoff Fig. 12 for the
# condition below. For the A2 verification sweep use 3.0e-3 and 7.0e-3 - 3 mm
# sits at Eo_d = 1.36 (C_L = +0.288, wall peaked), 7 mm at Eo_d = 9.30
# (C_L = -0.246, core peaked).
# Estimated per condition from Fig. 12(a1) (J_f = 0.535), shifted slightly down
# because higher liquid flow gives smaller bubbles - Fig. 12 has no J_f = 0.753
# series, so these ARE interpolations and lift scales as d^2. Refine if you
# digitise Fig. 12; report the comparison at the 2.2/4.0 mm bracket either way.
const D_B_FIG12 = Dict(0.027 => 2.1e-3, 0.112 => 2.7e-3,
                       0.230 => 3.2e-3, 0.347 => 3.6e-3)

# Superficial velocities, from the Fig. 3(a1) series (constant J_f, four gas
# flows). The mixture model carries ONE velocity, so the inlet is the total
# volumetric flux j = j_l + j_g and the inlet void is j_g/j.
#
#   Fig. 3(a1)  J_f = 0.753,  J_g = 0.027 / 0.112 / 0.230 / 0.347
#   Fig. 3(b1)  J_g = 0.180,  J_f = 0.376 / 0.535 / 0.753 / 1.391
#
# J_f = 0.753 also keeps y+ near 35 on the mesh as generated (sized for
# U_bulk = 1.0), so no regeneration is needed for this series.
J_LIQUID = 0.753         # [m/s]
J_GAS    = 0.112         # [m/s]  sweep: 0.027 / 0.112 / 0.230 / 0.347

D_BUBBLE = get(D_B_FIG12, J_GAS, 3.0e-3)     # [m]

# Digitised Fig. 3(a1), one file per gas flow. Set to `nothing` for Tier A only.
REFERENCE_DATA = joinpath(@__DIR__, "data", "liu_bankoff_1993",
                          "Jf0753_Jg" * lpad(round(Int, J_GAS*1000), 4, '0') * ".csv")

# -----------------------------------------------------------------------------
# Fluid: air and water at 20 C, 1 atm
# -----------------------------------------------------------------------------
RHO_W, MU_W = 998.2, 1.002e-3
RHO_A, MU_A = 1.204,  1.825e-5
SIGMA       = 0.0728             # [N/m]

D_PIPE   = 38.1e-3
J_TOTAL  = J_LIQUID + J_GAS
ALPHA_IN = J_GAS/J_TOTAL

Re   = RHO_W*J_TOTAL*D_PIPE/MU_W
Tu   = 0.05
k_in = 1.5*(Tu*J_TOTAL)^2
w_in = sqrt(k_in)/(0.07*D_PIPE*0.09^0.25)
nut_in = k_in/w_in

# Friction velocity from Petukhov, the same correlation the mesh generator uses
# to size the first cell. Needed for the developed initial condition below.
f_D    = (0.790*log(Re) - 1.64)^-2
U_TAU  = J_TOTAL*sqrt(f_D/8)
K_EQ   = U_TAU^2/sqrt(0.09)        # equilibrium k in the log layer
R_PIPE = 0.5*D_PIPE
# 1/7 power law: <U>/U_max = 2n^2/((n+1)(2n+1)) = 98/120 for n = 7.
U_MAX  = J_TOTAL/(98/120)

# -----------------------------------------------------------------------------
#  Developed pipe turbulence, as a function of wall distance
# -----------------------------------------------------------------------------
#
#  WHY THE INLET CANNOT BE UNIFORM. Core turbulence in a pipe is not produced
#  locally - the centreline has almost no shear - it is TRANSPORTED there from
#  the wall. That transport is diffusive and slow, so a uniform inlet cannot
#  build a developed core over any length this domain has:
#
#      diffusion time wall -> core = L^2/(2*nu_t) = (15 mm)^2/(2*3e-6) = 37 s
#      residence time to z/D = 36                                      = 1.6 s
#
#  MEASURED with a uniform `k_in`, core values at the measurement plane:
#
#                        k           omega
#      measured       3.49e-05        3.30
#      homogeneous
#      decay theory   3.63e-04        7.05
#      developed pipe 2.16e-03       ~35
#
#  Even PERFECT homogeneous decay lands 6x below the developed value, so this is
#  not a closure deficiency - a uniform inlet simply cannot get there. The
#  consequence is a turbulent dispersion deficit of the same order: spreading the
#  wall-generated void over half the radius in the residence time needs
#  D_t = 28*nu and the run delivers 3.2*nu, which is why the computed profile is
#  flat across the core and spikes at the wall instead of rising gradually from
#  r/R ~ 0.3 as Liu & Bankoff measure.
#
#  The experiment has a long upstream development section, so the liquid arrives
#  fully developed. Prescribing that is more faithful than a uniform inlet, not
#  less.
#
#  FORM. `k/u_tau^2` runs from 1/sqrt(Cmu) = 3.33 in the log layer to ~1 at the
#  centreline; `nu_t` is the capped Nikuradse mixing length used by the initial
#  condition below, so the two are consistent; `omega = k/nu_t` then follows.
#  At the first cell this gives omega = 2466 against the equilibrium
#  u_tau/(sqrt(Cmu)*kappa*y) = 2487, i.e. self-consistent to 1%.
#
#  NOTE the VELOCITY inlet stays a uniform plug, as in Lubchenko et al. The
#  velocity profile develops perfectly well over 40 D - measured wall/core
#  Uz = 0.746 against 0.63-0.73 for a turbulent pipe - it is only the turbulence
#  that cannot.
k_dev(yw)   = U_TAU^2*(1/sqrt(0.09)*(1 - yw/R_PIPE) + 1.0*(yw/R_PIPE))
nut_dev(yw) = U_TAU*min(0.41*yw, 0.09*R_PIPE)
om_dev(yw)  = k_dev(yw)/nut_dev(yw)

# Wall distance from a face/cell centre. Floored so `omega` stays finite if a
# face centre ever lands exactly on the wall.
wall_dist(x, y) = clamp(R_PIPE - hypot(x, y), 1.0e-5, R_PIPE)

# DEVELOPED VELOCITY at the inlet, not a plug.
#
#  A uniform inlet convects through and erases any profile seeded in the
#  interior - measured here, the 1/7 law written by `initialise!` was gone
#  within 250 iterations. What survives is a plug with the whole boundary layer
#  crushed into r/R > 0.85:
#
#      z/D = 36    r/R    Uz       dUz/dr    P/eps   tau/tau_w
#                  0.125  0.9015    -0.01     0.00     0.000
#                  0.625  0.9012    -0.08     0.00     0.000
#                  0.925  0.8436  -171.26     1.33     0.785
#
#  `Uz` varies by 0.08% across 80% of the radius. A developed pipe MUST satisfy
#  tau/tau_w = r/R - that is a force balance on a cylindrical control volume,
#  not a turbulence model - and this run gives 0.000 out to r/R = 0.8.
#
#  The consequence runs all the way to the answer: no shear, so P/eps = 0 across
#  the core, so `k` decays homogeneously from the inlet, so `nu_t` falls from a
#  prescribed 79*nu to 10-30*nu, so the turbulent dispersion is a third of the
#  28*nu needed to spread the wall void inward - and `alpha` stays flat at 0.074
#  across the core with a spike at the wall, instead of the gradual rise from
#  r/R ~ 0.3 that Liu & Bankoff measure.
#
#  Lubchenko et al. use a uniform inlet over 42 D and it develops for them,
#  because k-epsilon with sigma_k = 1.0 transports momentum and `k` inward
#  roughly twice as fast. Measured here it does not, so the profile is
#  prescribed instead. That is also the more faithful choice: the experimental
#  facility has a long upstream section, so the liquid arrives developed.
#
#  U_MAX = J_TOTAL/(98/120) makes the ANALYTIC area-mean exactly J_TOTAL; the
#  discrete mean will differ slightly, so check `j_g + j_l` against 0.865 in the
#  run output rather than assuming it.
U_inlet(coords, t, i) = begin
    yw = clamp(R_PIPE - hypot(coords[1], coords[2]), 0.0, R_PIPE)
    SVector(0.0, 0.0, U_MAX*(yw/R_PIPE)^(1/7))
end

k_inlet(coords, t, i)   = k_dev(wall_dist(coords[1], coords[2]))
om_inlet(coords, t, i)  = om_dev(wall_dist(coords[1], coords[2]))
nut_inlet(coords, t, i) = nut_dev(wall_dist(coords[1], coords[2]))

# Eotvos numbers, so the log records which side of the sign change this run is on.
Eo   = 9.81*(RHO_W - RHO_A)*D_BUBBLE^2/SIGMA
d_H  = D_BUBBLE*cbrt(1 + 0.163*Eo^0.757)
Eo_d = 9.81*(RHO_W - RHO_A)*d_H^2/SIGMA

@info """Adiabatic air-water bubbly upflow
    D          = $(D_PIPE*1e3) mm
    j_l, j_g   = $J_LIQUID, $J_GAS m/s   ->  j = $J_TOTAL, alpha_in = $(round(ALPHA_IN, digits=4))
    Re         = $(round(Int, Re))
    d_bubble   = $(D_BUBBLE*1e3) mm
    Eo_d       = $(round(Eo_d, digits=3))  ->  expect $(Eo_d < 6.06 ? "WALL" : "CORE") peaked void"""

# -----------------------------------------------------------------------------
# Mesh
# -----------------------------------------------------------------------------
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "bubbly_pipe_sector", "constant", "polyMesh")
isdir(mesh_file) || error("""
Mesh not found at $mesh_file

Generate it first:
    cd examples/0_GRIDS/bubbly_pipe_sector
    julia --project=. make_bubbly_pipe_sector.jl && ./run_blockMesh.sh
""")
mesh = FOAM3D_mesh(mesh_file, scale=1.0, integer_type=Int64, float_type=Float64)

# -----------------------------------------------------------------------------
#  Normalise the inlet profile to the DISCRETE mesh
# -----------------------------------------------------------------------------
#  `U_MAX = J_TOTAL/(98/120)` makes the ANALYTIC area-mean of the 1/7 law exactly
#  J_TOTAL, but the discrete sum over inlet faces does not agree: 0.86755 against
#  0.865 on this mesh, a 0.29% excess. That is small, but it is a systematic bias
#  in the delivered j_l and j_g of a case whose whole purpose is quantitative
#  comparison, and it costs nothing to remove.
#
#  Rescaling from the actual face areas rather than hard-coding a factor means
#  this re-derives itself if the mesh is regenerated - which it has been once
#  already, when the wall layer was refined from y+ = 40 to y+ = 8.
let rng = first(b.IDs_range for b in mesh.boundaries if b.name == :inlet)
    A = 0.0; Q = 0.0
    for f in rng
        fc = mesh.faces[f].centre
        a  = mesh.faces[f].area
        yw = clamp(R_PIPE - hypot(fc[1], fc[2]), 0.0, R_PIPE)
        A += a
        Q += U_MAX*(yw/R_PIPE)^(1/7)*a
    end
    global U_MAX *= J_TOTAL/(Q/A)
    @printf("Inlet 1/7 profile normalised on %d faces: <Uz> -> %.5f m/s, U_max = %.5f
",
            length(rng), J_TOTAL, U_MAX)
end

backend = CPU(); hardware = Hardware(backend=backend, workgroup=AutoTune())
mesh_dev = adapt(backend, mesh)

velocity = [0.0, 0.0, J_TOTAL]        # upward, tube axis is +z
noSlip   = [0.0, 0.0, 0.0]

# -----------------------------------------------------------------------------
# Physics
# -----------------------------------------------------------------------------
model = Physics(
    time = Transient(),
    fluid = Fluid{Multiphase}(
        model = Mixture(diameter = D_BUBBLE, alpha_transport = :implicit),

        # LATERAL FORCES: the "L corrected, TD, no WL" combination of Lubchenko
        # et al. (2018), IJMF 98:36-44, Fig. 1 - a constant C_L0 = 0.025 damped
        # to zero within half a bubble diameter of the wall, turbulent
        # dispersion, and NO wall lubrication force.
        #
        # WHY NOT Antal. Lubchenko's Fig. 1 catalogues the four combinations and
        # this case reproduced two of them before the damping existed:
        #
        #   lift + TD + Antal   "strong repulsion pushes all gas out of the
        #                        first few computational cells and results in an
        #                        unphysical peak 1-3 bubble diameters from the
        #                        wall"
        #     MEASURED here, r/R = 0.833/0.917/1.000:
        #         alpha = 0.105  0.013  0.245     <- depleted cell, peak moved out
        #
        #   lift + TD           "significant over-prediction of void fraction at
        #                        the wall due to dominance of velocity gradients
        #                        in near-wall region"
        #     MEASURED here with `wall_lubrication = nothing`: core flat, wall
        #     peak unopposed.
        #
        # Neither is a discretisation defect. Both are the documented behaviour
        # of these closures, and the damping is the published remedy.
        #
        # C_L0 = 0.025 is Lubchenko's value, citing Baglietto & Christon (2013);
        # it is an ORDER OF MAGNITUDE below Tomiyama's small-bubble ceiling of
        # 0.288. `ConstantLift` has no low-slip cutoff - if that proves twitchy,
        # `TomiyamaLift(C_max = 0.025)` gives the same ceiling while keeping the
        # `tanh(0.121*Re_p)` roll-off.
        #
        # WALL LUBRICATION is Lubchenko's own Eq. 26 - the force that exactly
        # cancels turbulent dispersion when grad(alpha) takes the shape a layer
        # of spherical bubbles touching the wall must have. It recovers the void
        # PEAK that lift damping alone flattens, and it carries no tunable
        # coefficients and no dependence on grad(alpha), which matters here
        # because the odd-even mode this case has been fighting lives in exactly
        # that gradient.
        #
        # DISPERSION is the Burns et al. (2004) Favre-averaged drag form. In a
        # mixture model that reduces to the Fickian term times 1/(1 - alpha) -
        # C_D and |U_r| cancel against drag - so it is 11% stronger at the peak
        # void of this case and a factor of 50 stronger in a filled wall cell.
        # See `fad_factor`.
        #
        # Sc_t = 1.0, not 0.9: sigma_TD is "usually taken as 1" in Lubchenko
        # Eq. 7, and this is a validation case against that paper.
        lift             = ShaverPodowski(inner = ConstantLift(C_L = 0.025)),
        # wall_lubrication = LubchenkoWL(),
        dispersion_Sc    = 1.0,
        dispersion_route = :laplacian,

        sigma = SIGMA,

        # phase 1 is tracked by alpha; here that is the dispersed GAS.
        phases = (
            Phase(rho = RHO_A, mu = MU_A),     # phase 1 = AIR = tracked
            Phase(rho = RHO_W, mu = MU_W),     # phase 2 = WATER
        ),
        liquid_phase = 2,

        # Reference density for the `p_rgh` split. Stated explicitly even though
        # the library now defaults to the liquid phase, because this is a
        # validation case and the choice is load bearing: without it the
        # buoyancy source becomes `-g.h grad(rho)`, which at this measurement
        # plane is 4.9e5 N/m^3 against gravity's 8.8e3 - 56x - and grows with
        # height, so a developed inlet profile is a plug by z/D = 10. See
        # `multiphase_rho_ref`.
        rho_ref = RHO_W,
        gravity = Gravity([0.0, 0.0, -9.81]),
    ),
    turbulence = RANS{KOmegaSST}(walls = (:pipeWall, :wallUnheated)),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
)

BCs = assign(
    region = mesh_dev,
    (
        U = [
            DirichletFunction(:inlet, U_inlet),
            Zerogradient(:outlet),
            Wall(:pipeWall, noSlip),
            Wall(:wallUnheated, noSlip),
            Symmetry(:symmetryX), Symmetry(:symmetryY),
        ],
        p_rgh = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Zerogradient(:pipeWall), Zerogradient(:wallUnheated),
            Symmetry(:symmetryX), Symmetry(:symmetryY),
        ],
        alpha = [
            Dirichlet(:inlet, ALPHA_IN),
            Zerogradient(:outlet),
            Zerogradient(:pipeWall), Zerogradient(:wallUnheated),
            Symmetry(:symmetryX), Symmetry(:symmetryY),
        ],
        k = [
            DirichletFunction(:inlet, k_inlet), Zerogradient(:outlet),
            KWallFunction(:pipeWall), KWallFunction(:wallUnheated),
            Symmetry(:symmetryX), Symmetry(:symmetryY),
        ],
        omega = [
            DirichletFunction(:inlet, om_inlet), Zerogradient(:outlet),
            OmegaWallFunction(:pipeWall), OmegaWallFunction(:wallUnheated),
            Symmetry(:symmetryX), Symmetry(:symmetryY),
        ],
        nut = [
            DirichletFunction(:inlet, nut_inlet), Zerogradient(:outlet),
            NutWallFunction(:pipeWall), NutWallFunction(:wallUnheated),
            Symmetry(:symmetryX), Symmetry(:symmetryY),
        ],
    )
)

schemes = (
    U     = Schemes(time=Euler, divergence=Upwind, laplacian=Linear, gradient=Gauss),
    # `p` is read by the pressure equation as `schemes.p.laplacian` even though
    # the field solved is `p_rgh`. Both entries are required.
    p     = Schemes(time=Euler, divergence=Upwind, gradient=Gauss, laplacian=Linear),
    p_rgh = Schemes(time=Euler, divergence=Upwind, gradient=Gauss, laplacian=Linear),
    alpha = Schemes(time=Euler, divergence=Upwind, laplacian=Linear, gradient=Gauss),
    k     = Schemes(time=Euler, divergence=Upwind, laplacian=Linear, gradient=Gauss),
    omega = Schemes(time=Euler, divergence=Upwind, laplacian=Linear, gradient=Gauss),
    # Wall distance, for the SST blending functions.
    y     = Schemes(gradient=Midpoint),
)

solvers = (
    U = SolverSetup(solver=Bicgstab(), preconditioner=DILU(),
                    convergence=1e-7, relax=1.0, rtol=1e-3, atol=1e-10),
    p_rgh = SolverSetup(solver=AMG(mode = Bicgstab(), smoother = AMGGaussSeidel(sweep = AMGForwardSweep())), preconditioner=DILU(),
                    convergence=1e-7, relax=0.9, rtol=1e-4, atol=1e-10),
    alpha = SolverSetup(solver=Bicgstab(), preconditioner=DILU(),
                    convergence=1e-7, relax=0.9, rtol=1e-4, atol=1e-10),
    k = SolverSetup(solver=Bicgstab(), preconditioner=DILU(),
                    convergence=1e-7, relax=1.0, rtol=1e-3, atol=1e-10),
    omega = SolverSetup(solver=Bicgstab(), preconditioner=DILU(),
                    convergence=1e-7, relax=1.0, rtol=1e-3, atol=1e-10),
    # Wall distance. `KOmegaSST` needs it, and without this entry the run fails
    # inside `wall_distance!` after the solve has started rather than at
    # configuration time - so it is worth having even though nothing above
    # references `y` directly.
    y = SolverSetup(solver=Cg(), preconditioner=Jacobi(),
                    convergence=1e-8, relax=1, rtol=1e-2),
)

# 2.1 m of pipe at j = 1.1 m/s is a 1.9 s flow-through; four of them settles the
# radial profile, which is the only quantity being measured.
DT, ITERS = 2e-3, 2500
runtime = Runtime(iterations=ITERS, time_step=DT, write_interval=250)
config  = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
                        hardware=hardware, boundaries=BCs)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.fluid.p_rgh, 0.0)
initialise!(model.fluid.alpha, ALPHA_IN)
initialise!(model.turbulence.k, K_EQ)
initialise!(model.turbulence.omega, w_in)
initialise!(model.turbulence.nut, nut_in)
# Overwritten per-cell below with the same developed profile the inlet imposes,
# so the interior starts where the boundary is feeding it rather than relaxing
# toward it across a domain too short for that to happen.

# -----------------------------------------------------------------------------
#  DEVELOPED initial condition, not a plug
# -----------------------------------------------------------------------------
#
#  Starting from a uniform velocity field means there is no shear anywhere
#  except the wall face at t = 0, so `k` has almost no production while its
#  dissipation runs at full rate. Every wall function then branches on a
#  K-BASED wall distance,
#
#      y+ = Cmu^0.25*y*sqrt(k)/nu      (RANS_functions.jl)
#
#  compared against `yPlusLam = 11.53`. Once `k` falls far enough that this
#  drops below the threshold, all three wall functions switch to their VISCOUS
#  branch and near-wall production shuts off - and the flow can no longer
#  recover, because it now has no eddy viscosity with which to build the
#  velocity profile that would generate the shear it needs.
#
#  MEASURED before this change, z/D = 20, wall cell against the second cell:
#
#      run                    nut          Uz wall / 2nd
#      lift on              4.06e-09      0.862 / 0.864     <- FLAT, no boundary layer
#      lift off             3.62e-07      0.682 / 0.823     <- a real profile
#
#  with `nut/nu = 0.00` against the 14.8 that y+ = 36 calls for, and `k` five
#  orders below the equilibrium 7.55e-3. That matters far beyond the turbulence
#  model, because BOTH lateral forces that oppose lift scale with `nut`:
#  turbulent dispersion is `nu_t/Sc_t`, and Lubchenko Eq. 26 carries the same
#  factor. With `nut = 0` lift acts alone, which is Lubchenko et al. Fig. 1's
#  black curve - all gas driven to the wall, none of it dispersing - and any
#  radial structure lift creates prints straight onto `alpha` with nothing to
#  smooth it.
#
#  Seeding a 1/7 power law removes the race: real shear, and therefore `k`
#  production and `nut`, exist from the first iteration, so the wall functions
#  start and stay on the log branch. This changes only the PATH to steady state,
#  not the answer - the inlet stays a uniform plug, as in Lubchenko et al., so
#  the 40 D development section still does its job.
let U = model.momentum.U, k = model.turbulence.k,
    om = model.turbulence.omega, nut = model.turbulence.nut
    for (i, c) in enumerate(mesh.cells)
        r  = hypot(c.centre[1], c.centre[2])
        yw = clamp(R_PIPE - r, 1e-6, R_PIPE)

        U.x.values[i] = 0.0
        U.y.values[i] = 0.0
        U.z.values[i] = U_MAX*(yw/R_PIPE)^(1/7)

        # Nikuradse-style mixing length: kappa*y near the wall, capped at
        # 0.09*R in the core. The parabolic form kappa*y*(1 - y/R) was tried
        # first and is wrong at the centreline, where it collapses to zero and
        # sends omega to 2000 - the cap keeps nut/nu ~ 80 there instead of 3.6.
        nu_t = nut_dev(yw)
        k.values[i]   = k_dev(yw)
        nut.values[i] = nu_t
        om.values[i]  = om_dev(yw)
    end
end

@printf("""
Developed initial condition
    u_tau      = %.5f m/s      (Petukhov, f = %.5f)
    U_max      = %.4f m/s      1/7 power law
    k_eq       = %.4e m2/s2    -> y+(k) at the first cell = %.1f
    nut(wall)  = %.3e m2/s     -> nut/nu = %.1f
""", U_TAU, f_D, U_MAX, K_EQ,
     0.09^0.25*0.759e-3*sqrt(K_EQ)/(MU_W/RHO_W),
     0.41*U_TAU*0.759e-3*(1 - 0.759e-3/R_PIPE),
     0.41*U_TAU*0.759e-3*(1 - 0.759e-3/R_PIPE)/(MU_W/RHO_W))

residuals = run!(model, config, inner_loops=5)

# =============================================================================
#  Radial void profile at the measurement plane
# =============================================================================
#
#  Sampled 50 D downstream of the inlet, at the end of the measurement section.
#  The sector has symmetry planes, so the azimuthal spread at fixed radius is
#  numerical error by construction - it is reported alongside the profile
#  because it bounds how much of any disagreement with data is the solution
#  rather than the model.

"""
    radial_void_profile(model, z_sample; nbins=20) -> (r_over_R, alpha, spread)

Bin cell-centred void by radius at one axial station.
"""
function radial_void_profile(model, z_sample; nbins = 20, dz = 0.02)
    mesh = model.domain
    cells = Array(mesh.cells)
    a = Array(model.fluid.alpha.values)

    r = Float64[]; av = Float64[]
    for (i, c) in enumerate(cells)
        z = c.centre[3]
        abs(z - z_sample) < dz || continue
        push!(r, hypot(c.centre[1], c.centre[2]))
        push!(av, a[i])
    end
    isempty(r) && error("no cells found near z = $z_sample")

    R = maximum(r)
    edges = range(0.0, R, length = nbins + 1)
    rc = Float64[]; ac = Float64[]; sp = Float64[]
    for j in 1:nbins
        m = (r .>= edges[j]) .& (r .< edges[j+1])
        count(m) == 0 && continue
        push!(rc, 0.5*(edges[j] + edges[j+1])/R)
        push!(ac, sum(av[m])/count(m))
        push!(sp, maximum(av[m]) - minimum(av[m]))   # azimuthal error at this r
    end
    return rc, ac, sp
end

# Liu & Bankoff measured at L/D = 36, so sample there rather than at the end of
# the mesh. The mesh names the section boundaries `pipeWall`/`wallUnheated` but
# neither is heated, so the station is free to sit anywhere developed.
z_sample = 36.0*D_PIPE

rc, ac, sp = radial_void_profile(model, z_sample)

# Wall-peaked or core-peaked? Compare the outer 20% of the radius against the
# inner 40% - the discriminator A2 turns on.
outer = ac[rc .> 0.8]
inner = ac[rc .< 0.4]
peak = mean(outer) > mean(inner) ? "WALL peaked" : "CORE peaked"
expected = Eo_d < 6.06 ? "WALL peaked" : "CORE peaked"

println()
@printf("Radial void profile at z = %.3f m (%.0f D)\n", z_sample, z_sample/D_PIPE)
@printf("%8s %10s %12s\n", "r/R", "alpha", "azim.spread")
for (x, y, s) in zip(rc, ac, sp)
    @printf("%8.3f %10.5f %12.2e\n", x, y, s)
end
println()
@printf("  inner (r/R<0.4) mean : %.5f\n", mean(inner))
@printf("  outer (r/R>0.8) mean : %.5f\n", mean(outer))
@printf("  Eo_d = %.3f  ->  %s;  measured: %s  %s\n",
        Eo_d, expected, peak, peak == expected ? "OK" : "MISMATCH")

# Area-averaged void, for the Zuber-Findlay check (A3).
area_avg = 2*sum(rc .* ac)/sum(2 .* rc)
@printf("  area-averaged alpha  : %.5f   (inlet %.5f)\n", area_avg, ALPHA_IN)

# =============================================================================
#  Comparison against Liu & Bankoff
# =============================================================================
#
#  The measurement spans r/R = -1..+1, so mirrored pairs bound the combined
#  digitising and measurement scatter. Across the four gas flows that floor is
#  4-6% of the peak value, and it is the tolerance any agreement claim has to be
#  judged against - matching more closely than the data repeats is not a better
#  result, it is an over-fitted one.

"""
    read_reference(path) -> (r_over_R, alpha)

Signed radius, so both sides of the pipe are returned as measured.
"""
function read_reference(path)
    r = Float64[]; a = Float64[]
    for line in eachline(path)
        ln = strip(line)
        (isempty(ln) || startswith(ln, "#") || startswith(ln, "r_over_R")) && continue
        parts = split(ln, ',')
        push!(r, parse(Float64, parts[1])); push!(a, parse(Float64, parts[2]))
    end
    return r, a
end

"""Scatter between mirrored points: the noise floor of the reference itself."""
function mirror_scatter(r, a)
    d = Float64[]
    for i in eachindex(r)
        r[i] < 0 || continue
        j = argmin(abs.(r .- abs(r[i])))
        r[j] > 0 && abs(r[j] - abs(r[i])) < 0.05 && push!(d, abs(a[i] - a[j]))
    end
    return isempty(d) ? NaN : sum(d)/length(d)
end

if REFERENCE_DATA !== nothing && isfile(REFERENCE_DATA)
    rr, ar = read_reference(REFERENCE_DATA)
    floor_ = mirror_scatter(rr, ar)

    # Fold the reference and interpolate the model onto its radii.
    ra = abs.(rr); o = sortperm(ra); ra, af = ra[o], ar[o]
    model_at(x) = begin
        k = searchsortedfirst(rc, x)
        k <= 1        && return ac[1]
        k > length(rc) && return ac[end]
        t = (x - rc[k-1])/(rc[k] - rc[k-1])
        (1 - t)*ac[k-1] + t*ac[k]
    end
    resid = [model_at(x) - y for (x, y) in zip(ra, af)]
    rms = sqrt(sum(abs2, resid)/length(resid))

    ipk_d = argmax(af); ipk_m = argmax(ac)
    println()
    println("Comparison with Liu & Bankoff (1993) Fig. 3(a1)")
    @printf("  reference          : %s
", basename(REFERENCE_DATA))
    @printf("  peak alpha   data %.4f at r/R %.3f  |  model %.4f at r/R %.3f
",
            af[ipk_d], ra[ipk_d], ac[ipk_m], rc[ipk_m])
    @printf("  core alpha   data %.4f              |  model %.4f
",
            sum(af[ra .< 0.5])/count(ra .< 0.5), mean(inner))
    @printf("  peak/core    data %.2f                 |  model %.2f
",
            af[ipk_d]/(sum(af[ra .< 0.5])/count(ra .< 0.5)), mean(outer)/mean(inner))
    @printf("  RMS residual       : %.4f
", rms)
    @printf("  reference scatter  : %.4f  (mirrored pairs)
", floor_)
    @printf("  --> %s
", rms <= floor_ ? "within the data's own scatter" :
            "exceeds scatter by " * string(round(rms/floor_, digits=2)) * "x")
else
    @info """No reference data found at $REFERENCE_DATA (Tier A only).
    Digitised Liu & Bankoff Fig. 3(a1) profiles live in data/liu_bankoff_1993/."""
end


# =============================================================================
#  Figure
# =============================================================================
#
#  `Plots` is loaded HERE and not in the package, for the same reason the boiling
#  curve case does it: a solver library should not force a plotting stack on
#  everyone who loads it.
#
#  Both mirrored halves of the measurement are drawn as separate markers rather
#  than averaged. The spread between them IS the reference's own scatter, and
#  showing it keeps the eye honest about how close agreement can meaningfully be.
using Plots

let
    plt = plot(;
        xlabel = "r/R", ylabel = "void fraction  α  [-]",
        title = "Air-water bubbly upflow, J_f = $(J_LIQUID) m/s, J_g = $(J_GAS) m/s",
        legend = :topleft, framestyle = :box, minorgrid = true,
        size = (760, 560), dpi = 200, xlims = (0, 1))

    if REFERENCE_DATA !== nothing && isfile(REFERENCE_DATA)
        rr, ar = read_reference(REFERENCE_DATA)
        neg = rr .< 0
        scatter!(plt, abs.(rr[neg]), ar[neg];
            marker = (:circle, 5), color = :black, markerstrokewidth = 0,
            label = "Liu & Bankoff 1993, Fig. 3(a1)  (r < 0)")
        scatter!(plt, rr[.!neg], ar[.!neg];
            marker = (:utriangle, 5), color = :black, markeralpha = 0.45,
            markerstrokewidth = 0, label = "same, mirrored half  (r > 0)")
    end

    plot!(plt, rc, ac;
        marker = (:diamond, 5), color = :crimson, linewidth = 2,
        label = "XCALibre  (d_b = $(round(D_BUBBLE*1e3, digits=2)) mm)")

    # The peak position is the sharp target: invariant at r/R ~ 0.89-0.90 across
    # the whole gas-flow range, so a misplaced peak indicts the lift/lubrication
    # balance rather than any single coefficient.
    vline!(plt, [0.895]; linestyle = :dash, color = :grey,
           label = "measured peak, r/R ≈ 0.895")

    figpath = joinpath(@__DIR__, "data",
                       "bubbly_pipe_Jg" * lpad(round(Int, J_GAS*1000), 4, '0') * ".png")
    savefig(plt, figpath)
    @info "radial void profile figure written" figpath
    display(plt)
end
