# =============================================================================
#  Single-phase validation of the pipe: wall treatment and friction
# =============================================================================
#
#  Same mesh, same liquid, same turbulence model as the boiling case - but NO
#  multiphase, NO phase change, NO boiling. Just developed turbulent pipe flow.
#
#  WHY THIS EXISTS
#
#  The RPI convective flux is `q_conv = h_c*(T_w - T_l)*(1 - A_b)` with
#
#      h_c = rho*cp*u_tau/T+        and        u_tau = Cmu^0.25 * sqrt(k)
#
#  so `h_c` is directly proportional to the friction velocity the wall function
#  produces, and `u_tau` is entirely a function of the near-wall `k`. A measured
#  `h_conv` of 3692-5388 W/m^2/K against a Dittus-Boelter estimate of ~10,200
#  suggested `u_tau` might be 2-3x low - which would starve `q_conv` and force
#  the partition to make up the difference through evaporation, inflating the
#  wall superheat.
#
#  That hypothesis is about the TURBULENCE MODEL, not the boiling model, so it
#  should be tested without any boiling in the way.
#
#  TWO INDEPENDENT KNOWN ANSWERS
#
#    1. u_tau = U*sqrt(f/8)          with f from Petukhov
#    2. dp/dx = f*rho*U^2/(2*D)      the same f, measured a different way
#
#  Both come from the same correlation, but they are measured from different
#  fields (k versus pressure), so agreement is a real check rather than a
#  tautology. Petukhov is good to ~5% for 1e4 < Re < 5e6.
#
#  WHAT A FAILURE MEANS
#
#  If `u_tau` from `k` is low while the pressure drop is right, the wall
#  functions are producing too little near-wall `k` and every `h_c` in the
#  boiling model inherits it. If BOTH are low, the flow is not developed - extend
#  the inlet section or run longer.
# =============================================================================

using XCALibre
using Printf

# -----------------------------------------------------------------------------
# Conditions - matched to the boiling case
# -----------------------------------------------------------------------------
const CASE = :D6_L250
const p_sat = 0.4e6
const U_bulk = 5.53
const D = 6.0e-3
const L_heated = 250.0e-3
const L_dev = 10*D
const L_total = L_dev + L_heated

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "lh2_pipe_sector", "constant", "polyMesh")
mesh = FOAM3D_mesh(mesh_file, scale=1.0, integer_type=Int64, float_type=Float64)

backend = CPU(); workgroup = AutoTune(); activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Saturated-liquid properties from the same Helmholtz EOS the boiling case uses,
# so this is the identical fluid rather than an approximation of it.
saturation = build_saturation_curve(H2(), p=(0.25e6, 1.25e6), T=(19.0, 33.0),
                                    np=201, nT=201, verbose=false)
const T_sat = saturation_temperature(saturation, p_sat)
const LH2 = phase_properties_at(H2(), p_sat, T_sat, branch=:liquid)

const rho_l = LH2.rho
const mu_l  = LH2.mu
const nu_l  = mu_l/rho_l
const Re    = rho_l*U_bulk*D/mu_l

# -----------------------------------------------------------------------------
# The reference answers
# -----------------------------------------------------------------------------
"""Petukhov (1970) friction factor, valid 1e4 < Re < 5e6."""
petukhov(Re) = (0.790*log(Re) - 1.64)^-2

const f_darcy   = petukhov(Re)
const u_tau_ref = U_bulk*sqrt(f_darcy/8)
const tau_w_ref = rho_l*u_tau_ref^2
const dpdx_ref  = f_darcy*rho_l*U_bulk^2/(2*D)

@info """Single-phase reference conditions
    fluid         : saturated LH2 at $(p_sat/1e6) MPa, T_sat = $(round(T_sat, digits=3)) K
    rho, mu       : $(round(rho_l, digits=3)) kg/m^3, $(round(mu_l*1e6, digits=4)) uPa.s
    U_bulk, D     : $U_bulk m/s, $(D*1e3) mm
    Re            : $(round(Int, Re))
    Petukhov f    : $(round(f_darcy, digits=6))
    ---- EXPECTED ----
    u_tau         : $(round(u_tau_ref, digits=5)) m/s
    tau_wall      : $(round(tau_w_ref, digits=4)) Pa
    dp/dx         : $(round(dpdx_ref, digits=2)) Pa/m   ($(round(dpdx_ref*L_total, digits=2)) Pa over the pipe)"""

# -----------------------------------------------------------------------------
# Physics: incompressible, k-omega SST. No energy equation - this is a purely
# hydrodynamic check, and adding heat transfer would only introduce a second
# thing to be wrong.
# -----------------------------------------------------------------------------
velocity = [0.0, 0.0, U_bulk]
noSlip = [0.0, 0.0, 0.0]

const Tu = 0.05
const k_inlet = 1.5*(Tu*U_bulk)^2
const omega_inlet = sqrt(k_inlet)/(0.07*D*0.09^0.25)
const nut_inlet = k_inlet/omega_inlet

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu_l),
    turbulence = RANS{KOmegaSST}(walls=(:pipeWall, :wallUnheated)),
    energy = Energy{Isothermal}(),
    domain = mesh_dev)

BCs = assign(region = mesh_dev, (
    U = [Dirichlet(:inlet, velocity), Zerogradient(:outlet),
         Wall(:pipeWall, noSlip), Wall(:wallUnheated, noSlip),
         Symmetry(:symmetryX), Symmetry(:symmetryY)],
    p = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0),
         Wall(:pipeWall), Wall(:wallUnheated),
         Symmetry(:symmetryX), Symmetry(:symmetryY)],
    k = [Dirichlet(:inlet, k_inlet), Zerogradient(:outlet),
         KWallFunction(:pipeWall), KWallFunction(:wallUnheated),
         Symmetry(:symmetryX), Symmetry(:symmetryY)],
    omega = [Dirichlet(:inlet, omega_inlet), Zerogradient(:outlet),
             OmegaWallFunction(:pipeWall), OmegaWallFunction(:wallUnheated),
             Symmetry(:symmetryX), Symmetry(:symmetryY)],
    nut = [Dirichlet(:inlet, nut_inlet), Zerogradient(:outlet),
           NutWallFunction(:pipeWall), NutWallFunction(:wallUnheated),
           Symmetry(:symmetryX), Symmetry(:symmetryY)],
))

schemes = (
    U     = Schemes(divergence=Upwind, gradient=Gauss, laplacian=Linear),
    p     = Schemes(divergence=Upwind, gradient=Gauss, laplacian=Linear),
    k     = Schemes(divergence=Upwind, gradient=Gauss, laplacian=Linear),
    omega = Schemes(divergence=Upwind, gradient=Gauss, laplacian=Linear),
    y     = Schemes(gradient=Midpoint),
)

solvers = (
    U = SolverSetup(solver=Bicgstab(), preconditioner=DILU(),
                    convergence=1e-8, relax=0.7, rtol=1e-3, atol=1e-12),
    p = SolverSetup(solver=Cg(), preconditioner=DILU(),
                    convergence=1e-8, relax=0.3, rtol=1e-3, atol=1e-12, itmax=1000),
    k = SolverSetup(solver=Bicgstab(), preconditioner=DILU(),
                    convergence=1e-8, relax=0.6, rtol=1e-3, atol=1e-12),
    omega = SolverSetup(solver=Bicgstab(), preconditioner=DILU(),
                        convergence=1e-8, relax=0.6, rtol=1e-3, atol=1e-12),
    y = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-8, relax=0.9, rtol=1e-2),
)

runtime = Runtime(iterations=2000, time_step=1, write_interval=500)
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
                       hardware=hardware, boundaries=BCs)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, omega_inlet)
initialise!(model.turbulence.nut, nut_inlet)

residuals = run!(model, config)

# =============================================================================
# Validation
# =============================================================================
bnds = get_boundaries(mesh_dev.boundaries)
wall_idx = findfirst(b -> b.name === :pipeWall, bnds)
wall_faces = collect(bnds[wall_idx].IDs_range)
wall_cells = [mesh_dev.boundary_cellsID[f] for f in wall_faces]

# --- 1. u_tau from the wall function, exactly as the RPI model computes it ----
# `Cmu^0.25 * sqrt(k)` is the same expression `wall_friction_velocity!` uses, so
# this measures the quantity the boiling model actually consumes rather than a
# proxy for it.
k_wall = [model.turbulence.k[c] for c in wall_cells]
u_tau_sim = 0.09^0.25 .* sqrt.(max.(k_wall, 0.0))
areas = [mesh_dev.faces[f].area for f in wall_faces]
A_wall = sum(areas)
u_tau_avg = sum(u_tau_sim .* areas)/A_wall

# --- 2. Pressure gradient over the heated section -----------------------------
# Measured away from the inlet development region and the outlet BC, so it is a
# fully developed value in both cases.
cell_z(c) = mesh_dev.cells[c].centre[3]
z_lo, z_hi = L_dev + 0.25*L_heated, L_dev + 0.75*L_heated
band(zl, zh) = [i for i in eachindex(mesh_dev.cells) if zl <= cell_z(i) <= zh]
lo_cells = band(z_lo - 0.02*L_heated, z_lo + 0.02*L_heated)
hi_cells = band(z_hi - 0.02*L_heated, z_hi + 0.02*L_heated)

p_lo = sum(model.momentum.p[i] for i in lo_cells)/length(lo_cells)
p_hi = sum(model.momentum.p[i] for i in hi_cells)/length(hi_cells)
dpdx_sim = (p_lo - p_hi)/(z_hi - z_lo)          # positive: pressure falls downstream

# XCALibre's incompressible solver works in KINEMATIC pressure (p/rho), so the
# measured gradient must be scaled by rho before comparing with a Pa/m reference.
dpdx_sim_Pa = dpdx_sim*rho_l

# --- Report -------------------------------------------------------------------
@printf("\n%-22s %-14s %-14s %-10s\n", "quantity", "simulated", "expected", "ratio")
println("-"^64)
@printf("%-22s %-14.5f %-14.5f %-10.3f\n", "u_tau [m/s]", u_tau_avg, u_tau_ref, u_tau_avg/u_tau_ref)
@printf("%-22s %-14.4f %-14.4f %-10.3f\n", "tau_wall [Pa]",
        rho_l*u_tau_avg^2, tau_w_ref, (rho_l*u_tau_avg^2)/tau_w_ref)
@printf("%-22s %-14.2f %-14.2f %-10.3f\n", "dp/dx [Pa/m]", dpdx_sim_Pa, dpdx_ref, dpdx_sim_Pa/dpdx_ref)

# h_c that the boiling model would build from this u_tau, against Dittus-Boelter.
# The RPI convective flux is proportional to this, so the ratio here is the
# ratio q_conv would be wrong by.
const Pr_l = mu_l*LH2.cp/LH2.k
const Nu_db = 0.023*Re^0.8*Pr_l^0.4
const h_db = Nu_db*LH2.k/D
@printf("\nPr = %.4f,  Dittus-Boelter Nu = %.1f  ->  h = %.0f W/m^2/K\n", Pr_l, Nu_db, h_db)
@printf("y+ and T+ come from the mesh and the wall function; compare `h_conv` in\n")
@printf("the boiling case's surface file against %.0f W/m^2/K.\n", h_db)

println("""

INTERPRETATION
  u_tau ratio ~ 1        wall function is fine; a low h_c must come from T+
  u_tau ratio ~ 0.3-0.5  near-wall k is too small - this is the h_c deficit, and
                         it is a turbulence-model problem, not a boiling one
  BOTH ratios low        flow not developed: lengthen L_dev or run longer
  dp/dx right, u_tau low k is wrong while the momentum balance is right, which
                         points at the k wall function specifically
""")
