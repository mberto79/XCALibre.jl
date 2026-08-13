# =============================================================================
#  blockMeshDict generator - 90 degree O-grid sector of a vertical heated pipe
# =============================================================================
#
#  Emits system/blockMeshDict (and a minimal system/controlDict) for a quarter
#  sector of the vertically-mounted heated tubes of:
#
#      Tatsumoto, Shirai, Shiotsu, Hata, Naruo, Kobayasi & Inatani,
#      "Forced convection heat transfer of saturated liquid hydrogen in
#       vertically-mounted heated pipes",
#      AIP Conf. Proc. 1573, 44-51 (2014).  doi:10.1063/1.4860681
#
#  Run with Julia, then run blockMesh in this directory; XCALibre loads the
#  result with FOAM3D_mesh.
#
#      julia make_lh2_pipe_sector.jl
#      ./run_blockMesh.sh
#
#  ---------------------------------------------------------------------------
#  Geometry (paper, "Experimental apparatus and method")
#  ---------------------------------------------------------------------------
#  SS316 tube heaters, wall thickness 0.5 mm, inner diameter D and heated length
#  L, giving the two L/D ratios the paper is built around:
#
#      D = 4 mm : L = 100 mm (L/D = 25.0),  L = 167 mm (L/D = 41.7)
#      D = 6 mm : L = 150 mm (L/D = 25.0),  L = 250 mm (L/D = 41.7)
#
#  Liquid hydrogen flows UPWARD through the tube, so the axis is +z and gravity
#  is -z. An unheated development length is included upstream of the heated
#  section: the paper states the entrance lengths are more than 10 x D, and the
#  flow is turbulent throughout (Re > 1e4), so the profile entering the heated
#  section is developed. Modelling it explicitly is more faithful than imposing
#  a developed profile at the inlet of the heated section.
#
#  ---------------------------------------------------------------------------
#  Topology - butterfly (O-grid) quarter
#  ---------------------------------------------------------------------------
#  A quarter of the standard 5-block butterfly: one core block plus the two
#  outer blocks that reach the wall. This removes the axis singularity entirely
#  (there is no cell edge on the centreline), which is what makes an O-grid the
#  right choice here rather than the wedge used for the K-Site tank - a wedge
#  collapses onto the axis and leaves the azimuthal direction rank-deficient.
#
#      y
#      ^
#      |  v6 . _
#      |  |     ` .  v5
#      |  v3-----v2  \        NORTH = (v3 v2 v5 v6)
#      |  |      |    |       EAST  = (v1 v4 v5 v2)
#      |  |CORE  |    |       CORE  = (v0 v1 v2 v3)
#      |  v0-----v1---v4
#      +----------------------> x
#
#  The two cut planes (x = 0 and y = 0) are `symmetry` patches. A 90 degree
#  sector is valid because the case is axisymmetric in the mean; it cannot
#  represent azimuthal asymmetry, which for a uniformly heated vertical tube in
#  upward flow is a reasonable modelling choice and is stated as such in the
#  case file.
#
#  ---------------------------------------------------------------------------
#  Near-wall resolution - the point of this script
#  ---------------------------------------------------------------------------
#  The case is meshed for a HIGH y+ (wall function) treatment, target 30-50.
#  That is a deliberate requirement rather than a convenience: the RPI wall
#  boiling model gets its single-phase heat transfer coefficient from the
#  thermal law of the wall, whose logarithmic branch is only valid once the
#  first cell centre is clear of the buffer layer (y+ > ~30). Resolving to
#  y+ ~ 1 would put the first cell inside the viscous sublayer, where the wall
#  function is not merely inaccurate but the wrong formula.
#
#  The first cell height is therefore SOLVED for, from the target y+ and a
#  Petukhov friction estimate, rather than guessed; the achieved y+ range is
#  reported at the end. Because an O-grid outer block has a radial thickness
#  that varies along its length (largest on the axis, smallest at the 45 degree
#  diagonal), the wall cell height varies too, and the reported range shows by
#  how much.
# =============================================================================

using Printf

# ----------------------------------------------------------------------------
# Case selection
# ----------------------------------------------------------------------------
# Any of the four (D, L) combinations in the paper. `:D6_L250` is the L/D = 41.7
# large-tube case shown in Figs. 3(a) and 4(a).
const CASE = :D6_L250

const D, L_heated = if CASE === :D4_L100
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

# ----------------------------------------------------------------------------
# Flow conditions used ONLY to size the near-wall cell
# ----------------------------------------------------------------------------
# These need to match the case being run, because y+ scales with the friction
# velocity. Defaults are the 0.7 MPa saturated-liquid state (from the Helmholtz
# H2 EOS, i.e. the same source the solver's property tables come from) at a
# representative flow velocity.
const U_bulk = 5.33          # [m/s]      paper Figs. 3-4 use 1.5 - 11.6 m/s
const rho_l  = 56.75         # [kg/m^3]   saturated LH2 at 0.7 MPa
const mu_l   = 6.99e-6       # [Pa s]

const y_plus_target = 40.0   # midpoint of the 30-50 band

# ----------------------------------------------------------------------------
# Mesh controls
# ----------------------------------------------------------------------------
const dev_length_factor = 10.0   # unheated inlet length, in tube diameters

# Unheated EXIT length, in tube diameters, between the end of the heated wall and
# the outlet plane.
#
# WHY IT EXISTS. Without it the heated wall runs right up to the outlet, which
# puts an outflow boundary INSIDE an active source region. Both `Zerogradient`
# and `Extrapolated` assert `d(alpha)/dn = 0` at the outlet, and that is
# maximally false there: vapour is still being generated in the last cell, so
# `alpha` is accumulating at roughly 4.5e-3 per cell pass at 40 kW/m^2, i.e. an
# axial gradient of order 3 1/m rather than zero. The corner cell is then the one
# place in the domain where a source is applied AND the field is forced flat, and
# it shows up as a step change in `alpha` across the last cell row - measured on
# this case.
#
# 10 D, raised from 3 D. At 3 D a converged ~4.3% excess in `u_tau` sat over the
# last ~3 cells of the heated wall (9 mm, about 1.5 D), with everything else -
# q_conv peaking, q_evap dropping, T_wall below trend - following mechanically
# from it. Two candidates produce a perturbation on that length scale: the
# elliptic pressure response to volumetric expansion STOPPING at the end of the
# plate, which is physical and would persist; or the outlet still being felt
# upstream, which would mean 3 D was simply not enough. Lengthening the run
# separates them - if the excess is unchanged it is the source step.
#
# The cost is small: the exit block is uniform and unheated, so it adds cells but
# no stiffness, and the flow there is single-phase convection.
#
# 0.0 reproduces the original two-block mesh exactly, outlet plane and all.
const exit_length_factor = 10.0
# Radius of the core/ring interface as a fraction of R. The outer ring is then
# (1 - core_frac)*R thick, so RAISING this THINS the ring: 0.45 -> 0.725 halves
# it from 0.55R to 0.275R.
const core_frac         = 0.725
const n_core            = 6      # cells across the core block, each direction
const n_outer           = 2      # cells across the OUTER ring (the wall layer)
# `n_inner` is COMPUTED, not chosen - see `cells_for_geometric`. Set an integer
# here to override the automatic count.
const n_inner_override  = nothing
# Axial cells per tube diameter.
#
#   4.0  production. dz = 1.5 mm, ~44,700 cells for D6_L250.
#   2.0  interim, for rapid turnaround. dz = 3.0 mm, ~22,400 cells.
#
# WHAT HALVING IT ACTUALLY BUYS. The cell count halves, so the cost PER STEP
# halves - but the NUMBER of steps does not change, because that is set by the
# flow-through time and `dt`, not by the mesh. One flow-through is
# L_total/U = 328 mm / 5.33 m/s = 61.5 ms, i.e. ~30,750 steps at dt = 2e-6 s
# either way. So expect ~2x, not more.
#
# `dt` is not the limit you might expect either: at dz = 1.5 mm the convective
# CFL is only U*dt/dz = 0.007, so the time step is set by the stiffness of the
# wall closure and the compressible pressure path, not by the mesh. Coarsening
# axially therefore does NOT license a larger `dt` on CFL grounds.
#
# WHAT IT COSTS, given what is currently being debugged. The wall-tangential
# Laplacian coupling is A/d with A = h_wall*ds and d = dz, so DOUBLING dz HALVES
# the axial coupling and doubles the normal-to-tangential anisotropy - already
# ~305:1 at dz = 1.5 mm with an 86 um wall cell, and worse again now that
# `wall_cell_scale` is back to 1.0. An axial odd-even mode is damped less on this
# mesh, not more. Working against that, upwind differencing supplies more
# numerical diffusion per cell at larger dz, which damps the same mode.
#
# The two effects push opposite ways, so an interim mesh is fine for turnaround
# but is NOT a clean control for checkerboarding: a result that changes between
# 2.0 and 4.0 says something about the mesh, not about the physics.
const n_axial_per_D     = 2.0

# ============================================================================
# Derived geometry
# ============================================================================
const R = D/2
const L_dev = dev_length_factor*D
const L_exit = exit_length_factor*D
const L_total = L_dev + L_heated + L_exit

# Radius of the CORE/RING interface. The core's outer boundary is an ARC at this
# radius (see the `edges` section), not a straight-edged square, so the outer
# ring has the SAME thickness `R - r_i` all the way round. The core block's
# corner vertex therefore sits at r_i/sqrt(2) in each coordinate rather than at
# r_i, which is what puts it on the circle.
const r_i = core_frac*R
const r_c = r_i/sqrt(2)

# ============================================================================
# Near-wall sizing
# ============================================================================

"""
    friction_velocity(U, D, rho, mu) -> (u_tau, Re, f)

Wall friction velocity from the Petukhov (1970) smooth-pipe friction factor,

    f = (0.790 ln(Re) - 1.64)^-2,   tau_w = (f/8) rho U^2

valid for 3000 < Re < 5e6, which covers the whole experimental range (the paper
states Re > 1e4 throughout).
"""
function friction_velocity(U, D, rho, mu)
    Re = rho*U*D/mu
    f = (0.790*log(Re) - 1.64)^-2
    tau_w = (f/8)*rho*U^2
    return sqrt(tau_w/rho), Re, f
end

"""
    first_cell_height(y_plus, u_tau, nu) -> m

Height of the wall-adjacent cell whose CENTRE sits at the requested `y+`.

The wall function is evaluated at the cell centre, so the centre is what must
land in the log layer; the cell is therefore twice that distance tall.
"""
first_cell_height(y_plus, u_tau, nu) = 2*y_plus*nu/u_tau

"""
    grading_for_last_cell(thickness, n, h_last) -> expansion_ratio

blockMesh `simpleGrading` ratio (last cell size / first cell size) that puts a
cell of height `h_last` against the wall, given `n` cells spanning `thickness`.

With per-cell ratio `r`, a geometric distribution gives

    h_first = t (r - 1)/(r^n - 1),    h_last = h_first r^(n-1)

which is inverted here by bisection on `r`. Returns a ratio BELOW one, because
the cells must shrink towards the wall (blockMesh's ratio is last/first and the
block's local axis runs from the core outwards).
"""
function grading_for_last_cell(thickness, n, h_last)
    n >= 2 || return 1.0
    h_uniform = thickness/n
    # Already coarser than uniform would be: no grading needed.
    h_last >= h_uniform && return 1.0

    h_of_r(r) = abs(r - 1) < 1e-12 ? thickness/n :
                thickness*(r - 1)*r^(n - 1)/(r^n - 1)

    # `h_of_r` is monotonically INCREASING in r over (0, 1]: it tends to zero as
    # r -> 0 and equals thickness/n at r = 1. Cells must therefore SHRINK along
    # the block (r < 1) to put a small cell against the wall, and since
    # h_last < h_uniform the root is bracketed by (0, 1] as it stands.
    lo, hi = 1.0e-6, 1.0
    for _ in 1:200
        mid = (lo + hi)/2
        if h_of_r(mid) < h_last
            lo = mid
        else
            hi = mid
        end
    end
    r = (lo + hi)/2
    return r^(n - 1)      # blockMesh wants last/first, which is < 1 here
end

const nu_l = mu_l/rho_l
const u_tau, Re_bulk, f_darcy = friction_velocity(U_bulk, D, rho_l, mu_l)

# Multiplier on the y+-derived wall cell height. 1.0 hits `y_plus_target`.
#
# WHY THIS EXISTS. The RPI evaporative source is deposited entirely into the
# first wall cell, but `TolubinskyKostanchuk` gives a bubble departure diameter
# of 0.6 mm - roughly EIGHTEEN times the 34 um cell that y+ = 40 produces here.
# Concentrating a bubble's worth of vapour into a cell far smaller than the
# bubble is both physically wrong and the measured seed of a static checkerboard
# in `p_rgh` next to the wall: the source is smooth, the pressure response is
# not, and the wall cells are ~1900x more weakly coupled tangentially than
# normally (A/d = 9.1e-6 vs 0.0176), so an odd-even mode along the wall is
# essentially undamped.
#
# Raising this coarsens the wall layer, cutting the per-cell source amplitude and
# moving the first cell closer to the bubble scale. 2.5 takes 34 um -> 85 um and
# y+ ~40 -> ~100, which leaves the log-law wall functions valid (they want
# y+ > 30) while reducing the forcing by the same factor.
#
# BACK TO 1.0. The scaling was a mitigation for the checkerboard described above,
# and it cost the thing the whole mesh is designed around: at 2.0 the achieved y+
# was ~80, outside the 30-50 band the thermal wall function is meshed for, and
# `unit_test_lh2_pipe_sector_mesh.jl` failed on exactly that.
#
# The mitigation is no longer what is holding the checkerboard down. The seed was
# traced to the BULK phase change closure using `|grad(alpha)|` as its
# interfacial area density - a VOF quantity, maximised by a 2*dx oscillation, so
# the source grew with the noise it was responding to. That is now closed
# properly with `DispersedBubbles` (see `2_phase_change_models.jl`), which is
# algebraic in `alpha` and cannot amplify a gradient at all.
#
# Note what returning to 1.0 does cost, so it can be recognised if it bites:
# halving the wall cell DOUBLES the volumetric wall source (A/V = 1/h) and
# roughly doubles the normal-to-tangential coupling anisotropy. If a wall-tangent
# odd-even mode reappears, that is the first thing to put back.
const wall_cell_scale = 1.0

const h_wall_target = wall_cell_scale*first_cell_height(y_plus_target, u_tau, nu_l)

# With the core boundary on an ARC at r_i, the ring is a true annulus: every
# radial edge runs from r_i to R, so the thickness is the same all the way round
# and one grading gives one wall cell height everywhere. (Before the arc edges
# the core was a straight-edged square, the thickness varied between R - r_c on
# the axes and R - sqrt(2) r_c at the diagonal, and the achieved y+ varied with
# it - a 1.5x spread that no single `simpleGrading` could remove.)
const t_ring = R - r_i

t_ring > 0 || error("""
core_frac = $core_frac leaves no room for the outer ring: it must be < 1.
Values around 0.4-0.5 give the best cell quality.""")

# ============================================================================
#  THREE-LAYER CROSS SECTION
# ============================================================================
#
#   core square  ->  INNER ring  ->  OUTER ring  ->  pipe wall
#
#  OUTER ring: bounded by two concentric ARCS (r_m and R), so it is a true
#  annulus - uniform thickness, uniform cell height, no grading. Its cells are
#  the wall cells, and since it is `n_outer` cells of equal size the wall cell
#  height is EXACTLY t_outer/n_outer everywhere. That is what makes y+ a single
#  number rather than a range.
#
#  INNER ring: absorbs the square-to-circle transition. Its inner boundary is
#  the core's STRAIGHT edges (corners at radius r_c) and its outer boundary is
#  the arc at r_m, so its thickness varies - but it never touches the wall, so
#  that variation costs nothing.
#
#  Sizing is inverted from the old scheme: rather than solving a grading to hit
#  a target wall cell, the outer ring is BUILT at the target thickness and the
#  inner ring is graded to meet it.
const t_outer = n_outer*h_wall_target       # uniform cells => exact wall height
const r_m = R - t_outer                     # core/outer interface radius

r_m > r_c || error("""
The outer ring (t_outer = $(round(t_outer*1e6, digits=2)) um over $n_outer cells)
reaches inside the core corner radius r_c = $(round(r_c*1e3, digits=4)) mm.
Reduce `core_frac`, `n_outer`, or `wall_cell_scale`.""")

# Inner ring thickness: exact at the block corners, larger mid-chord where the
# core's straight edge sags inward. Graded so its OUTERMOST cell matches the
# outer ring's uniform cell, avoiding a size jump at r_m.
const t_inner_min = r_m - r_c
const t_inner_max = r_m - r_c*cosd(22.5)
const t_inner_mean = (t_inner_min + t_inner_max)/2

"""
    cells_for_geometric(thickness, h_first, h_last) -> Int

Number of cells a geometric distribution needs to span `thickness` while
starting at `h_first` and ending at `h_last`.

Closed form rather than a search. For `n` cells with per-cell ratio `r`,

    h_last = h_first*r^(n-1)              and     thickness = h_first*(r^n - 1)/(r - 1)

Eliminating `n` between the two collapses the sum to `(r*h_last - h_first)/(r - 1)`,
so

    r = (thickness - h_first)/(thickness - h_last)
    n = 1 + ln(h_last/h_first)/ln(r)

rounded to an integer (at least 2). The grading is then RE-SOLVED for that
integer count, so `h_last` is hit exactly and `h_first` lands close to the
request - the rounding error goes into the first cell, where a small mismatch is
harmless, rather than into the wall cell, where it is not.

### Why size it this way

The inner ring bridges two blocks whose cell sizes are already fixed: the core's
uniform cell on one side and the outer ring's wall cell on the other. Choosing
`n_inner` by hand leaves a size jump at whichever interface the grading does not
happen to match. Solving for it makes both interfaces smooth by construction.
"""
function cells_for_geometric(thickness, h_first, h_last)
    (h_first > 0 && h_last > 0) || error("cell sizes must be positive")
    thickness > max(h_first, h_last) || error("""
thickness $(thickness) cannot hold cells of $(h_first) and $(h_last).""")
    # Uniform request: no geometric progression to solve.
    isapprox(h_first, h_last; rtol=1e-6) && return max(2, round(Int, thickness/h_first))
    r = (thickness - h_first)/(thickness - h_last)
    (r > 0 && !isapprox(r, 1; rtol=1e-12)) || return max(2, round(Int, thickness/h_first))
    return max(2, round(Int, 1 + log(h_last/h_first)/log(r)))
end

# Core cell size along the axis: the core block spans 0 -> r_c in `n_core` cells.
const h_core = r_c/n_core

const n_inner = n_inner_override === nothing ?
    cells_for_geometric(t_inner_mean, h_core, h_wall_target) : n_inner_override
const n_radial = n_inner + n_outer

const inner_grading = grading_for_last_cell(t_inner_mean, n_inner, h_wall_target)

# What the first inner-ring cell actually came out as, for the report: the
# rounding of `n_inner` lands here rather than in the wall cell.
const h_inner_first = h_wall_target/inner_grading

# Retained names used by the reporting block below.
const t_axis = t_outer
const t_diag = t_outer
const t_mean = t_outer
const radial_grading = inner_grading

"""
Actual wall cell height in a block of the given radial thickness, i.e. the
inverse of [`grading_for_last_cell`](@ref). The same grading applied to a
thinner block gives a thinner wall cell, which is why the achieved `y+` is a
range rather than a single number.
"""
function wall_cell_height(thickness, n, grading)
    r = grading^(1/(n - 1))
    abs(r - 1) < 1e-12 && return thickness/n
    return thickness*(r - 1)*r^(n - 1)/(r^n - 1)
end

# The outer ring is uniform, so the wall cell height is the same everywhere and
# these two limits coincide by construction rather than by luck.
const h_wall_axis = t_outer/n_outer
const h_wall_diag = t_outer/n_outer
const yplus_axis = (h_wall_axis/2)*u_tau/nu_l
const yplus_diag = (h_wall_diag/2)*u_tau/nu_l

# ============================================================================
# Vertices
# ============================================================================
const c45 = R/sqrt(2)

# Cross-section vertices, counter-clockwise per block when viewed from +z.
const m45 = r_m/sqrt(2)
const k45 = r_c/sqrt(2)

# Three concentric rings of vertices. Every vertex in a ring is at the SAME
# radius, which is what lets the outer ring be a true annulus.
const xy = [
    (0.0,  0.0),    #  0  v0  centreline
    # --- core corners, radius r_c. STRAIGHT edges between them.
    (r_c,  0.0),    #  1  v1   0 deg
    (k45,  k45),    #  2  v2  45 deg
    (0.0,  r_c),    #  3  v3  90 deg
    # --- inner/outer ring interface, radius r_m. ARC edges between them.
    (r_m,  0.0),    #  4  v4   0 deg
    (m45,  m45),    #  5  v5  45 deg
    (0.0,  r_m),    #  6  v6  90 deg
    # --- pipe wall, radius R. ARC edges between them.
    (R,    0.0),    #  7  v7   0 deg
    (c45,  c45),    #  8  v8  45 deg
    (0.0,  R),      #  9  v9  90 deg
]

const n_xy = length(xy)

# The mesh is a single axial stack split into blocks so the wall patch can be
# split with it: development (unheated), heated, and - when `exit_length_factor`
# is non-zero - an unheated exit run. With no exit length this collapses to the
# original two-block stack and the z levels are unchanged.
const z_levels = L_exit > 0 ?
    [0.0, L_dev, L_dev + L_heated, L_total] :
    [0.0, L_dev, L_total]

vertex_id(ixy, iz) = iz*n_xy + ixy

const n_axial_dev = max(3, round(Int, n_axial_per_D*L_dev/D))
const n_axial_heat = max(4, round(Int, n_axial_per_D*L_heated/D))
const n_axial_exit = L_exit > 0 ? max(3, round(Int, n_axial_per_D*L_exit/D)) : 0

# (cells, label) per axial block, in order. Drives the block emission, the wall
# patch split and the symmetry patches, so they cannot drift apart.
const axial_blocks = L_exit > 0 ?
    [(n_axial_dev, "development (unheated)"),
     (n_axial_heat, "heated section"),
     (n_axial_exit, "exit run (unheated)")] :
    [(n_axial_dev, "development (unheated)"),
     (n_axial_heat, "heated section")]

const n_axial_total = sum(first, axial_blocks)

# Block definitions in the cross-section: (bottom-face vertex order, nx, ny, grading)
# `grading` is the simpleGrading triple applied to the block's (x1, x2, z) axes.
# FIVE blocks: one core, two inner-ring, two outer-ring. The EAST/NORTH pair in
# each ring is the usual butterfly split; the vertex order follows the existing
# convention exactly (EAST: axis 1 radial; NORTH: axis 1 azimuthal, axis 2
# radial), so only the vertex indices change between the two rings.
const blocks_xy = [
    # CORE: uniform, no wall in sight
    ((0, 1, 2, 3), n_core,  n_core,  (1.0, 1.0)),

    # INNER ring, r_c -> r_m. Graded so its outermost cell matches the outer
    # ring's uniform cell size, leaving no jump at the interface.
    ((1, 4, 5, 2), n_inner, n_core,  (inner_grading, 1.0)),   # EAST
    ((3, 2, 5, 6), n_core,  n_inner, (1.0, inner_grading)),   # NORTH

    # OUTER ring, r_m -> R. UNIFORM: two concentric arcs, no grading, so every
    # wall cell is exactly t_outer/n_outer high.
    ((4, 7, 8, 5), n_outer, n_core,  (1.0, 1.0)),             # EAST
    ((6, 5, 8, 9), n_core,  n_outer, (1.0, 1.0)),             # NORTH
]

# ============================================================================
# Emit blockMeshDict
# ============================================================================

function write_blockmeshdict(path)
    open(path, "w") do io
        println(io, """
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     |
    \\\\  /    A nd           | Generated by make_lh2_pipe_sector.jl
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
// 90 degree O-grid sector of a vertically-mounted heated pipe.
//
//   Tatsumoto et al., AIP Conf. Proc. 1573, 44-51 (2014)
//
//   case            : $CASE
//   inner diameter  : $(D*1e3) mm
//   heated length   : $(L_heated*1e3) mm   (L/D = $(round(L_heated/D, digits=1)))
//   development     : $(round(L_dev*1e3, digits=2)) mm  ($(dev_length_factor) D, unheated)
//   exit run        : $(round(L_exit*1e3, digits=2)) mm  ($(exit_length_factor) D, unheated)
//   Re              : $(round(Int, Re_bulk))
//   target y+       : $y_plus_target
//   achieved y+     : $(round(yplus_diag, digits=1)) - $(round(yplus_axis, digits=1))
//   cells           : $(( n_core^2 + 2*n_core*n_radial )*n_axial_total)

FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters 1;

vertices
(""")

        for (iz, z) in enumerate(z_levels)
            println(io, "    // z = $z")
            for (x, y) in xy
                @printf(io, "    (%.10g %.10g %.10g)\n", x, y, z)
            end
        end

        println(io, ");\n\nblocks\n(")

        for (iz, (n_axial, label)) in enumerate(axial_blocks)
            println(io, "    // --- $label ---")
            for (verts, n1, n2, (g1, g2)) in blocks_xy
                bot = [vertex_id(v, iz - 1) for v in verts]
                top = [vertex_id(v, iz) for v in verts]
                @printf(io, "    hex (%d %d %d %d %d %d %d %d) (%d %d %d) simpleGrading (%.8g %.8g 1)\n",
                        bot[1], bot[2], bot[3], bot[4], top[1], top[2], top[3], top[4],
                        n1, n2, n_axial, g1, g2)
            end
        end

        println(io, ");\n\nedges\n(")
        # Arc edges at both ends of every axial block.
        #
        # OUTER arcs (v4->v5->v6) put the block corners on the pipe wall.
        #
        # INNER arcs (v1->v2->v3) put the CORE boundary on a circle of radius
        # r_i as well. Without them the core is a straight-edged square and the
        # outer ring is thickest on the axes and thinnest at the 45 degree
        # diagonal - a 1.5x spread here - which makes the wall cell height, and
        # hence the achieved y+, vary around the circumference no matter what
        # grading is applied. Two concentric circles give a ring of UNIFORM
        # thickness, so one radial grading produces one wall cell height
        # everywhere.
        for iz in 0:(length(z_levels) - 1)
            z = z_levels[iz + 1]
            # OUTER arcs (v4->v5->v6) put the block corners on the pipe wall.
            # INNER arcs (v1->v2->v3) put the core/ring interface on a circle of
            # radius r_i.
            #
            # BOTH are needed for a uniform near-wall layer. With the inner
            # boundary curved the outer block is a true ANNULUS: every radial
            # edge runs exactly from r_i to R, so one grading gives ONE wall cell
            # height all the way round. Leave the inner edges straight and the
            # chord sags inward by (1 - cos(22.5 deg)) = 7.6% of r_i, making the
            # ring thicker mid-edge and the wall cell height vary with it - 20%
            # on this geometry, which no `simpleGrading` can remove.
            # Arcs on BOTH boundaries of the outer ring - r_m (v4-v5-v6) and
            # R (v7-v8-v9) - which is what makes it a true annulus. The core's
            # own edges (v1-v2-v3) are deliberately left STRAIGHT; the inner ring
            # absorbs that square-to-circle transition.
            for (a, b, mid_deg, rad) in ((7, 8, 22.5, R),   (8, 9, 67.5, R),
                                         (4, 5, 22.5, r_m), (5, 6, 67.5, r_m))
                mx = rad*cosd(mid_deg)
                my = rad*sind(mid_deg)
                @printf(io, "    arc %d %d (%.10g %.10g %.10g)\n",
                        vertex_id(a, iz), vertex_id(b, iz), mx, my, z)
            end
        end
        println(io, ");\n\nboundary\n(")

        # --- helper to emit a face from cross-section vertex pair + axial level
        # A side face of an axial block spans two cross-section vertices at the
        # lower and upper z of that block.
        side_face(a, b, iz) = (vertex_id(a, iz - 1), vertex_id(b, iz - 1),
                               vertex_id(b, iz), vertex_id(a, iz))

        function emit_patch(name, type, faces)
            println(io, "    $name")
            println(io, "    {")
            println(io, "        type $type;")
            println(io, "        faces")
            println(io, "        (")
            for f in faces
                @printf(io, "            (%d %d %d %d)\n", f[1], f[2], f[3], f[4])
            end
            println(io, "        );")
            println(io, "    }")
        end

        # Inlet: bottom faces of all five cross-section blocks, at z = 0.
        inlet_faces = [reverse([vertex_id(v, 0) for v in verts])
                       for (verts, _, _, _) in blocks_xy]
        emit_patch("inlet", "patch", inlet_faces)

        # Outlet: top faces at z = L_total.
        top_iz = length(z_levels) - 1
        outlet_faces = [[vertex_id(v, top_iz) for v in verts]
                        for (verts, _, _, _) in blocks_xy]
        emit_patch("outlet", "patch", outlet_faces)

        # Unheated wall and heated wall, split by axial block. The wall is the
        # OUTER ring's outer arc: EAST face v7->v8, NORTH face v8->v9.
        #
        # Block 1 is the inlet development run and block 3, when present, the
        # exit run - BOTH are unheated, so both go on `wallUnheated`. Only block
        # 2 carries the `FixedHeatFlux` in the case file. Getting this wrong
        # would heat the exit section and defeat the point of having one.
        unheated_iz = L_exit > 0 ? (1, 3) : (1,)
        emit_patch("wallUnheated", "wall",
                   vcat([[side_face(7, 8, iz), side_face(8, 9, iz)]
                         for iz in unheated_iz]...))
        emit_patch("pipeWall", "wall",
                   [side_face(7, 8, 2), side_face(8, 9, 2)])

        # Symmetry plane y = 0 (the x-z plane), running outwards from the axis:
        # CORE v0->v1, INNER v1->v4, OUTER v4->v7.
        sym_y = vcat([[side_face(0, 1, iz), side_face(1, 4, iz),
                       side_face(4, 7, iz)] for iz in 1:length(axial_blocks)]...)
        emit_patch("symmetryY", "symmetry", sym_y)

        # Symmetry plane x = 0 (the y-z plane), running inwards to the axis:
        # OUTER v9->v6, INNER v6->v3, CORE v3->v0.
        sym_x = vcat([[side_face(3, 0, iz), side_face(6, 3, iz),
                       side_face(9, 6, iz)] for iz in 1:length(axial_blocks)]...)
        emit_patch("symmetryX", "symmetry", sym_x)

        println(io, ");\n\nmergePatchPairs\n(\n);\n")
        println(io, "// ************************************************************************* //")
    end
end

function write_controldict(path)
    open(path, "w") do io
        println(io, """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
// Minimal dictionary: blockMesh requires one to exist, nothing here is used.
application     blockMesh;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
""")
    end
end

# ============================================================================
# Run
# ============================================================================

const here = @__DIR__
mkpath(joinpath(here, "system"))

write_blockmeshdict(joinpath(here, "system", "blockMeshDict"))
write_controldict(joinpath(here, "system", "controlDict"))

const n_cells_xy = n_core^2 + 2*n_core*n_radial
const n_cells = n_cells_xy*n_axial_total

println("""
=============================================================================
 LH2 vertical heated pipe - 90 degree O-grid sector
=============================================================================
 Case                : $CASE
 Inner diameter D    : $(D*1e3) mm
 Heated length  L    : $(L_heated*1e3) mm     (L/D = $(round(L_heated/D, digits=2)))
 Development length  : $(round(L_dev*1e3, digits=2)) mm  ($dev_length_factor D, unheated)
 Exit run            : $(round(L_exit*1e3, digits=2)) mm  ($exit_length_factor D, unheated)
 Total length        : $(round(L_total*1e3, digits=2)) mm

 Flow conditions used for near-wall sizing
   U_bulk            : $U_bulk m/s
   rho, mu           : $rho_l kg/m^3, $mu_l Pa s
   Reynolds number   : $(round(Int, Re_bulk))
   Darcy f (Petukhov): $(round(f_darcy, digits=5))
   u_tau             : $(round(u_tau, digits=4)) m/s

 Near-wall resolution
   target y+         : $y_plus_target
   wall cell height  : $(round(h_wall_diag*1e6, digits=2)) - $(round(h_wall_axis*1e6, digits=2)) um
   achieved y+       : $(round(yplus_diag, digits=1)) - $(round(yplus_axis, digits=1))
   radial grading    : $(round(radial_grading, digits=5))  (inner ring, last/first)
   cell size match   : core $(round(h_core*1e6, digits=2)) um -> inner-ring first cell $(round(h_inner_first*1e6, digits=2)) um  (ratio $(round(h_inner_first/h_core, digits=3)))
                       inner-ring last cell $(round(h_wall_target*1e6, digits=2)) um -> wall cell $(round(t_outer/n_outer*1e6, digits=2)) um  (ratio $(round((t_outer/n_outer)/h_wall_target, digits=3)))
   inner/outer cells : $(n_inner) + $(n_outer)

 Mesh
   core block        : $n_core x $n_core
   radial cells      : $n_radial
   axial cells       : $(join(["$n ($label)" for (n, label) in axial_blocks], " + "))
   total cells       : $n_cells

 Written:
   system/blockMeshDict
   system/controlDict

 Next:
   ./run_blockMesh.sh
   julia --project=. test/unit_test_lh2_pipe_sector_mesh.jl
=============================================================================""")

if !(30 <= yplus_diag && yplus_axis <= 50)
    @warn """The achieved y+ range ($(round(yplus_diag, digits=1)) - $(round(yplus_axis, digits=1))) \
falls outside the 30-50 band the wall functions want.

y+ below ~30 puts the first cell centre in the buffer layer, where the log-law \
branch of the thermal wall function - and therefore the RPI convective flux - is \
not valid. Adjust `n_radial` or `core_frac`, or revisit `U_bulk` if the case \
velocity has changed."""
end
