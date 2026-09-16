# =============================================================================
#  blockMeshDict generator - NASA K-Site LH2 tank, axisymmetric wedge
# =============================================================================
#
#  Emits system/blockMeshDict (and a minimal system/controlDict) for a wedge of
#  the K-Site ellipsoidal tank. Run this with Julia, then run `blockMesh` in
#  this directory (or `./run_blockMesh.sh` if the path contains spaces);
#  XCALibre loads the result with `FOAM3D_mesh`.
#
#      julia make_ksite_wedge.jl
#      blockMesh
#
#  ---------------------------------------------------------------------------
#  Geometry
#  ---------------------------------------------------------------------------
#  Fernandes et al. (2026), Sec. 2: the K-Site tank is "approximately an
#  ellipsoid, with a major diameter of 2.20 m, minor diameter of 1.93 m, a
#  surface area of 13.98 m^2 and volume of 4.89 m^3".
#
#  Taking the two diameters at face value, a = 1.100 m and b = 0.965 m, both
#  derived quantities reproduce the paper exactly:
#
#      V = (4/3) pi a^2 b            = 4.891 m^3   (paper: 4.89)
#      S = 2 pi a^2 (1 + (1-e^2)/e * atanh(e)) = 13.974 m^2  (paper: 13.98)
#
#  (An earlier revision of this file assumed a 1.2 major:minor axis ratio taken
#  from a secondary source. That is wrong - the true ratio is 2.20/1.93 = 1.14 -
#  and it produced a tank with b = 0.932 m.)
#
#  Surface area matters as much as volume here: the K-Site cases are specified
#  by wall heat FLUX, so S sets the total heat input (3.5 W/m^2 x 13.98 m^2 =
#  48.9 W, matching the paper's 49.0 W for case K(i)).
#
#  ---------------------------------------------------------------------------
#  Topology
#  ---------------------------------------------------------------------------
#  An O-grid in the (r,z) half-plane - a rectangular core plus north, east and
#  south blocks reaching the elliptical wall - with the core and east regions
#  each split into three rows, so that an EQUATORIAL BAND of flat,
#  uniform-height layers runs all the way from the axis to the wall:
#
#     z ^
#       |  O_top
#       |    |`-.
#       |    |    `-.     N
#       |  C3+---------+C2 `-.
#       |    | core_hi |  E_hi `O_ne
#       |A_hi+---------+B_hi-----+W_hi     z = +band_half  -.
#       |    |core_band|  E_band |                           } flat, uniform
#       |A_lo+---------+B_lo-----+W_lo     z = -band_half  -'  layers; a face
#       |    | core_lo |  E_lo .O_se                           lies on z = 0
#       |  C0+---------+C1 .-'
#       |    |    .-'     S
#       |    |.-'
#       |  O_bot
#       +-------------------------------> r
#
#  Why the band. Both K-Site cases are 50% full, so the liquid/vapour interface
#  sits at z = 0. In the plain O-grid the east block's layers fanned out towards
#  the wall: the cell layer straddling z = 0 was 2.0 cm tall at the axis and
#  4.0 cm at the wall. A VOF interface smeared across cells whose height varies
#  along it gives a density field that is not a function of z alone, so it
#  cannot be hydrostatic. Discretely, balance on the layer's top and bottom
#  faces needs p_rgh inside the layer to sit g*h*(rho_l - rho_m) above the
#  liquid, which varies with the layer half-height h(r) - and the faces BETWEEN
#  layer cells carry no density jump to balance the resulting radial gradient.
#  MEASURED on that mesh with no heat flux and no phase change: the predicted
#  p_rgh offset matched the solution at every radius (ratio 0.97-1.01; 3.3 Pa at
#  the axis, 6.8 Pa at the wall), and the ~0.14 m/s^2 it implies drove a
#  spurious inward current of up to 0.34 m/s along the interface of a tank that
#  should have stayed at rest.
#
#  Inside the band every layer is horizontal with the same height at every
#  radius, and the layer count is even so that a face lies exactly on z = 0: a
#  sharp initial interface then sits on faces where gh = 0. The band has to
#  contain the interface for the whole run. An upper bound on the K(i) level
#  drop - all 49 W going into boil-off for 17.5 hr - is ~2.6 cm.
#
#  The band is centred on the equator, so it serves 50% fill. A different fill
#  level would need it centred on that z_fill instead.
#
#  The wall arcs of the cap blocks (E_lo, E_hi) are graded so the wall cells
#  shrink to the band's spacing at the band edge; the size jump against the
#  north/south blocks is moved out to the 45 deg corners, well away from the
#  interface.
#
#  The r = 0 block faces have coincident vertex pairs, which blockMesh collapses
#  into prism cells (standard OpenFOAM wedge practice).
#
#  The two wedge planes are written as `symmetry` patches (not OpenFOAM
#  `wedge`), because XCALibre has no wedge type and the case assigns
#  `Symmetry(:wedgeFront)` / `Symmetry(:wedgeBack)`.
# =============================================================================

using Printf

# ----------------------------------------------------------------------------
# User parameters
# ----------------------------------------------------------------------------
const D_major     = 2.20        # [m] major (equatorial) diameter  - Fernandes et al. Sec. 2
const D_minor     = 1.93        # [m] minor (polar) diameter        - Fernandes et al. Sec. 2

const wedge_angle = 5.0         # [deg] total included angle
const core_frac   = 0.4         # core half-extent as a fraction of the semi-axes
const split_deg   = 45.0        # ellipse angle at which the core diagonal meets the wall

const h_base      = 0.02        # [m] target base cell size
const wall_grading = 5.0        # last/first cell ratio across the near-wall blocks
                                # (>1 refines towards the tank wall)

# Equatorial band of flat, uniform layers - see "Why the band" above.
const band_half   = 0.10        # [m] the band spans -band_half <= z <= +band_half
const h_band      = h_base      # [m] target layer height inside the band

# ----------------------------------------------------------------------------
# Derived geometry
# ----------------------------------------------------------------------------
a = D_major/2      # equatorial semi-axis [m] = 1.100
b = D_minor/2      # polar semi-axis      [m] = 0.965

V_actual = 4/3*pi*a^2*b
# oblate spheroid surface area (b < a)
let e = sqrt(1 - (b/a)^2)
    global S_actual = 2pi*a^2*(1 + (1 - e^2)/e*atanh(e))
end

r_c = core_frac*a
z_c = core_frac*b

0 < band_half < z_c || error(
    "`band_half` = $band_half m must lie inside the core half-height z_c = $(round(z_c, sigdigits=4)) m")

phi = deg2rad(split_deg)
half = deg2rad(wedge_angle/2)

# Ellipse parameter of the wall point at the band edge, on (a cos t, b sin t).
t_band = asin(band_half/b)

# (r,z) points, indices match the vertex numbering used below
#   0:C0  1:C1  2:C2  3:C3  4:O_bot  5:O_se  6:O_ne  7:O_top
#   8:A_lo  9:B_lo  10:W_lo  11:A_hi  12:B_hi  13:W_hi   (band corners:
#                                                         axis, core edge, wall)
pts_rz = [
    (0.0,  -z_c),                        # 0  C0
    (r_c,  -z_c),                        # 1  C1
    (r_c,   z_c),                        # 2  C2
    (0.0,   z_c),                        # 3  C3
    (0.0,  -b),                          # 4  O_bot   phi = -90
    (a*cos(phi), -b*sin(phi)),           # 5  O_se    phi = -45
    (a*cos(phi),  b*sin(phi)),           # 6  O_ne    phi = +45
    (0.0,   b),                          # 7  O_top   phi = +90
    (0.0,  -band_half),                  # 8  A_lo
    (r_c,  -band_half),                  # 9  B_lo
    (a*cos(t_band), -b*sin(t_band)),     # 10 W_lo    z = -band_half, on the wall
    (0.0,   band_half),                  # 11 A_hi
    (r_c,   band_half),                  # 12 B_hi
    (a*cos(t_band),  b*sin(t_band)),     # 13 W_hi    z = +band_half, on the wall
]
const NV = length(pts_rz)      # vertices per wedge plane

"""3D coordinates of an (r,z) point rotated to the back (-) or front (+) wedge plane."""
rotate(r, z, sgn) = (r*cos(half), sgn*r*sin(half), z)

# ----------------------------------------------------------------------------
# Cell counts (chosen so shared block edges agree)
# ----------------------------------------------------------------------------
ncell(len) = max(1, round(Int, len/h_base))

n_core_r = ncell(r_c)                              # axis -> r_c
n_cap    = ncell(z_c - band_half)                  # band edge -> +-z_c (cap rows)
n_band   = 2*max(1, round(Int, band_half/h_band))  # EVEN, so a face lies on z = 0
dz_band  = 2*band_half/n_band
# mean core-to-wall span, used for the radial direction of the N/E/S blocks
span_rad = ((a - r_c) + (b - z_c))/2
n_rad    = ncell(span_rad)

n_cells_total = (n_core_r + n_rad)*(2*n_cap + n_band) + 2*n_core_r*n_rad

# ----------------------------------------------------------------------------
# Wall-arc grading of the cap blocks
# ----------------------------------------------------------------------------
"""Length of the elliptical wall between ellipse parameters t1 < t2 (composite Simpson)."""
function wall_arc_length(t1, t2; n=2000)
    f(t) = sqrt((a*sin(t))^2 + (b*cos(t))^2)
    h = (t2 - t1)/n
    s = f(t1) + f(t2)
    for i in 1:n-1
        s += (isodd(i) ? 4 : 2)*f(t1 + i*h)
    end
    return s*h/3
end

"""
    geometric_growth(L, n, c_end) -> rho

Per-cell growth factor of `n` geometric cells that sum to `L`, with the cell at
one end of size `c_end` and each further cell `rho` times the previous one.
"""
function geometric_growth(L, n, c_end)
    total(rho) = abs(rho - 1) < 1e-12 ? n*c_end : c_end*(rho^n - 1)/(rho - 1)
    lo, hi = 1e-3, 1e3
    total(lo) <= L <= total(hi) || error(
        "cannot grade $n cells over $L m to an end cell of $c_end m")
    for _ in 1:200
        mid = sqrt(lo*hi)
        total(mid) < L ? (lo = mid) : (hi = mid)
    end
    return sqrt(lo*hi)
end

L_band_wall = wall_arc_length(-t_band, t_band)     # W_lo -> W_hi
L_cap_wall  = wall_arc_length(-phi, -t_band)       # O_se -> W_lo (and W_hi -> O_ne)
c_band_wall = L_band_wall/n_band
rho_cap     = geometric_growth(L_cap_wall, n_cap, c_band_wall)
c_corner    = c_band_wall*rho_cap^(n_cap - 1)      # cap wall cell at the 45 deg corner

# blockMesh expansion ratio = last cell / first cell along the edge direction.
R_lo_wall = 1/rho_cap^(n_cap - 1)   # O_se -> W_lo: the band end is LAST
R_hi_wall = rho_cap^(n_cap - 1)     # W_hi -> O_ne: the band end is FIRST

# ----------------------------------------------------------------------------
# Emit blockMeshDict
# ----------------------------------------------------------------------------
foam_header(class, object) = """
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM                                        |
|  \\\\    /   O peration     |                                                 |
|   \\\\  /    A nd           | Generated by make_ksite_wedge.jl (XCALibre.jl)  |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       $class;
    object      $object;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

io = IOBuffer()
print(io, foam_header("dictionary", "blockMeshDict"))

@printf(io, "\n// K-Site tank: a = %.4f m, b = %.4f m, V = %.4f m^3, S = %.4f m^2\n",
        a, b, V_actual, S_actual)
@printf(io, "// wedge = %.1f deg   cells = %d\n", wedge_angle, n_cells_total)
@printf(io, "// equatorial band |z| <= %.4f m: %d flat layers of %.4f m, a face on z = 0\n\n",
        band_half, n_band, dz_band)

println(io, "scale   1;\n")

# --- vertices: front plane 0..NV-1, back plane NV..2NV-1 ---------------------
#
# The FRONT plane (theta = +2.5 deg) must come first. With the in-plane quads
# ordered as below, the block-local axes are x = +r, y = +z, and z = plane0 ->
# plane1. Since x_hat cross z_hat = -y_hat, the third axis has to point in -y,
# i.e. from +theta to -theta. Listing the back plane first gives a left-handed
# block and blockMesh rejects it with "has inward-pointing faces".
const PLANES = ((+1, "front", 0), (-1, "back", NV))

println(io, "vertices")
println(io, "(")
for (sgn, label, off) in PLANES
    @printf(io, "    // %s plane (theta = %+.2f deg)\n", label, sgn*wedge_angle/2)
    for (i, (r, z)) in enumerate(pts_rz)
        x, y, zz = rotate(r, z, sgn)
        @printf(io, "    (%18.12f %18.12f %18.12f)   // %d\n", x, y, zz, i - 1 + off)
    end
end
println(io, ");\n")

# --- blocks -----------------------------------------------------------------
# Each entry: (name, front-plane vertex quad, (n1,n2), 12 edge expansion ratios)
#
# Ratios follow OpenFOAM's edge numbering for `edgeGrading`:
#   x: 1 (0 1)  2 (3 2)  3 (7 6)  4 (4 5)
#   y: 5 (0 3)  6 (1 2)  7 (5 6)  8 (4 7)
#   z: 9 (0 4) 10 (1 5) 11 (2 6) 12 (3 7)
# A block whose edges in each direction agree is written as `simpleGrading`.
#
# Grading refines towards the tank wall in every near-wall block, accounting
# for which way the local axis points. In the cap blocks E_lo/E_hi the y-edge on
# the core side stays uniform and the one on the wall arc is graded towards the
# band (see "Wall-arc grading" above).
g = wall_grading
simple(gx, gy) = (gx, gx, gx, gx, gy, gy, gy, gy, 1.0, 1.0, 1.0, 1.0)
cap(R_wall) = (1/g, 1/g, 1/g, 1/g, 1.0, R_wall, R_wall, 1.0, 1.0, 1.0, 1.0, 1.0)

blocks = [
    ("core_lo",   (0, 1, 9, 8),    (n_core_r, n_cap),  simple(1.0, 1.0)),
    ("core_band", (8, 9, 12, 11),  (n_core_r, n_band), simple(1.0, 1.0)),
    ("core_hi",   (11, 12, 2, 3),  (n_core_r, n_cap),  simple(1.0, 1.0)),
    ("east_lo",   (1, 5, 10, 9),   (n_rad, n_cap),     cap(R_lo_wall)),   # x: core -> wall
    ("east_band", (9, 10, 13, 12), (n_rad, n_band),    simple(1/g, 1.0)), # x: core -> wall
    ("east_hi",   (12, 13, 6, 2),  (n_rad, n_cap),     cap(R_hi_wall)),   # x: core -> wall
    ("north",     (3, 2, 6, 7),    (n_core_r, n_rad),  simple(1.0, 1/g)), # y: core -> wall
    ("south",     (4, 5, 1, 0),    (n_core_r, n_rad),  simple(1.0, g)),   # y: wall -> core
]

hexverts(q) = (q[1], q[2], q[3], q[4], q[1]+NV, q[2]+NV, q[3]+NV, q[4]+NV)

# 3D coordinates of a global vertex id (0..NV-1 front plane, NV..2NV-1 back plane)
function vertex_xyz(id)
    sgn, _, off = id < NV ? PLANES[1] : PLANES[2]
    r, z = pts_rz[id - off + 1]
    return collect(rotate(r, z, sgn))
end

cross3(u, v) = [u[2]*v[3] - u[3]*v[2], u[3]*v[1] - u[1]*v[3], u[1]*v[2] - u[2]*v[1]]

# blockMesh requires each hex to be right-handed: (v0->v1) x (v0->v3) must point
# along (v0->v4). A left-handed block is rejected with "has inward-pointing
# faces", so assert it here rather than discovering it at meshing time.
#
# The local axes are averaged over all four parallel edges rather than taken at
# v0 alone: on the axis blocks v0 lies at r = 0, where the two wedge planes
# coincide, so the single v0->v4 edge there has zero length and the triple
# product degenerates to 0. Averaging keeps the test meaningful.
function assert_right_handed(name, v)
    p(i) = vertex_xyz(v[i])
    mean4(pairs) = sum(p(j) .- p(i) for (i, j) in pairs) ./ 4

    xa = mean4(((1,2), (4,3), (5,6), (8,7)))   # local x
    ya = mean4(((1,4), (2,3), (5,8), (6,7)))   # local y
    za = mean4(((1,5), (2,6), (3,7), (4,8)))   # local z

    handedness = sum(cross3(xa, ya) .* za)
    handedness > 0 || error(
        "block `$name` is left-handed (triple product = $handedness); " *
        "blockMesh would reject it as having inward-pointing faces")
    return handedness
end

# Every edge shared by two blocks must carry the same number of cells and the
# same point distribution, or blockMesh cannot merge the blocks. Checked here
# for the same reason as handedness: this file is often edited where blockMesh
# is not available to find out.
function assert_consistent_edges(blocks)
    seen = Dict{Tuple{Int,Int}, Tuple{Int,Float64,String}}()
    for (name, q, (n1, n2), gr) in blocks
        # the two wedge planes of a block must be graded identically
        (gr[1] == gr[4] && gr[2] == gr[3] && gr[5] == gr[8] && gr[6] == gr[7]) || error(
            "block `$name` is graded differently on its two wedge planes")
        v = hexverts(q)
        for (va, vb, n, R) in ((v[1], v[2], n1, gr[1]), (v[4], v[3], n1, gr[2]),
                               (v[1], v[4], n2, gr[5]), (v[2], v[3], n2, gr[6]))
            key = va < vb ? (va, vb) : (vb, va)
            Rk  = va < vb ? R : 1/R            # ratio oriented low id -> high id
            if haskey(seen, key)
                n0, R0, other = seen[key]
                (n == n0 && isapprox(Rk, R0; rtol=1e-9)) || error(
                    "edge $key: block `$name` has $n cells (ratio $Rk) but " *
                    "block `$other` has $n0 cells (ratio $R0)")
            else
                seen[key] = (n, Rk, name)
            end
        end
    end
    return nothing
end
assert_consistent_edges(blocks)

function grading_string(gr)
    uniform(r) = all(gr[i] == gr[first(r)] for i in r)
    if uniform(1:4) && uniform(5:8) && uniform(9:12)
        return @sprintf("simpleGrading (%.10g %.10g %.10g)", gr[1], gr[5], gr[9])
    end
    return "edgeGrading (" * join((@sprintf("%.10g", x) for x in gr), " ") * ")"
end

println(io, "blocks")
println(io, "(")
for (name, q, (n1, n2), gr) in blocks
    v = hexverts(q)
    assert_right_handed(name, v)
    @printf(io, "    hex (%d %d %d %d %d %d %d %d) (%d %d 1) %s  // %s\n",
            v..., n1, n2, grading_string(gr), name)
end
println(io, ");\n")

# --- edges: elliptical arcs on the tank wall --------------------------------
# Arc interpolation points at the mid-parameter of each wall segment.
tb = rad2deg(t_band)
arcs = [(4,  5,  -(90 + split_deg)/2),   # O_bot -> O_se
        (5,  10, -(split_deg + tb)/2),   # O_se  -> W_lo
        (10, 13,   0.0),                 # W_lo  -> W_hi  (the band, through the equator)
        (13, 6,   (split_deg + tb)/2),   # W_hi  -> O_ne
        (6,  7,   (90 + split_deg)/2)]   # O_ne  -> O_top

println(io, "edges")
println(io, "(")
for (v1, v2, mid_deg) in arcs
    m = deg2rad(mid_deg)
    r_m, z_m = a*cos(m), b*sin(m)
    for (sgn, _, off) in PLANES
        x, y, z = rotate(r_m, z_m, sgn)
        @printf(io, "    arc %d %d (%18.12f %18.12f %18.12f)\n", v1+off, v2+off, x, y, z)
    end
end
println(io, ");\n")

# --- boundary ---------------------------------------------------------------
# Face quads follow OpenFOAM's hex-local convention:
#   x-min (0 4 7 3)  x-max (1 2 6 5)  y-min (0 1 5 4)
#   y-max (3 7 6 2)  z-min (0 3 2 1)  z-max (4 5 6 7)
face(v, idx) = (v[idx[1]+1], v[idx[2]+1], v[idx[3]+1], v[idx[4]+1])
V = Dict(name => hexverts(q) for (name, q, _, _) in blocks)

wall_faces = [face(V["east_lo"],   (1,2,6,5)),    # x-max -> arc
              face(V["east_band"], (1,2,6,5)),    # x-max -> arc
              face(V["east_hi"],   (1,2,6,5)),    # x-max -> arc
              face(V["north"],     (3,7,6,2)),    # y-max -> arc
              face(V["south"],     (0,1,5,4))]    # y-min -> arc

# The r = 0 faces of the blocks touching the axis. blockMesh does NOT collapse
# these into edges even though their vertex pairs are coincident: it emits them
# as genuine zero-area faces. Left unnamed they land in an auto-generated
# `defaultFaces` patch, so name them explicitly instead. They carry no flux
# (every kernel multiplies by area) and take an `Empty` BC in XCALibre, exactly
# like `frontAndBack` on a 2D-from-3D mesh.
axis_faces = [face(V[name], (0,4,7,3))            # x-min -> axis
              for name in ("core_lo", "core_band", "core_hi", "north", "south")]

# Vertices 0..NV-1 are the front plane, so the block-local z-min face lies on
# wedgeFront and z-max on wedgeBack (see PLANES above).
front_faces = [face(V[name], (0,3,2,1)) for (name, _, _, _) in blocks]
back_faces  = [face(V[name], (4,5,6,7)) for (name, _, _, _) in blocks]

function print_patch(io, name, type, faces)
    println(io, "    $name")
    println(io, "    {")
    println(io, "        type            $type;")
    println(io, "        faces")
    println(io, "        (")
    for f in faces
        @printf(io, "            (%d %d %d %d)\n", f...)
    end
    println(io, "        );")
    println(io, "    }")
end

println(io, "boundary")
println(io, "(")
print_patch(io, "tankWall",   "wall",     wall_faces)
print_patch(io, "wedgeBack",  "symmetry", back_faces)
print_patch(io, "wedgeFront", "symmetry", front_faces)
print_patch(io, "axis",       "empty",    axis_faces)
println(io, ");\n")

println(io, "mergePatchPairs\n(\n);\n")
println(io, "// ************************************************************************* //")

mkpath("system")
write(joinpath("system", "blockMeshDict"), String(take!(io)))

# --- minimal controlDict so blockMesh will run ------------------------------
cio = IOBuffer()
print(cio, foam_header("dictionary", "controlDict"))
print(cio, """
application     blockMesh;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
writeFormat     ascii;
writePrecision  12;
runTimeModifiable true;

// ************************************************************************* //
""")
write(joinpath("system", "controlDict"), String(take!(cio)))

# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
println("Wrote system/blockMeshDict and system/controlDict\n")
@printf("  semi-axes         : a = %.4f m (equatorial), b = %.4f m (polar)\n", a, b)
@printf("  major diameter    : %.4f m   (paper: 2.20 m)\n", 2a)
@printf("  minor diameter    : %.4f m   (paper: 1.93 m)\n", 2b)
@printf("  volume            : %.4f m^3 (paper: 4.89 m^3)\n", V_actual)
@printf("  surface area      : %.4f m^2 (paper: 13.98 m^2)\n", S_actual)
@printf("  => total heat     : %.2f W at 3.5 W/m^2 (paper K(i): 49.0 W)\n", 3.5*S_actual)
@printf("                      %.2f W at 2.0 W/m^2 (paper K(ii): 28.0 W)\n", 2.0*S_actual)
@printf("  wedge angle       : %.1f deg\n", wedge_angle)
@printf("  cells             : %d  (core r %d; z %d cap + %d band + %d cap; radial %d)\n",
        n_cells_total, n_core_r, n_cap, n_band, n_cap, n_rad)
@printf("  base cell size    : %.4f m, wall grading %.1f\n", h_base, wall_grading)
@printf("  equatorial band   : |z| <= %.4f m, %d flat layers of %.4f m, a face on z = 0\n",
        band_half, n_band, dz_band)
@printf("  wall spacing      : band %.4f m; cap arcs graded %.4f -> %.4f m (growth %.3f/cell)\n",
        c_band_wall, c_band_wall, c_corner, rho_cap)
@printf("                      north/south arcs %.4f m (meet the caps at the 45 deg corners)\n",
        wall_arc_length(phi, pi/2)/n_core_r)
@printf("                      core-side cap spacing %.4f m against band %.4f m\n",
        (z_c - band_half)/n_cap, dz_band)
println()
println("Fill levels (z of the flat interface, for setField_Box!):")
println("  Both K-Site cases K(i) and K(ii) use 50% fill, i.e. z_fill = 0 exactly")
println("  (the ellipsoid is symmetric about the equator), on a face of the band.")
println("  Other fill levels below lie OUTSIDE the band, which would need re-centring:")
for f in (0.25, 0.50, 0.90)
    target = 4/3*f - 2/3
    u = target
    for _ in 1:60; u -= (u - u^3/3 - target)/(1 - u^2); end
    @printf("  %2d%% by volume -> z_fill = %+.6f m\n", round(Int, f*100), u*b)
end
println("\nNext: run `blockMesh` here (or ./run_blockMesh.sh), then validate with")
println("  julia --project=. test/unit_test_ksite_wedge_mesh.jl")
