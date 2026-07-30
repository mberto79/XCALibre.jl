# =============================================================================
#  blockMeshDict generator - NASA K-Site LH2 tank, axisymmetric wedge
# =============================================================================
#
#  Emits system/blockMeshDict (and a minimal system/controlDict) for a wedge of
#  the K-Site ellipsoidal tank. Run this with Julia, then run `blockMesh` in
#  this directory; XCALibre loads the result with `FOAM3D_mesh`.
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
#  Butterfly (O-grid) in the (r,z) half-plane: a rectangular core plus north,
#  east and south blocks reaching the elliptical wall. This keeps cell quality
#  reasonable and confines the axis degeneracy to the r = 0 block faces, which
#  blockMesh collapses into prism cells (standard OpenFOAM wedge practice).
#
#      z
#      ^          . - O_top - .
#      |       .   |    N     |  .
#      |     O_ne__|__________|
#      |     |     C3        C2  \
#      |     |  E  |   CORE   |   |
#      |     |     C0        C1   |      (mirrored below z = 0)
#      |     O_se--|----------|
#      |       .   |    S     |  .
#      |          ` - O_bot - `
#      +--------------------------> r
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

phi = deg2rad(split_deg)
half = deg2rad(wedge_angle/2)

# (r,z) points, indices match the vertex numbering used below
#   0:C0  1:C1  2:C2  3:C3  4:O_bot  5:O_se  6:O_ne  7:O_top
pts_rz = [
    (0.0,  -z_c),               # 0 C0
    (r_c,  -z_c),               # 1 C1
    (r_c,   z_c),               # 2 C2
    (0.0,   z_c),               # 3 C3
    (0.0,  -b),                 # 4 O_bot   phi = -90
    (a*cos(phi), -b*sin(phi)),  # 5 O_se    phi = -45
    (a*cos(phi),  b*sin(phi)),  # 6 O_ne    phi = +45
    (0.0,   b),                 # 7 O_top   phi = +90
]

"""3D coordinates of an (r,z) point rotated to the back (-) or front (+) wedge plane."""
rotate(r, z, sgn) = (r*cos(half), sgn*r*sin(half), z)

# ----------------------------------------------------------------------------
# Cell counts (chosen so shared block edges agree)
# ----------------------------------------------------------------------------
ncell(len) = max(1, round(Int, len/h_base))

n_core_r = ncell(r_c)             # axis -> r_c   (core r-dir; N/S first dir)
n_core_z = ncell(2*z_c)           # -z_c -> +z_c  (core z-dir; E second dir)
# mean core-to-wall span, used for the radial direction of the N/E/S blocks
span_rad = ((a - r_c) + (b - z_c))/2
n_rad    = ncell(span_rad)

n_cells_total = n_core_r*n_core_z + n_rad*n_core_z + 2*n_core_r*n_rad

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
@printf(io, "// wedge = %.1f deg   cells = %d\n\n", wedge_angle, n_cells_total)

println(io, "scale   1;\n")

# --- vertices: front plane 0..7, back plane 8..15 ---------------------------
#
# The FRONT plane (theta = +2.5 deg) must come first. With the in-plane quads
# ordered as below, the block-local axes are x = +r, y = +z, and z = plane0 ->
# plane1. Since x_hat cross z_hat = -y_hat, the third axis has to point in -y,
# i.e. from +theta to -theta. Listing the back plane first gives a left-handed
# block and blockMesh rejects it with "has inward-pointing faces".
const PLANES = ((+1, "front", 0), (-1, "back", 8))

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
# Each entry: (name, back-face vertex quad, (n1,n2), grading triple)
# Grading is chosen so cells refine towards the tank wall in every near-wall
# block, accounting for which way the local axis points.
g = wall_grading
blocks = [
    ("core",  (0,1,2,3), (n_core_r, n_core_z), (1.0,   1.0,   1.0)),
    ("east",  (1,5,6,2), (n_rad,    n_core_z), (1/g,   1.0,   1.0)),  # x: core -> wall
    ("north", (3,2,6,7), (n_core_r, n_rad),    (1.0,   1/g,   1.0)),  # y: core -> wall
    ("south", (4,5,1,0), (n_core_r, n_rad),    (1.0,   g,     1.0)),  # y: wall -> core
]

# 3D coordinates of a global vertex id (0..7 front plane, 8..15 back plane)
function vertex_xyz(id)
    sgn, _, off = id < 8 ? PLANES[1] : PLANES[2]
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

println(io, "blocks")
println(io, "(")
for (name, q, (n1, n2), gr) in blocks
    v = (q[1], q[2], q[3], q[4], q[1]+8, q[2]+8, q[3]+8, q[4]+8)
    assert_right_handed(name, v)
    @printf(io, "    hex (%d %d %d %d %d %d %d %d) (%d %d 1) simpleGrading (%g %g %g)  // %s\n",
            v..., n1, n2, gr..., name)
end
println(io, ");\n")

# --- edges: elliptical arcs on the tank wall --------------------------------
# Arc interpolation points at the mid-angle of each wall segment.
arcs = [(4, 5, -(90 + split_deg)/2),   # O_bot -> O_se
        (5, 6,   0.0),                 # O_se  -> O_ne  (through the equator)
        (6, 7,  (90 + split_deg)/2)]   # O_ne  -> O_top

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
function hexverts(q)
    (q[1], q[2], q[3], q[4], q[1]+8, q[2]+8, q[3]+8, q[4]+8)
end
face(v, idx) = (v[idx[1]+1], v[idx[2]+1], v[idx[3]+1], v[idx[4]+1])

v_core  = hexverts((0,1,2,3))
v_east  = hexverts((1,5,6,2))
v_north = hexverts((3,2,6,7))
v_south = hexverts((4,5,1,0))

wall_faces = [face(v_east,  (1,2,6,5)),    # east  x-max -> arc
              face(v_north, (3,7,6,2)),    # north y-max -> arc
              face(v_south, (0,1,5,4))]    # south y-min -> arc

# The r = 0 faces of the core, north and south blocks. blockMesh does NOT
# collapse these into edges even though their vertex pairs are coincident: it
# emits them as genuine zero-area faces. Left unnamed they land in an
# auto-generated `defaultFaces` patch, so name them explicitly instead. They
# carry no flux (every kernel multiplies by area) and take an `Empty` BC in
# XCALibre, exactly like `frontAndBack` on a 2D-from-3D mesh.
axis_faces = [face(v_core,  (0,4,7,3)),    # core  x-min -> axis
              face(v_north, (0,4,7,3)),    # north x-min -> axis
              face(v_south, (0,4,7,3))]    # south x-min -> axis

# Vertices 0..7 are the front plane, so the block-local z-min face lies on
# wedgeFront and z-max on wedgeBack (see PLANES above).
front_faces = [face(v, (0,3,2,1)) for v in (v_core, v_east, v_north, v_south)]
back_faces  = [face(v, (4,5,6,7)) for v in (v_core, v_east, v_north, v_south)]

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
@printf("  cells             : %d  (core %dx%d, radial %d)\n",
        n_cells_total, n_core_r, n_core_z, n_rad)
@printf("  base cell size    : %.4f m, wall grading %.1f\n", h_base, wall_grading)
println()
println("Fill levels (z of the flat interface, for setField_Box!):")
println("  Both K-Site cases K(i) and K(ii) use 50% fill, i.e. z_fill = 0 exactly")
println("  (the ellipsoid is symmetric about the equator).")
for f in (0.25, 0.50, 0.90)
    target = 4/3*f - 2/3
    u = target
    for _ in 1:60; u -= (u - u^3/3 - target)/(1 - u^2); end
    @printf("  %2d%% by volume -> z_fill = %+.6f m\n", round(Int, f*100), u*b)
end
println("\nNext: run `blockMesh` here, then validate with")
println("  julia --project=. test/unit_test_ksite_wedge_mesh.jl")
