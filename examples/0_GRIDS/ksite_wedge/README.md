# K-Site LH2 tank — axisymmetric wedge mesh

Mesh for the NASA K-Site liquid hydrogen tank self-pressurisation case
(`test/0_TEST_CASES/3d_LH2_ksite_selfpressurisation.jl`).

## Generate

```bash
cd examples/0_GRIDS/ksite_wedge
julia make_ksite_wedge.jl     # writes system/blockMeshDict
blockMesh                     # writes constant/polyMesh
```

OpenFOAM does not accept paths containing spaces; from such a path use
`./run_blockMesh.sh`, which meshes in a temporary directory and copies
`constant/polyMesh` back.

Then validate, from the repository root:

```bash
julia --project=. test/unit_test_ksite_wedge_mesh.jl
```

The validation test skips cleanly if `constant/polyMesh` does not exist, so it
is safe to leave registered in `runtests.jl`.

## Geometry

Fernandes et al. (2026), Sec. 2, give the tank as approximately ellipsoidal
with a major diameter of 2.20 m and a minor diameter of 1.93 m. Taking both at
face value reproduces the paper's other two figures:

| | |
|---|---|
| equatorial semi-axis `a` | 1.100 m |
| polar semi-axis `b` | 0.965 m |
| volume | 4.891 m³ (paper: 4.89 m³) |
| surface area | 13.974 m² (paper: 13.98 m²) |

Surface area matters as much as volume: the K-Site cases are specified by wall
heat flux, so it sets the total heat input (3.5 W/m² × 13.97 m² = 48.9 W against
the paper's 49.0 W for K(i)).

## Topology

An O-grid in the (r,z) half-plane — a rectangular core plus north, east and
south blocks reaching the elliptical wall — with the core and east regions each
split into three rows, so that an **equatorial band** of flat, uniform-height
layers runs from the axis to the wall:

```
 z ^
   |  O_top
   |    |`-.
   |    |    `-.     N
   |  C3+---------+C2 `-.
   |    | core_hi |  E_hi `O_ne
   |A_hi+---------+B_hi-----+W_hi     z = +band_half  -.
   |    |core_band|  E_band |                           } flat, uniform
   |A_lo+---------+B_lo-----+W_lo     z = -band_half  -'  layers; a face
   |    | core_lo |  E_lo .O_se                           lies on z = 0
   |  C0+---------+C1 .-'
   |    |    .-'     S
   |    |.-'
   |  O_bot
   +-------------------------------> r
```

### Why the band

Both K-Site cases are 50 % full, so the interface sits at z = 0. In the plain
O-grid the east block's layers fanned out towards the wall, and the cell layer
straddling z = 0 was 2.0 cm tall at the axis but 4.0 cm at the wall. A VOF
interface smeared across cells whose height varies along it gives a density
field that is not a function of z alone, so it cannot be hydrostatic.

Measured on that mesh with no heat flux and no phase change, this drove a
spurious inward current of up to 0.34 m/s along the interface of a tank that
should have stayed at rest. The `p_rgh` offset the mechanism predicts,
`g·h·(ρ_l − ρ_m)`, matched the solution at every radius. The full account is in
the header of `make_ksite_wedge.jl`.

Inside the band every layer is horizontal with the same height at every radius,
and the layer count is even so a face lies exactly on z = 0. The band is centred
on the equator and therefore serves 50 % fill only; the generator prints the
interface height for other fills, which lie outside it.

The r = 0 block faces have coincident vertex pairs, which blockMesh collapses
into prism cells — standard OpenFOAM wedge practice.

The wedge planes are written as `symmetry` patches rather than OpenFOAM
`wedge`, because XCALibre has no wedge type; the case assigns
`Symmetry(:wedgeFront)` / `Symmetry(:wedgeBack)`.

Patches: `tankWall` (wall), `wedgeFront`, `wedgeBack` (symmetry), `axis` (empty).

## Why the validation test matters

The collapsed axis faces are one thing that cannot be checked without running
blockMesh. A zero-area face carries no flux and is harmless, but a face whose
`delta` is *also* zero would produce 0/0 in every surface-normal-gradient kernel
in the multiphase solver. `unit_test_ksite_wedge_mesh.jl` asserts
`min(delta) > 0` and that all face metrics are finite, alongside volume, wall
area, patch names, the reconstruction of a uniform field, and that the layers
around z = 0 are flat from the axis to the wall.

The generator itself checks that every block is right-handed and that every
edge shared by two blocks carries the same cell count and grading, since it is
often edited where blockMesh is not available.

## Resolution

| parameter | default | controls |
|---|---|---|
| `h_base` | 0.02 m | target base cell size |
| `wall_grading` | 5 | last/first cell ratio across the near-wall blocks |
| `band_half` | 0.10 m | half-height of the equatorial band |
| `h_band` | `h_base` | layer height inside the band (rounded to an even count) |

The default is a coarse starter mesh. The thermal boundary layer under a
3.5 W/m² wall flux is thin relative to a 1.1 m tank, so a converged run will
need considerably more wall-normal resolution. `band_half` must stay large
enough to contain the interface for the whole run; an upper bound on the K(i)
level drop, if all 49 W went into boil-off for 17.5 hr, is about 2.6 cm.
