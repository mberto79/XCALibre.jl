# K-Site LH2 tank — axisymmetric wedge mesh

Mesh for the NASA K-Site liquid hydrogen tank self-pressurisation case
(`test/0_TEST_CASES/3d_LH2_ksite_selfpressurisation.jl`).

## Generate

```bash
cd examples/0_GRIDS/ksite_wedge
julia make_ksite_wedge.jl     # writes system/blockMeshDict
blockMesh                     # writes constant/polyMesh
```

Then validate, from the repository root:

```bash
julia --project=. test/unit_test_ksite_wedge_mesh.jl
```

The validation test skips cleanly if `constant/polyMesh` does not exist, so it
is safe to leave registered in `runtests.jl`.

## Geometry

NASA TM-103804 reports the tank as ellipsoidal with major:minor axis ratio 1.2,
major diameter 2.2 m, and volume 4.89 m³. **These three are mutually
inconsistent** — a pure ellipsoid with a = 1.1 m and ratio 1.2 gives 4.645 m³,
5 % short.

Anchoring on **volume and axis ratio** instead gives:

| | |
|---|---|
| equatorial semi-axis `a` | 1.118924 m |
| polar semi-axis `b` | 0.932436 m |
| major diameter | 2.2378 m (rounds to the reported 2.2 m) |
| volume | 4.8900 m³ (exactly as reported) |

So the 2.2 m figure is simply the rounded one. Volume is anchored because it
sets the ullage volume and therefore the self-pressurisation rate directly — a
5 % volume error would feed straight into dp/dt.

Set `anchor = :diameter` in `make_ksite_wedge.jl` to use a = 1.1 m exactly
instead (giving V = 4.645 m³).

## Fill levels

The interface height for a given fill fraction, from
`f = (u − u³/3 + 2/3)/(4/3)` with `u = z/b`:

| fill (by volume) | `z_fill` |
|---|---|
| 29 % | −0.268504 m |
| 49 % | −0.012433 m |
| 83 % | +0.443780 m |

## Topology

Butterfly (O-grid) in the (r,z) half-plane — a rectangular core plus north,
east and south blocks reaching the elliptical wall:

```
 z
 ^          . - O_top - .
 |       .   |    N     |  .
 |     O_ne__|__________|
 |     |     C3        C2  \
 |     |  E  |   CORE   |   |
 |     |     C0        C1   |
 |     O_se--|----------|
 |       .   |    S     |  .
 |          ` - O_bot - `
 +--------------------------> r
```

This keeps cell quality reasonable and confines the axis degeneracy to the
r = 0 block faces. Those faces have coincident vertex pairs, which blockMesh
collapses into prism cells — standard OpenFOAM wedge practice.

The wedge planes are written as `symmetry` patches rather than OpenFOAM
`wedge`, because XCALibre has no wedge type; the case assigns
`Symmetry(:wedgeFront)` / `Symmetry(:wedgeBack)`.

Patches: `tankWall` (wall), `wedgeFront`, `wedgeBack` (symmetry).

## Why the validation test matters

The collapsed axis faces are the one thing that could not be checked without
running blockMesh. A zero-area face carries no flux and is harmless, but a face
whose `delta` is *also* zero would produce 0/0 in every surface-normal-gradient
kernel in the multiphase solver. `unit_test_ksite_wedge_mesh.jl` asserts
`min(delta) > 0` and that all face metrics are finite, alongside volume, wall
area, patch names and the conditioning of the moment matrix that `reconstruct!`
inverts.

Analytical conditioning of a 5° wedge cell (checked before the mesh was built)
is comfortable — bulk cells ~5, on-axis cells ~22, and wall cells down to
Δr = 1e−4 m still under 1e3.

## Resolution

Controlled by `h_base` (target base cell size, default 0.02 m → ~3300 cells)
and `wall_grading` (last/first cell ratio in the near-wall blocks, default 5).

The default is a coarse starter mesh. The thermal boundary layer under a
3.5 W/m² wall flux is thin relative to a 1.1 m tank, so a converged run will
need considerably more wall-normal resolution — refine via `h_base` and
`wall_grading` and re-check the conditioning number reported by the validation
test.
