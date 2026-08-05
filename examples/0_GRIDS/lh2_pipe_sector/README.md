# LH2 vertical heated pipe — 90° O-grid sector

Mesh for the forced-convection boiling cases of

> Tatsumoto, Shirai, Shiotsu, Hata, Naruo, Kobayasi & Inatani,
> *Forced convection heat transfer of saturated liquid hydrogen in
> vertically-mounted heated pipes*,
> AIP Conf. Proc. **1573**, 44–51 (2014). [doi:10.1063/1.4860681](https://doi.org/10.1063/1.4860681)

Used by [test/0_TEST_CASES/3d_LH2_pipe_forced_convection.jl](../../../test/0_TEST_CASES/3d_LH2_pipe_forced_convection.jl).

## Build

```bash
julia make_lh2_pipe_sector.jl     # writes system/blockMeshDict
./run_blockMesh.sh                # writes constant/polyMesh
julia --project=. ../../../test/unit_test_lh2_pipe_sector_mesh.jl
```

`run_blockMesh.sh` stages the case in `/tmp` first, because OpenFOAM's
`fileName` type cannot handle the spaces in this repository's path.

## Geometry

The paper's four tube heaters, selected with `CASE` at the top of the generator:

| `CASE` | D [mm] | L [mm] | L/D |
|---|---|---|---|
| `:D4_L100` | 4 | 100 | 25.0 |
| `:D4_L167` | 4 | 167 | 41.7 |
| `:D6_L150` | 6 | 150 | 25.0 |
| `:D6_L250` | 6 | 250 | 41.7 |

`L/D` is the parameter the paper is built around — it is what the DNB
correlation keys off (Eqs. 1–4), so the two ratios must both be reachable.

Liquid hydrogen flows **upward**, so the tube axis is `+z` and gravity is `-z`.
An unheated development length of 10 D precedes the heated section, matching the
paper's statement that the entrance lengths exceed 10 D. It is meshed as a
separate axial block so the wall splits into two patches:

| Patch | Type | Note |
|---|---|---|
| `inlet` | patch | z = 0 |
| `outlet` | patch | z = L_dev + L |
| `wallUnheated` | wall | development section, adiabatic |
| `pipeWall` | wall | **heated** section — carries `FixedHeatFlux` and the RPI model |
| `symmetryX` | symmetry | the x = 0 plane |
| `symmetryY` | symmetry | the y = 0 plane |

## Topology

A quarter of the standard five-block butterfly: one core block plus the two
outer blocks that reach the wall.

```
 y
 ^
 |  v6 . _
 |  |     ` .  v5
 |  v3-----v2  \        NORTH = (v3 v2 v5 v6)
 |  |      |    |       EAST  = (v1 v4 v5 v2)
 |  |CORE  |    |       CORE  = (v0 v1 v2 v3)
 |  v0-----v1---v4
 +----------------------> x
```

An O-grid rather than the wedge used for the K-Site tank, because a wedge
collapses onto the axis: the azimuthal direction then contributes nothing to the
least-squares moment matrix in `reconstruct!` and the whole reconstruction goes
rank-deficient. There is no cell edge on the centreline here, so that failure
mode does not exist.

A 90° sector assumes azimuthal symmetry. For a uniformly heated vertical tube in
upward flow that is a reasonable modelling choice, but it does mean the mesh
cannot represent circumferential asymmetry — including any azimuthal
non-uniformity in the onset of boiling.

## Near-wall resolution

The mesh targets **y+ = 30–50**, and the first cell height is *solved for*
rather than guessed:

1. `Re = ρ U D / μ`
2. Petukhov smooth-pipe friction factor `f = (0.790 ln Re − 1.64)^−2`
3. `u_τ = sqrt((f/8) U²)`
4. first cell **centre** at `y = y⁺ ν / u_τ`, so the cell is twice that tall
5. bisection for the `simpleGrading` ratio that puts a cell of exactly that
   height against the wall

This matters more than usual here. The RPI wall boiling model takes its
single-phase heat transfer coefficient from the thermal law of the wall, whose
logarithmic branch is only valid once the first cell centre is clear of the
buffer layer. Resolving to y+ ≈ 1 would put the first cell *inside* the viscous
sublayer, where that formula is not merely inaccurate but the wrong one.

Because an O-grid outer block is thicker on the axes than at the 45° diagonal,
the wall cell height — and hence y+ — varies around the circumference. The
generator reports the range and **warns if it leaves the 30–50 band**, which is
the signal to adjust `n_radial` or `core_frac`.

The sizing depends on the flow conditions, so `U_bulk`, `rho_l` and `mu_l` at
the top of the generator must match the case being run. The defaults are the
0.7 MPa saturated-liquid state taken from the same Helmholtz H₂ equation of
state that supplies the solver's property tables.

Example output for `:D6_L250` at 5.33 m/s:

```
 Reynolds number   : 259637
 u_tau             : 0.2296 m/s
 wall cell height  : 34.17 - 51.68 um
 achieved y+       : 31.8 - 48.2
 radial grading    : 0.35157
 total cells       : 119232
```

## Tuning

| Parameter | Effect |
|---|---|
| `core_frac` | Core half-width / radius. Must be < 0.707 or the core corner leaves the pipe. 0.4–0.5 gives the best cell quality and the narrowest y+ spread. |
| `n_radial` | Cells from core edge to wall. More cells ⇒ gentler grading for the same wall spacing. |
| `n_core` | Cells across the core block, and azimuthally in the outer blocks. |
| `n_axial_per_D` | Axial cells per diameter. |
| `y_plus_target` | Midpoint of the wanted band. |
