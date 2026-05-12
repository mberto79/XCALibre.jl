# EFM Development Notes

## 2026-05-10 Film Solver Fixes

This session addressed incorrect thin-film predictions in the EFM test cases:
case 1 lacked rivulet formation, while cases 5 and 8 showed excessive narrowing
and wetting outside the expected film core.

Main causes found:

- The corrected conservative film flux could reverse prescribed Dirichlet inlet
  fluxes after pressure/capillary corrections, allowing local inlet backflow.
- The momentum equation was using the pressure/capillary-corrected film flux for
  convection. This fed correction fluxes back into `U` instead of using only the
  convective film flux.
- Face wetting was too permissive at fronts and edges because face-averaged
  wetting allowed fluxes to originate from dry cells.
- The previous lateral contact-line projection suppressed useful inlet/front
  behavior and acted like an ad hoc workaround for the boundary-flux issue.

Changes made in `Solvers_1_FilmModel.jl`:

- Split the momentum convection flux (`phif_U`) from the corrected conservative
  film flux (`phif`).
- Added local Dirichlet flux limiting so pressure/capillary corrections cannot
  reverse a fixed-height boundary flux.
- Applied donor-side wetting to advective and correction face fluxes.
- Added a separate capillary face wetting mask (`wcf`) based on local face
  connectivity.
- Restored the full contact-line force direction from the wetting gradient,
  with mesh-scale regularization rather than lateral projection.
- Computed film gravity/pressure geometry from local mesh normals instead of
  fixed plate-direction assumptions.

Observed checks after the changes:

- Case 1 developed rivulet segmentation downstream without the lateral
  projection workaround.
- Case 5 no longer showed sampled wet cells outside the inlet/core span, and the
  film narrowed more gradually.
- Case 8 retained a gradual contraction rather than collapsing too quickly.
- `julia --project=. -e 'using XCALibre'` and `git diff --check` passed.

Superseded follow-up:

- A geometry/alignment damping of the contact-line source on low-Weber,
  near-vertical, flow-parallel straight side edges was tested. It did not stop
  the case-5 downstream narrowing, so it should not be treated as the current
  solution.

## 2026-05-11 Contact-Line Kinematic Regime

The case-5 narrowing was traced to the static contact-angle source acting as a
persistent dewetting force on low-Weber, near-vertical side contact lines after
the leading transient had passed. The fix is now based on the contact-line
kinematics and the local surface orientation, not on global plate directions or
straight-edge detection.

Current model behavior:

- `∇w` is the wetting-gradient normal to the local contact line, pointing from
  dry cells into wet cells.
- `U ⋅ ∇w < 0` identifies an advancing contact line moving into dry substrate.
  In near-vertical, low-Weber regions, the static contact-line source is active
  only in this advancing state, so it does not keep pulling established
  flow-parallel side edges inward.
- The full static contact-angle source is retained when the local wall-normal
  component of gravity is significant. This preserves shallow-plate rivulet and
  dry-patch behavior such as case 1.
- The full source is also retained for locally inertia-dominated film motion
  (`We > 1`), preserving the high-flow narrowing behavior seen in case 8.

Generality check:

- The limiter uses only local vectors (`U`, `∇w`, and the local gravity
  decomposition), so it is independent of global `x/y/z` axes, inlet direction,
  side-edge straightness, and the CD1_2 plate shape.
- The first implementation of this change fixed the horizontal-surface fallback
  but still relied on generic volume finite-volume face normals for several
  film operators. That was correct for the flat/extruded Meredith-style cases,
  but it was not a sufficiently general surface formulation.

## 2026-05-11 Mesh2 Geometry Check

The EFM solver is currently a `Mesh2` surface solver. In this mesh type,
`cell.volume` is the 2D control-area measure, `face.area` is the edge length,
and `face.normal` is the in-surface edge-normal vector. Therefore the standard
XCALibre operators (`flux!`, `grad!`, `surface_gradient!`, and `div!`) already
use the correct surface finite-volume geometry for the present solver target.

Current implementation:

- The solver uses the existing XCALibre Mesh2 operators for advection,
  Green-Gauss gradients, surface-normal gradients, divergence, capillary
  corrections, and capillary time-step distances.
- Each cell still stores a local substrate normal reconstructed from its polygon
  nodes. This normal is used for gravity decomposition and tangent projection of
  film velocity and pressure-gradient corrections.
- The tangential gravity vector is computed as `g - (g ⋅ n)n` for each cell; if
  the tangential component is zero, no arbitrary fallback direction is
  introduced.
- Internal wetting boundary conditions are derived from the configured `h`
  boundary conditions, rather than hard-coded case-specific patch names.

Future Mesh3 integration note:

- A later generic `Mesh3` surface integration will need a first-class surface
  metric layer or operator overloads that pass explicit surface control areas,
  edge lengths, edge co-normals, and film-side substrate normals into the
  standard operators. The current solver should not carry duplicate local
  gradient/divergence implementations for the Mesh2 path.

Observed checks after the kinematic-regime and Mesh2 geometry changes:

- Case 5, 3000 iterations: established film width plateaus after the initial
  narrowing, with sampled widths around 0.503-0.516 m from `x = 0.2` to
  `x = 0.7`.
- Case 1, 3000 iterations: rivulet segmentation remains present, with five wet
  bands at `x = 0.2` and two at `x = 0.3` using a 2% height threshold.
- Case 8, 2000 iterations: downstream narrowing remains present, with sampled
  width reducing from about 0.488 m at `x = 0.1` to 0.459 m at `x = 0.3`.
- `julia --project=. -e 'using XCALibre'` and `git diff --check` passed.
