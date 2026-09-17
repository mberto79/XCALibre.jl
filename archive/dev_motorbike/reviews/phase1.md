# Plan review: phase 1 — motorBike parity and stability
> Adversarial agent reviews this; the user may also override.

## Hidden assumptions
Assumptions the plan makes but never states. CONFIRM or reject (rejected becomes a decision below):
- [x] The copied mesh files are the only geometry/connectivity inputs used by both runs; hashes match for all five `polyMesh` files.
- [x] A quantitative 200-iteration bounded run is sufficient only after operator, aggregation, and determinism gates isolate the rapid explosion.
- [x] Existing uncommitted slip/symmetry edits are experiments and remain user-owned until focused tests justify them.

## Decisions needed
### D1: boundary parser shape
- [x] A (SOTA): tokenize the declared outer boundary list and brace-delimited patch dictionaries, validate required fields/counts, and ignore unknown values.
- [ ] B: skip only recognized `inGroups` line patterns (smaller but format-fragile).

### D2: slip discretisation evidence
- [x] A (SOTA): derive face and matrix behavior from slip/no-slip invariants, verify oblique normals, and aggregate wall functions by unique cell.
- [ ] B: mirror the closest OpenFOAM coefficient implementation directly (risks mismatched matrix conventions).

### D3: case parity
- [x] A (SOTA): make each field's inlet/outlet/slip/wall semantics equivalent and explicitly test backflow, moving ground, and potential initialization.
- [ ] B: retain the current approximations and adjust only relaxation (may mask a boundary defect).

## Out of scope (confirm)
- [x] Preserve patch-group membership or accept group names in Julia boundary assignment.
- [x] Match OpenFOAM's exact linear solvers or convergence trajectory.

## adversarial review outcome
- REVISE accepted: exploding cell 210839 touches three motorbike wall faces and no slip face; wall/no-slip behavior precedes relaxation tuning.
- The parser must enter only the declared outer `N (...)` list, validate exactly N complete patches, and handle comments/list syntax.
- `Wall` currently retains owner-cell normal velocity instead of imposing OpenFOAM `noSlip`; add full-vector oblique-normal tests.
- Wall production/constraint kernels write once per face, so multi-face wall cells race and overwrite; aggregate and constrain unique cells once.
- Add per-iteration extrema/argmax/patch flux, row inspection, frozen-turbulence, and cross-thread repeatability gates.
