# P1-M13 - preconditioner guidance and defaults (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R5, R10. Governing decisions: D39, D40, D53, D56, D57.

## Problem, quantified

Users get three distributed pressure preconditioners (Jacobi, GAMG, BoomerAMG) with docstrings written from pre-D53 measurements, when PETSc's rtol was relative to the right-hand side and the pressure solves were looser. The docs do not say when each one fails, and two literature-backed defaults are missing.

## What the sources say (research 2026-09-18; PETSc 3.25 source, PETSc/hypre manuals, PETSc4FOAM PRACE WP294)

- AMG-CG takes 1-2 orders of magnitude fewer iterations than single-level PCs on 3D pressure (PETSc4FOAM, 64M cells: IC-CG 603, BoomerAMG-CG 35). Jacobi/IC iterations grow as about N^(1/3) in 3D; AMG iterations are close to mesh-independent.
- AMG setup does not pay off at low cells per rank; strong scaling degrades below about 20k cells per core (PETSc4FOAM).
- hypre HMIS coarsens each rank independently, and GAMG's MIS aggregation depends on the partition, so AMG iterates change with rank count. Jacobi's do not, beyond reduction order.
- GAMG in PETSc 3.25 defaults: threshold -1, reuse_interpolation true, agg_nsmooths 1, aggressive coarsening 1 level, coarse_eq_limit 50, smoother Chebyshev+Jacobi. The eigenvalue estimate uses CG only when MAT_SPD is set, and Chebyshev fails if that estimate is low. With reuse_interpolation, a re-setup redoes only PtAP (gamg.c:560).
- PCHYPRE has no partial reuse: every PCSetUp is a full BoomerAMG setup (ihypre.c:557), so freezing saves more for BoomerAMG than for GAMG.
- PETSc's BoomerAMG defaults differ from hypre's own: Falgout coarsening, classical interpolation, strong threshold 0.25, P_max 0 (unlimited). hypre recommends HMIS + ext+i with P_max 4 and a strong threshold of 0.5-0.6 for 3D. PETSc4FOAM's tuned pressure set is strong 0.7, HMIS, ext+i, P_max 2, agg_nl 2.
- No published rule for how often to rebuild; PETSc says reuse while changes are small.
- GPU: Jacobi and the GAMG solve phase run on the device, with GAMG setup partly on the host. BoomerAMG needs a GPU-built hypre, where PETSc switches to PMIS, ext+i and l1-Jacobi automatically.
- OpenFOAM GAMG is Galerkin with piecewise-constant P (unsmoothed pairwise aggregation), aggregates cached, coarse coefficients summed each solve.

## Approach

Literature first; XCALibre runs only where our framework could differ from what the sources say (D57). Defaults change only where a source recommends it AND one run on our case does not contradict it.

## Configuration space

mesh {BFS 5 mm 499,503 cells; BFS 4 mm 1,320,368 cells} x ranks {2, 8} x pc {jacobi, gamg, boomeramg} on CPU, plus GPU n=1 {jacobi, gamg}. One run each, 30 outer iterations, `dev/scripts/scaling_probe.jl`; metric per_iter and residuals at iteration 30.

## Steps

- [ ] **P1-M13-S1** declare MAT_SPD on matrices solved with GAMG or BoomerAMG, add `P_max=4` to the BoomerAMG defaults - mechanism: both PCs are documented SPD-only; the SPD flag selects PETSc's CG eigenvalue estimator; P_max bounds interpolation stencil growth with ext+i - cost: none per solve - verdict: accepted if the 5 mm n=2 per-iteration time and residual at 30 iterations are no worse than a 5% band.
- [ ] **P1-M13-S2** the configuration table above, post-D53 semantics - mechanism: measurement only - cost: about 14 runs - verdict: a table that confirms or contradicts each literature claim on our case.
- [ ] **P1-M13-S3** docstrings and the distributed guide gain a when-to-use / when-not table per preconditioner, and defaults follow S2 - mechanism: documentation - verdict: docs build with doctests green.

## Exit criterion

Telemetry for both meshes and both rank counts, per-preconditioner guidance in the docs, defaults justified by a source plus our run.

## Open questions

- Should freeze defaults change now that D53 makes the pressure solves tighter? Settled by S2.
