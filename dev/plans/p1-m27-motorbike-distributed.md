# P1-M27 — motorBike tutorial benchmarks run distributed (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R2, R8, R10. Governing decisions: D138.

## Problem, quantified

The motorBike RANS tutorial is the benchmark the user wants to run distributed, and three things block it. (1) Wall functions launch one kernel per patch, unlike every other boundary condition, which shares one kernel over all boundary faces and selects per face with `start <= fID <= stop`; a per-patch launch reads `BC.IDs_range[1]` before it computes `ndrange`, so a rank owning no face of that patch raises `BoundsError` on a 0-element range. Metis partitions the cell dual graph and knows nothing about patches, so this is reached at 6 and 8 ranks on the 353k-cell mesh (rank 2 `motorBike`=0 and rank 5 `lowerWall`=0 at n=6; rank 0 `lowerWall`=0 and ranks 6,7 `motorBike`=0 at n=8) and not at 1 or 2. (2) KOmegaSST is verified distributed only on the backward-facing step (restart bitwise at n=1,2,4, D132); it is unverified on a wall-function case with a curved body. (3) `potential_flow!` has never been run on a `DistributedMesh`; its rank-sensitive parts are `setReference!` and the `Dirichlet`-presence test that picks the reference.

## Approach

S1 removes the defect class rather than the four reported instances: one predicate guards every per-patch launch, so the launch is skipped instead of the range being indexed, and nothing collective is skipped because these generated wrappers only sequence local kernel launches. S2 and S3 are diagnosed from the run, not designed ahead of it — SST and `potential_flow!` already carry the distributed solve seam (`is_distributed_mesh`, `wrap_eqn`), so what remains is whatever the 8-rank traceback names, and each finding becomes its own step.

## Steps

- [ ] **P1-M27-S1** Guard the six per-patch wall-function launches in `RANS_functions.jl` (`average_wall_cells!`, `set_production!`, both `correct_nut_wall!`, `fix_wall_row!`, `constrain!`) against an empty `IDs_range`. Mechanism: one `no_wall_faces` predicate, early return, no change to any range that is non-empty. Blast radius: KOmega and KOmegaSST wall-function patches only; serial has no empty patch, so serial output is unchanged. Cost: one predicate call per patch per call site. Verdict: motorBike RANS at n=6 and n=8 completes 3 iterations; serial residual hashes bitwise unchanged; the distributed gate green.
- [ ] **P1-M27-S2** KOmegaSST on motorBike distributed. Mechanism: named after the n=8 run's first failure; the step is not written until S1's run has produced one. Verdict: SST motorBike runs at n=1,2,6,8 with rank-invariant residuals at the Q1 bar.
- [ ] **P1-M27-S3** `potential_flow!` on a `DistributedMesh`. Mechanism: audit `setReference!` (it must pin one global cell, not cell 1 on every rank) and the `has_fixed_potential` test for rank uniformity, then whatever the run names. Verdict: potential-flow initialisation of motorBike at n=1,2,6,8, rank-invariant at the Q1 bar and matching the serial field.

## Known sites of the same class, out of this milestone's scope

`LES_functions.jl:70` (`get_normal` reads `faces[BC.IDs_range[1]].normal`, so it needs a value and not only a guard) and `LES_filters.jl:89`. Distributed LES is a documented gap in `dev/spec.md`, so these are recorded, not fixed here.

## Exit criterion

The motorBike RANS tutorial runs distributed with KOmegaSST and with `potential_flow!` initialisation at 1, 2, 6 and 8 ranks, rank-invariant at the Q1 bar, with the distributed gate green.

## Expected size

3 steps, plus one step per distinct defect S2 or S3 uncovers.
