# P1-M27 — motorBike tutorial benchmarks run distributed (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R2, R8, R10. Governing decisions: D138.

## Problem, quantified

The motorBike RANS tutorial is the benchmark the user wants to run distributed, and three things block it. (1) Wall functions launch one kernel per patch, unlike every other boundary condition, which shares one kernel over all boundary faces and selects per face with `start <= fID <= stop`; a per-patch launch reads `BC.IDs_range[1]` before it computes `ndrange`, so a rank owning no face of that patch raises `BoundsError` on a 0-element range. Metis partitions the cell dual graph and knows nothing about patches, so this is reached at 6 and 8 ranks on the 353k-cell mesh (rank 2 `motorBike`=0 and rank 5 `lowerWall`=0 at n=6; rank 0 `lowerWall`=0 and ranks 6,7 `motorBike`=0 at n=8) and not at 1 or 2. (2) KOmegaSST is verified distributed only on the backward-facing step (restart bitwise at n=1,2,4, D132); it is unverified on a wall-function case with a curved body. (3) `potential_flow!` has never been run on a `DistributedMesh`; its rank-sensitive parts are `setReference!` and the `Dirichlet`-presence test that picks the reference.

## Approach

S1 removes the defect class rather than the four reported instances: one predicate guards every per-patch launch, so the launch is skipped instead of the range being indexed, and nothing collective is skipped because these generated wrappers only sequence local kernel launches. S2 and S3 are diagnosed from the run, not designed ahead of it — SST and `potential_flow!` already carry the distributed solve seam (`is_distributed_mesh`, `wrap_eqn`), so what remains is whatever the 8-rank traceback names, and each finding becomes its own step.

## Steps

- [x] **P1-M27-S1** LANDED (D139, D140): guard the six per-patch wall-function launches in `RANS_functions.jl` (`average_wall_cells!`, `set_production!`, both `correct_nut_wall!`, `fix_wall_row!`, `constrain!`) against an empty `IDs_range`. Mechanism: one `no_wall_faces` predicate, early return, no change to any range that is non-empty. Blast radius: KOmega and KOmegaSST wall-function patches only; serial has no empty patch, so serial output is unchanged. Cost: one predicate call per patch per call site. Verdict: serial regression test `unit_test_wall_function_empty_patch.jl` 4/4 and non-vacuous; motorBike RANS at n=6 completes 3 iterations (n=8 is not run unguarded on this machine, D144).
- [x] **P1-M27-S2** LANDED (D146): KOmegaSST with wall functions distributed, on the 10 mm backward-facing step rather than motorBike (user). Mechanism: `bfs_sst_wallfn_bcs` puts K/Omega/Nut wall functions on `:wall` and `:top`, which is the motorBike configuration, and a probe shows the mesh empties one of them on some rank at six ranks (rank 2 `top`=0) and eight (rank 2 `wall`=0, rank 3 `top`=0), so the same mesh gates S1 distributed. Verdict: `test_turbulence_sst_wallfn.jl` green at n=2 in 55 s; n=6 and n=8 pending the memory in D145.
- [ ] **P1-M27-S4** `_base_mesh` at the two remaining dimension tests a distributed mesh reaches (`Solvers_1_CSIMPLE.jl:347`, `Solvers_1_SIMPLE-MRF.jl:234`), where `typeof(mesh) <: Mesh3` skips the z-velocity convergence check on a distributed 3D run (D149). Blast radius: a convergence test, not a field. Verdict: distributed gate green.
- [x] **P1-M27-S3** LANDED (D141): `potential_flow!` solves through the distributed seam. Mechanism: it never called `wrap_eqn`, so every rank built a serial Krylov workspace and solved its own block; it now mirrors SIMPLE (`is_distributed_mesh` guard on preconditioner and workspace, `wrap_eqn`/`unwrap_eqn`, `petsc_options` forwarded, `update_preconditioner!` skipped), primes U ghosts before the first interpolation and syncs U after `reconstruct!`, whose ghost cells hold only part of their face list. `setReference!` needed nothing: the distributed method already maps a global cell id. Verdict: serial `test_potential_flow.jl` 501/501. Distributed, `test_potential_flow_mpi.jl` at n=2 first showed Phi correct to 1.1e-15 and the corrected flux to 1.7e-16 with U off by 0.70, which found D147 (`reconstruct!` took the 3D branch on a distributed 2D mesh) and D148 (the corrector loop pinned local cell 1 on every rank); both fixed, verdict is that test green at n=2,6.

## Known sites of the same class, out of this milestone's scope

`LES_functions.jl:70` (`get_normal` reads `faces[BC.IDs_range[1]].normal`, so it needs a value and not only a guard) and `LES_filters.jl:89`. Distributed LES is a documented gap in `dev/spec.md`, so these are recorded, not fixed here.

## Exit criterion

The motorBike RANS tutorial runs distributed with KOmegaSST and with `potential_flow!` initialisation at 1, 2, 6 and 8 ranks, rank-invariant at the Q1 bar, with the distributed gate green.

## Expected size

3 steps, plus one step per distinct defect S2 or S3 uncovers.
