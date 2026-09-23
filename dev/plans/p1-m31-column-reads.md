# P1-M31 - column reads at element access sites (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R7, R13, R14. Governing decisions: D168, D170, D172, D173.

## Problem, quantified

Per-field mesh storage costs +22-33% first-run compile against a same-session AoS base (1t 16.6-18.0 s, 2D 20.1-21.4 s), for three container forms alike; an AoS control on the same source equals base; the rise is Julia inference (2D whole script 22.6 → 29.7 s), spread over mesh-carrying methods (`dev/telemetry/memory_scaling.md` § S7, § S9).

## Approach

Every `faces[i]`/`cells[i]`/`nodes[i]` on a container calls a user-defined method that the optimizer inlines at each site; AoS access is builtins only. Hot sites read columns instead: `(; area, normal) = faces` once per kernel or function, then `area[fID]` - `getfield` on the container plus plain `Array` indexing, the same builtins AoS uses. Containers, `Mesh2`/`Mesh3`, readers, parts and adapt stay as P1-M28-S7; element access still works for cold code (R14). A kernel keeps receiving the container (or the mesh) and destructures inside, so no kernel signature grows. Readers operating on plain vectors before the mesh exists are not touched.

## Configuration space

motorBike KOmega (3D, Int32, 1t/8t/GPU/MPI n=4) and 2D BFS KOmegaSST (Int64) through `~/.cache/xcal_m28/chain.sh`; compile compared only against `XENV=env_base` samples from the same chain run (base drifts ~10% within a day).

## Steps

- [ ] **P1-M31-S1** Discretise (generated discretisation, BC functors, apply_bcs), Calculate (gradient, limiters, interpolation, divergence, laplacian, orthogonality, snGrad) and `Mesh_1_functions` read columns - mechanism: no container method at a hot site - cost: none at runtime (same loads) - verdict: SCREEN, not a gate: 2D and 1t compile move toward base by a visible share (≥3 s of the ~6 s gap on 2D); if nothing moves, D172's mechanism is refuted and the milestone stops for a restatement.
- [ ] **P1-M31-S2** Solvers (SIMPLE, PISO, shared functions), turbulence models and wall functions, wall distance, ModelFramework read columns - mechanism: as S1 - verdict: M28 strict class (1t, 2D, MPI n=4 bitwise; 8t, GPU ≥8 figures); compile within +10% of same-session base on 1t and 2D, two samples each; 8t pinned 100-iteration within noise of 16.5 s.
- [ ] **P1-M31-S3** remaining runtime sites (other solvers, Postprocess, IO writers, Distribute runtime paths) - mechanism: as S1 - verdict: strict class; suite files reached, each a separate command.

## Exit criterion

S2's bar met and S3 landed; P1-M28-S5 re-scoped and P1-M28-S6 (close) runs next.

## Open questions

- Whether destructuring the container inside a GPU kernel keeps the same register/local-memory profile as element reads (check `__local_depot` once on the GPU run, per the kernel-argument-cost memory).
