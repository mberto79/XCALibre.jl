# P1-M31 - column reads at element access sites (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R7, R13, R14. Governing decisions: D168, D170, D172-D175.

## Problem, quantified

Per-field mesh storage costs +22-33% first-run compile against a same-session AoS base (1t 16.6-18.0 s, 2D 20.1-21.4 s), for three container forms alike; an AoS control on the same source equals base; the rise is Julia inference (2D whole script 22.6 → 29.7 s), spread over mesh-carrying methods (`dev/telemetry/memory_scaling.md` § S7, § S9).

## Approach

Option A in full (D175). Scheme and BC functions receive the `faces`/`cells` containers plus `fID`/`cellID` instead of whole elements and read columns (`faces.area[fID]`); kernels read `cells.volume[i]` etc. The float type comes from the container's element type. `@define_boundary` scans its body and binds `face = faces[fID]` / `cell = cells[cellID]` only if the body names them, so unconverted and user BCs still work (at the old compile cost). Containers, `Mesh2`/`Mesh3`, readers, parts and adapt stay as P1-M28-S7; `mesh.faces[i].area` still works for cold code (R14). Readers on plain vectors are not touched.

## Configuration space

motorBike KOmega (3D, Int32, 1t/8t/GPU/MPI n=4) and 2D BFS KOmegaSST (Int64) through `~/.cache/xcal_m28/chain.sh`; compile compared only against `XENV=env_base` samples from the same chain run (base drifts ~10% within a day).

## Steps

Expected 4 steps (D175).

- [ ] **P1-M31-S1** Discretise kernels, `_scheme!`/`_scheme_source!` and every `scheme!`, BC signatures (macro with conditional binding, `Discretise_3` functors), the BCs the gate cases use, Calculate kernels and `Mesh_1_functions` read columns - mechanism: per-access IR is one array load per field read (D174) - cost: none at runtime - verdict: SCREEN: 2D and 1t compile close ≥3 s of the ~6 s 2D gap against same-chain base; strict class bitwise at 1t and 2D; if compile does not move, stop and report to the user.
- [ ] **P1-M31-S2** Solvers (SIMPLE, PISO, shared functions), turbulence models, wall functions, wall distance, ModelFramework - verdict: P1-M28 strict class (1t, 2D, MPI n=4 bitwise; 8t, GPU ≥8 figures); compile within +10% of same-chain base on 1t and 2D, two samples each; 8t pinned 100-iteration within noise of 16.5 s.
- [ ] **P1-M31-S3** remaining BC bodies, other solvers (CSIMPLE, Godunov, film, multiphase, MRF), Postprocess, IO, Distribute runtime paths - verdict: strict class; suite files reached, each a separate command.
- [ ] **P1-M31-S4** docs (BC-definition page: column reads preferred, element binding still accepted) and CHANGELOG - verdict: docs build.

## Exit criterion

S2's bar met and S3 landed; P1-M28-S5 re-scoped and P1-M28-S6 (close) runs next.

## Open questions

- Whether destructuring the container inside a GPU kernel keeps the same register/local-memory profile as element reads (check `__local_depot` once on the GPU run, per the kernel-argument-cost memory).
