# P1-M31 - flat column mesh and column reads (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R7, R13, R14. Governing decisions: D168, D170, D172-D179.

## Problem, quantified

Per-field mesh storage costs +22-33% first-run compile against a same-session AoS base (1t 16.6-18.0 s, 2D 20.1-21.4 s), for three container forms alike; an AoS control on the same source equals base; the rise is Julia inference (2D whole script 22.6 → 29.7 s), spread over mesh-carrying methods (`dev/telemetry/memory_scaling.md` § S7, § S9).

## Approach

Adopted by the user (D179). `Mesh2`/`Mesh3` hold every per-field array as a top-level field (`face_centre`, `face_area`, `cell_volume`, `node_coords`, ...); same-typed arrays share one of 9 type parameters (VV SVector3 arrays, VTF float arrays incl. `face_gDiff`, VR range arrays, VO owner pairs, VI, VS, VB, SV3, UR). `mesh.faces`/`cells`/`nodes` are rebuilt in `getproperty` as zero-copy `FaceArrays`/`CellArrays`/`NodeArrays` views, so readers, parts, Distribute and every `mesh.faces[i].area` site run unchanged (R14). Constructors keep their positional signature. `adapt` is written out per column with the element type carried over: a closure over `to` loses inference (D178). The nested S7 containers cost +22-33% compile (D176, D177); the flat struct compiles 25-35% below AoS.

Patches, both applying to 61751a58 and stacking in this order: `dev/archive/patches/p1-m31-flat-mesh-columns.diff` (S5), `dev/archive/patches/p1-m31-s1-column-reads.diff` (S6 starting point: scheme/BC signatures take `cells, faces`, `@define_boundary` binds `face`/`cell` only when a body names them, Calculate and boundary interpolation read columns).

## Configuration space

motorBike KOmega (3D, Int32, 1t/8t/GPU/MPI n=4) and 2D BFS KOmegaSST (Int64) through `~/.cache/xcal_m28/chain.sh`; compile compared only against `XENV=env_base` samples from the same chain run (base drifts ~10% within a day).

## Steps

Expected 4 steps (D175); restated to S5-S6 after the user adopted the flat layout (D179).

- [-] **P1-M31-S1** hot-path column reads - WITHDRAWN (D176): screen missed (2D +28%, 1t +25%, bitwise); carrying the columns unread costs +34-39%, so access rewrites cannot reach the cost. Diff: `dev/archive/patches/p1-m31-s1-column-reads.diff`.
- [-] **P1-M31-S2** column reads in solvers/turbulence - WITHDRAWN (D179): nested-container reads cannot move compile (D176); superseded by S5.
- [-] **P1-M31-S3** remaining column reads - WITHDRAWN (D179): as S2.
- [-] **P1-M31-S4** docs for column-read BCs - WITHDRAWN (D179): moves into S6 with the API change it documents.
- [x] **P1-M31-S5** LANDED (D180): apply `p1-m31-flat-mesh-columns.diff`; fix `test/distributed/test_offline.jl` (`getfield(mesh, :cells/:faces/:nodes)` no longer exists: compare via `getproperty` and assert the flat fields); update `dev/architecture.md` § mesh storage - mechanism: flat fields sharing type parameters (D177) - blast radius: every mesh consumer; readers, parts and Distribute unchanged by construction - verdict: strict class (1t, 2D, MPI n=4 with fresh parts bitwise; 8t, GPU ≥8 figures); compile ≤ same-chain AoS base +10% on 1t, 2D (two samples; measured −22-35%); GPU 20-iteration `run_s` within ±5% of base; 8t pinned 100-iteration within noise of 16.5 s; `test_offline.jl` + `test_partition.jl` at n=2,3; suite files `test_mesh_conversion.jl`, `unit_test_laplace.jl` via `suite_file.jl`.
- [x] **P1-M31-S7** LANDED (D182): type tags as one-element vectors (user request, D181): `get_float`/`get_int` become one-element arrays of the mesh float/integer type sharing `VTF`/`VI`, dropping the `SV3`/`UR` type parameters; constructors keep their signature, part format unchanged - mechanism: the fields only carry element types - blast radius: mesh construction, adapt, part write/read, float rebuild - verdict: strict class (1t, 2D bitwise; GPU ≥8 figures; MPI n=4 on existing parts bitwise); `test_mesh_conversion.jl`, `test_offline.jl`, `test_partition.jl`, `test_restart.jl` pass; compile not worse than S5 beyond noise.
- [x] **P1-M31-S6** LANDED (D185; bar restated from AoS base to S7 HEAD, the base gap moves to S8): discretisation path reads arrays directly: apply `p1-m31-s1-column-reads.diff` on S5, `_discretise_*_model!` read `cells.volume[i]`/`faces_range`, `_scheme!`/`scheme!`/BCs take `cells, faces`; docs BC page + CHANGELOG note the `scheme!`/BC signature change - mechanism: no element built per face in the GPU discretise kernels - verdict: GPU `gpu__discretise_scalar_model_` and `_vector_model_` per call ≤ base 3.50 / 4.94 ms (`dev/scripts/gpu_profile.jl`, base with `--project=env_base`), strict class, compile not worse than S5 beyond noise; docs build.
- [ ] **P1-M31-S8** kernel argument diet (D186): discretise kernels receive only the columns they read (cell faces/neighbours/nsign, cell volume and faces range, the face columns schemes touch) instead of `mesh`, and terms reach the kernel without a mesh - mechanism: by-value kernel arguments are copied to per-thread local memory, and the flat mesh doubled them (D184) - blast radius: every `discretise!` launch and the scheme/BC call chain - verdict: strict class; GPU discretise per call ≤ AoS base 3.50 / 4.95 ms; `__local_depot` ≤ base; compile not worse than S6 beyond noise.

## Exit criterion

S5, S6, S7 and S8 landed on their bars; then P1-M28-S6 (close) runs.

## Open questions

- Whether destructuring the container inside a GPU kernel keeps the same register/local-memory profile as element reads (check `__local_depot` once on the GPU run, per the kernel-argument-cost memory).
