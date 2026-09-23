# P1-M28 - mesh element arrays stored as structures of arrays

Milestone row: `dev/phaseRoadmap.md`. Source: `dev/archive/reviews/p1/memory-scaling-2026-09-23.md` § 3. Evidence: `dev/telemetry/memory_scaling.md`. Expected 6 steps (D159); restated at S4 by the user to 8 (D168); 9 after S7 (D171).

## mechanism

Hot kernels build a `Face3D`/`Cell` and read two or three fields inside one inlined scope, so the compiler drops unused loads; with the element array stored per field (StructArrays.jl) only the touched columns move from DRAM. `faces`, `cells` and `nodes` of `Mesh2`/`Mesh3` are wrapped once in the outer constructors, which every reader funnels through, so no kernel and no reader changes. Nested `SVector` fields stay packed per field. `boundaries` stays a plain vector. Derived per-face coefficients live as face fields computed by the face constructor, so a column costs the same as a separate array and cannot go stale (`face_gDiff` folds in after the layout lands, never before: under AoS it would widen every face read). `cell_nsign` holds only ±1 and becomes `Int8`.

## mechanism restatement (D168)

StructArrays.jl gave the bandwidth win but its type (one parameter per column plus names) inflates every mesh-carrying specialisation: compile +25-41% at S2, +51-67% at S4 against a +10% band. Two candidates, measured in order:
- B (S7): thin in-house containers `FaceArrays`/`CellArrays`/`NodeArrays <: AbstractVector{<:Face3D}` etc., columns grouped by type so the parameters are shared (e.g. one float-vector type for area/delta/weight, one SVector3 type for centre/normal/e), `getindex` builds the element, `setindex!` writes columns, `Adapt` per column, `.xdm` packs records. No call site changes; R14 kept. Adopted if first-run compile is within +10% of 875c16bb.
- A (S8, only if B misses): the mesh itself holds plain column arrays and every element access is rewritten (~250 sites). Short, informative accessor functions (e.g. `farea(mesh, fID)`) are allowed for readability if they cost at most 1-2% runtime against direct column indexing. R14 amended (D169).

## bars (ranked)

- STRICT (D163): residual histories and field hashes bitwise equal to `dev/telemetry/m28_baseline/` at 1t, 2D 1t and MPI n=4; 8t and GPU to at least 8 significant figures (their own rerun noise is 10.5); `mesh.faces[i].area` style access still works (R14).
- Banded: 1t 500-iteration time ≤ 162.9 s ±5%; first `run!` compile time within +10% of 875c16bb (D168; the D164 band was provisional and is withdrawn); equation `mesh_temp.faces` stays a StructArray (a materialised AoS copy per equation would OOM this box).
- Objective, read at close: 8-rank MPI clearly below 69.7 s, 8t below 78.0 s, GPU below 22.4 s, `B` falls with `C` roughly unchanged; isolated face interpolation ≥5x at 8t and cell gather ≥3.5x on the real mesh object.

## gate (per step, each command under five minutes, D160)

- 20-iteration motorBike smoke at 1t and 8t, residual text compared with the S1 baseline; one 2D example 20 iterations (exercises `Face2D`/`Mesh2`); GPU 20 iterations; MPI n=4 20 iterations with freshly generated parts (`motorbike_smoke.jl part`); first `run!` time from the `.time` files. All via `dev/scripts/motorbike_smoke.jl`, compared with `dev/scripts/cmpres.jl`.
- Named unit test files the step reaches, each a separate command. Full serial suite and 500-iteration per-point timings run once, at milestone close, one command per point.

## steps

- [x] P1-M28-S1 baselines at 875c16bb - `dev/telemetry/m28_baseline/`, reproducibility and compile times in `dev/telemetry/memory_scaling.md` (D163); footprint table deferred to S6 with the after table.
- [x] P1-M28-S2 StructArray wrap in the Mesh2/Mesh3 inner constructors, `.xdm` writer packs columns, VTK host copies via `adapt`, `_get_backend` off element arrays (D166); strict class passed, 8t −13%, 1t −11% per 100 iterations (D165), compile +25-41% accepted with attribution (D164); 12 suite files green, petscenv envs resolved.
- [x] P1-M28-S3 `test_offline.jl` asserts parts read back per field and equal online parts (2/2 at n=2,3); gate 5m04s, 4 s over Q2 (D167).
- [x] P1-M28-S4 `cell_nsign` Int8 via the mesh constructor, own type parameter `VS`, `.xdm` format 4; strict class bitwise, suite files and offline/partition tests green; compile rose further, triggering D168.
- [x] P1-M28-S7 option B: in-house `FaceArrays`/`CellArrays`/`NodeArrays` replace StructArrays (dependency dropped), same-typed columns share a parameter; strict class holds, 1t 20-iteration run −11% vs same-session base; REFUSED as the compile fix (+22% 1t, +28% 2D same-session, equal to S4; AoS control equals base; rise is inference) and kept as the storage S9 builds on (D170).
- [ ] P1-M28-S9 lazy element: `getindex` on the containers returns a view (container, index) whose `getproperty` reads one column and `setproperty!` writes one; `convert` to the plain element for writers and copies; element-typed signatures (`cell::Cell{TF}` in BCs and `_scheme_source!`) take the view. Blast radius: every element consumer. Bar: strict class; compile within +10% of same-session base on 1t and 2D, two samples each; 8t pinned 100-iteration within noise of S2 (16.5 s). One build, one measurement; refused → S8 (D171).
- [ ] P1-M28-S8 option A, ONLY if S7 is refused: whole mesh as plain column arrays, every element access rewritten, optional short accessors (≤1-2% runtime vs direct indexing, measured on the 8t pinned 100-iteration run); R14 amended (D169). Expect to split into its own milestone if it exceeds three steps.
- [ ] P1-M28-S5 (after S7/S8 settle the container) fold `face_gDiff` into `Face2D`/`Face3D` as last field with a positional constructor that derives it; drop the `Mesh2`/`Mesh3` field and type parameter and `update_face_gDiff!`; Laplacian reads `face.gDiff`; `_XDM_FORMAT` bumped. Any 9-argument (full-field) face constructor must RECOMPUTE gDiff, because `@set face.x = ...` in BlockMesher2D rebuilds through it and would otherwise keep a stale value. Blast radius: Laplacian, readers, parts. Bar: strict class, 8t time within noise of S4.
- [ ] P1-M28-S6 close: 500-iteration per-point timings (1t, 8t, n=1, n=8, GPU), refit `C`/`B`, footprint after, main-thread 8t profile of 100 iterations, GPU workgroup re-sweep; follow-ups (face-based assembly, extra face coefficients) decided only if a kernel still scales below 4x at 8t. Full serial suite by file. Blast radius: none shipped.

## pitfalls

- A whole element passed to a `@noinline` function or stored in a heap container forces every column to load.
- Kernel closures capturing `mesh` copy it per thread; pass the StructArrays as kernel arguments.
- Index arithmetic must stay in `TI` (`xcalibre-mesh-types` skill); a promotion to Int64 doubles the bytes again.
- `mesh_*.jld2` and `parts_*/` caches die with the type change; regenerate, never compare against them.
