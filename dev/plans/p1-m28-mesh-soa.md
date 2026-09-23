# P1-M28 - mesh element arrays stored as structures of arrays

Milestone row: `dev/phaseRoadmap.md`. Source: `dev/archive/reviews/p1/memory-scaling-2026-09-23.md` § 3. Evidence: `dev/telemetry/memory_scaling.md`. Expected 6 steps (D159).

## mechanism

Hot kernels build a `Face3D`/`Cell` and read two or three fields inside one inlined scope, so the compiler drops unused loads; with the element array stored per field (StructArrays.jl) only the touched columns move from DRAM. `faces`, `cells` and `nodes` of `Mesh2`/`Mesh3` are wrapped once in the outer constructors, which every reader funnels through, so no kernel and no reader changes. Nested `SVector` fields stay packed per field. `boundaries` stays a plain vector. Derived per-face coefficients live as face fields computed by the face constructor, so a column costs the same as a separate array and cannot go stale (`face_gDiff` folds in after the layout lands, never before: under AoS it would widen every face read). `cell_nsign` holds only ±1 and becomes `Int8`.

## bars (ranked)

- STRICT: CPU residual histories bitwise equal to the 875c16bb baseline at a fixed thread count (1t, 8t, 2D case, n=8 MPI); GPU residuals within tolerance of its baseline (reduction order, D116); `mesh.faces[i].area` style access still works (R14).
- Banded: 1t 500-iteration time ≤ 162.9 s ±5%; first `run!` compile time ≤ +10%; equation `mesh_temp.faces` stays a StructArray (a materialised AoS copy per equation would OOM this box).
- Objective, read at close: 8-rank MPI clearly below 69.7 s, 8t below 78.0 s, GPU below 22.4 s, `B` falls with `C` roughly unchanged; isolated face interpolation ≥5x at 8t and cell gather ≥3.5x on the real mesh object.

## gate (per step, each command under five minutes, D160)

- 20-iteration motorBike smoke at 1t and 8t, residual text compared with the S1 baseline; one 2D example 20 iterations (exercises `Face2D`/`Mesh2`); GPU `motorBike_gpu.jl` 20 iterations; n=8 MPI 20 iterations with freshly generated parts; `@time` of the first `run!`.
- Named unit test files the step reaches, each a separate command. Full serial suite and 500-iteration per-point timings run once, at milestone close, one command per point.

## steps

- [ ] P1-M28-S1 baselines at 875c16bb: smoke script in `dev/scripts/`, 20-iteration residual files (1t, 8t, 2D, GPU, n=8), compile time, footprint table. Blast radius: none shipped. Bar: files written, reruns reproduce bitwise on CPU.
- [ ] P1-M28-S2 StructArrays dependency with compat, `_soa` wrap of `faces`, `cells`, `nodes` in the `Mesh2`/`Mesh3` outer constructors, and whatever the wrap breaks (plain-`Vector` assumptions, `similar`, `collect`, `pointer`/`reinterpret`/`unsafe_wrap`/`sizeof` on mesh arrays; `adapt(CPU(), mesh)` must keep a StructArray). `Pkg.resolve()` every `dev/petscenv*` env. Blast radius: every mesh consumer. Bar: full strict class plus 1t/8t 100-iteration quick timing.
- [ ] P1-M28-S3 `.xdm` writer writes a StructArray column-safe (`collect` per rank part); write/read round trip compares every column bit for bit. Format unchanged (same bytes). Blast radius: distributed parts. Bar: round trip exact, n=8 smoke bitwise.
- [ ] P1-M28-S4 `cell_nsign` as `Int8` in every reader's connectivity step, `extract_subdomain` and the `.xdm` reader/writer; its own mesh type parameter; `_XDM_FORMAT` bumped. Check integer arithmetic on `ns`. Blast radius: every gather loop. Bar: strict class.
- [ ] P1-M28-S5 fold `face_gDiff` into `Face2D`/`Face3D` as last field with a positional constructor that derives it; drop the `Mesh2`/`Mesh3` field and type parameter and `update_face_gDiff!`; Laplacian reads `face.gDiff`; `_XDM_FORMAT` bumped. Blast radius: Laplacian, readers, parts. Bar: strict class, 8t time within noise of S4.
- [ ] P1-M28-S6 close: 500-iteration per-point timings (1t, 8t, n=1, n=8, GPU), refit `C`/`B`, footprint after, main-thread 8t profile of 100 iterations, GPU workgroup re-sweep; follow-ups (face-based assembly, extra face coefficients) decided only if a kernel still scales below 4x at 8t. Full serial suite by file. Blast radius: none shipped.

## pitfalls

- A whole element passed to a `@noinline` function or stored in a heap container forces every column to load.
- Kernel closures capturing `mesh` copy it per thread; pass the StructArrays as kernel arguments.
- Index arithmetic must stay in `TI` (`xcalibre-mesh-types` skill); a promotion to Int64 doubles the bytes again.
- `mesh_*.jld2` and `parts_*/` caches die with the type change; regenerate, never compare against them.
