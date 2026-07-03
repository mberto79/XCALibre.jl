# Phase 8 — Float32, HYPRE extension, offline partitioning, I/O + docs (Medium)

Umbrella: `distributed_plan_detailed.md` Phase 8. Four independent workstreams; order by demand.

## 1. Float32
- All `Distribute` types are TF-generic by construction (Phases 1–2); activate the PETSc
  Float32 petsclib keyed on `_get_float(mesh)`.
- Test: cavity in F32 vs F64 to loose tolerance; halo exchange bitwise-consistent in F32.
- Mixed precision (F32 fields + F64 coarse solve) noted as future work (see AMG memory).

## 2. HYPRE extension
- `[weakdeps]` `HYPRE` → `ext/XCALibreHYPREExt.jl`; `HYPRESolver <: AbstractDistributedSolver`
  via the IJ assembler (`start_assemble!`/`assemble!`/`finish_assemble!`); BoomerAMG default
  for the pressure Poisson.
- Backend selection: `prun!(...; linear_backend=:petsc|:hypre)`; `SolverSetup` untouched.
- Adjoint caveat: no first-class transpose solve — SPD pressure Poisson is self-adjoint;
  error for non-symmetric adjoint requests through HYPRE.

## 3. Offline partitioning
- `partition_mesh(meshfile, nparts; dir)`: runs Phase 1 pipeline on one node, writes
  per-rank local meshes + Partition + procs as JLD2; `distribute(dir; comm)` loads in
  parallel (each rank reads its own file — no rank-0 memory bottleneck).
- Test: offline vs online produce identical `DistributedMesh` (hash the arrays) and
  identical solutions.

## 4. I/O + docs
- Parallel OpenFOAM decomposed-case writer (NO `.pvtu`/VTK route — user decision 2026-07-03):
  one `processor<rank>/` folder per rank, each rank writes its own mesh + fields
  independently (matches OpenFOAM's decomposePar layout, so ParaView/reconstructPar work).
  Main addition over the serial OpenFOAM writer: emit `procBoundary<rank>to<neighbour>`
  patches (type `processor`, `myProcNo`/`neighbProcNo`, owned side of each ProcessorPatch)
  in each rank's `constant/polyMesh/boundary`; check OpenFOAM's implementation online for
  the exact entry format. Ghost cells are NOT written (owned cells/faces only).
- `gather(field, dmesh)` utility → rank-0 global field in ORIGINAL cell ordering
  (via `orig_cells`) for postprocessing.
- Docs pages: workflow (§2.2 umbrella), cluster setup (MPIPreferences system binary,
  CUDA-aware env), CI notes, limitations (laminar-only v1, no cross-partition periodics,
  Windows serial-only, 1-layer halo ⇒ no least-squares gradients).

## Exit criteria
F32 cavity green; HYPRE pressure solve matches PETSc within tolerance; offline == online;
docs build (`julia --project=docs docs/make.jl`).
