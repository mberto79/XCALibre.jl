# Phase 7 — GPU-native distributed validation, parallel I/O, Float32 (High)

Reprioritised 2026-07-03: this phase makes the distributed functionality *usable*
(GPU path proven natively, results writable, F32 available). AD moved to Phase 8 as an
optional add-on. Order: 1 → 2 → 3.

## 1. GPU-native validation (local machine)
Phase 6 validated the GPU field path only via `solve_on=CPU()` host staging — a stopgap,
NOT a supported/endorsed configuration. This deliverable retires it as the tested path.

- Prereqs (user toolchain, see `dev/gpu_native_setup.md`): system CUDA-aware MPI
  (`MPI.has_cuda() == true`) + system PETSc built `--with-cuda` against that MPI
  (`PetscHasExternalPackage(petsclib, "cuda") == true`), wired into `dev/petscenv` via
  MPIPreferences + `JULIA_PETSC_LIBRARY`.
- Verify the lab-deferred Phase 6 code paths on the local RTX 4070 (ranks share 1 GPU —
  correctness only, not scaling):
  - `mpiaijcusparse` MatConvert path (written, unverified): device solves end-to-end.
  - CUDA-aware halo exchange (auto path with `MPI.has_cuda()==true`) vs forced
    `cuda_aware=false` staging — results identical.
- Update `test/distributed/test_gpu.jl`: when PETSc has CUDA, run the cavity gate
  natively (no `solve_on`); keep the hard-error section conditional (auto-skips).
- Fix whatever the native path shakes out (Vec types, value-update path after MatConvert,
  option handling).

## 2. Parallel I/O (OpenFOAM decomposed-case writer)
- The`.pvtu`/VTK route should now be implemented only after "offline partitioning" in phase 8 — user decision 2026-07-03 modified manually directly in this sentence. You will prioritiese one `processor<rank>/` folder per
  rank, each rank writes its own mesh + fields independently (matches OpenFOAM's
  decomposePar layout, so ParaView/reconstructPar work).
- Main addition over the serial OpenFOAM writer: emit `procBoundary<rank>to<neighbour>`
  patches (type `processor`, `myProcNo`/`neighbProcNo`, owned side of each
  ProcessorPatch) in each rank's `constant/polyMesh/boundary`; check OpenFOAM's exact
  entry format online. Ghost cells/faces are NOT written (owned only).
- Wire into psimple!/ppiso! via `write_interval` (currently ignored in the loops).
- `gather(field, dmesh)` utility → rank-0 global field in ORIGINAL cell ordering
  (via `orig_cells`) for postprocessing.
- Test: write a partitioned case, reconstructPar (or field-level compare vs serial
  writer output through orig ids).

## 3. Float32
- All `Distribute` types are TF-generic by construction (Phases 1–2); activate the PETSc
  Float32 petsclib keyed on `_get_float(mesh)`.
- Test: cavity in F32 vs F64 to loose tolerance; halo exchange bitwise-consistent in F32.
- Mixed precision (F32 fields + F64 coarse solve) noted as future work (see AMG memory).

## Exit criteria
Native GPU cavity gate green n=1,2 (no solve_on), CUDA-aware halo == staged halo;
decomposed case opens in ParaView / reconstructs; F32 cavity green.
