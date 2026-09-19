# P1-M21 - owned-row single-copy linear system (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R7, R11. Governing decisions: D51, D67, D74, D76.

## Problem, quantified

Momentum holds `A0` and `A` as full CSRs including ghost rows (`ModelFramework_0_types.jl:193-194`), PETSc holds `mpiaij` plus the COO permutation arrays of `MatSetPreallocationCOO` (`XCALibrePETScExt.jl:145`): 3.5-4 copies for U, 2.5-3 for p. `psolve!` copies `x` in and out and `passemble!` copies `b` (`:201-216`): 12 vector copies per SIMPLE iteration. Six cell kernels run over ghosts (`discretise!` `:31`,`:116`; `green_gauss!`; `div!`; `inverse_diagonal!` `Solvers_0_functions.jl:102`; `H!` `:194`). Measured by P1-M16 at 4 mm n=2 (`dev/telemetry/memory_breakdown.md`): peak 1950 MB per rank = ~790 MB fixed + 1.85 KB per local cell; operator copies U 3.9, p 2.9; PETSc COO maps 99 MB of PETSc's 290 MB. `--heap-size-hint` is recorded as unusable (`dev/gotchas.md`).

## Approach

A single seam `n_rows(mesh)` (serial `length(mesh.cells)`, distributed `partition.n_owned`) feeds every row-range `ndrange` and the connectivity builder, so serial code paths do not change (D74). The CSR becomes `n_owned x n_local` on a `DistributedMesh`; owned ids are `<= n_owned` and ghost ids `> n_owned`, so the min/max canonical rows in `make_symmetric!` and `correct_mass_flux!` are owned by construction. PETSc vectors wrap the owned prefix. The COO map is measured before it is replaced. Shared-code cures (one sparsity per mesh, preallocated connectivity, no `adapt(CPU(), mesh)` per equation) go to the D67 PR with a list written here.

## Configuration space

mesh {5 mm, 4 mm} x ranks {1, 2, 8} x backend {CPU; CUDA n=1,2} x eqn {p, U, k, omega (SST)}; verdicts: residuals bitwise identical under Jacobi at every point, RSS and time from `dev/scripts/scaling_probe.jl` under `memguard.sh`.

## Steps

- [ ] **P1-M21-S1** `n_rows(mesh)` seam in `Solve` (identity) and `Distribute` (`n_owned`); `sparse_matrix_connectivity(::DistributedMesh)` emits rows `1:n_owned`; `_build_A` takes `(m, n)`; `SparseXCSR` verified to accept rectangular (`Multithread/spmvm.jl:4`, verify `size` and the CPU SpMV are never called distributed); the six kernels take their `ndrange` from `n_rows`; `residual`, `implicit_relaxation*`, `inverse_diagonal!`, `H!`, `update_equation!` follow - mechanism: ghost rows never existed in the operator - cost: none per iteration; fewer cells per kernel - verdict: suite green; residuals bitwise identical to `main`-of-branch at n=2 and n=8 (Jacobi); `check_ghosts` (M19-S2) zero.
- [-] **P1-M21-S2** ABSORBED BY P1-M16 (D91): COO maps are 34 percent of PETSc's heap, so S3 proceeds - measure the COO map footprint: PETSc `-memory_view` and `-malloc_view` at 5 mm n=2 before S3 - mechanism: measurement - cost: two runs - verdict: a number in `dev/telemetry/memory_breakdown.md`; if the map is under 15 percent of PETSc's per-rank memory, S3 is withdrawn and COO stays (it is PETSc's device-native assembly).
- [ ] **P1-M21-S3** (conditional on S2) host matrices switch to `MatCreateMPIAIJWithArrays` once plus `MatUpdateMPIAIJWithArray` per solve on a global-column copy of `colval` (`Int32`, one extra index array); device matrices keep COO - mechanism: PETSc's array path stores no permutation - cost: one nnz-length index array - verdict: RSS per rank falls by the S2 figure; residuals bitwise identical; both wrappers exist in PETSc.jl 0.4 (checked 2026-09-18).
- [ ] **P1-M21-S4** zero-copy vectors: `VecCreateMPIWithArray` on `x[1:n_owned]` and on `b` (`VecPlaceArray`/`VecResetArray` around each component's `bx`,`by`,`bz`); on CUDA the same through a hand-written `@ccall` to `VecCreateMPICUDAWithArray`/`VecCUDAPlaceArray` (not in the generated wrappers; symbols exist in any CUDA PETSc) - mechanism: owned entries are the contiguous prefix of every array and arrays are never reallocated - cost: none - verdict: `test_perf.jl` `psolve!` and `passemble!` allocations fall to the request overhead; residuals bitwise identical; GPU 5 mm n=1 per-iteration time recorded against 0.0746 s (`device_resident_petsc.md`).
- [ ] **P1-M21-S5** `run!` on a `DistributedMesh` calls `GC.gc(true)` after setup and before the first iteration; re-verify the `--heap-size-hint` claim by precompiling under the same flag (`julia --heap-size-hint=2G -e 'using Pkg; Pkg.precompile()'`) then launching ranks with it; document the per-rank recipe (node memory / ranks per node) in the guide and `dev/gotchas.md` - mechanism: the collector never sees sibling ranks - cost: one full collection - verdict: peak RSS per rank at 4 mm n=2 recorded before and after; the gotcha corrected or confirmed with its evidence.
- [ ] **P1-M21-S6** the D67 list, written to `dev/telemetry/memory_breakdown.md` § shared-code cures with the expected saving per item from M16's table: one `rowptr`/`colval` per mesh shared by U, p, k, omega, y; connectivity preallocated from `faces_range` sums with no `push!`/`sparsecsr` round trip; one host mesh copy per run instead of per equation; `A0` kept (needed per component) - mechanism: documentation for the separate PR - cost: none - verdict: the list exists with numbers.

## Exit criterion

Residuals bitwise identical under Jacobi at n=2 and n=8 CPU and n=1,2 GPU; per-rank peak RSS at 5 mm and 4 mm n=2 recorded against M16's table; per-iteration time within a 3 percent band of before (`equal_thermal.sh`); the D67 list written.

## Open questions

- S1: whether any BC `_extend_matrix` (periodic) adds entries for a ghost row; the colocation rule says no, `test_periodic.jl` at n=2,3 settles it.
- S4: whether PETSc.jl's `withlocalarray!` device branch is needed anywhere after zero-copy; if `VecCUDAPlaceArray` is missing from the CUDA PETSc's exports, fall back to `VecCUDAGetArray`+`copyto!` on device (one copy, still no host).
- S5: three `run!` calls in one process peak 890 MB above one (D90); find whether PETSc objects of earlier runs are ever destroyed before calling the extra garbage.
