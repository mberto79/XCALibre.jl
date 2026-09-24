# P1-M34 - robustness before merge (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R3, R8, R13. Governing decisions: D197, D208. Source: `dev/archive/reviews/p1/pre-merge-review-2026-09-24.md` § Crash / wrong result (A1-A5).

## Problem, quantified

- A1 (confirmed, D208): `Threads.@threads :static` in `src/Multithread/xvector.jl` `_foreach_chunk`/`_reduce_chunks` throws "`@threads :static` cannot be used concurrently or nested" when a CPU solve touching ≥ 2^16 elements runs inside another `@threads` loop or concurrently from `@spawn` tasks; main used `@sync/@spawn` and did not. Repro: nested `Krylov.kdot` on a 200k `XVector` at `-t 4` (scratchpad `verify_static.jl` of the review session; rewrite as a test).
- A2: `_setup(::CPU, ::AutoTune, n)` in `src/Multithread/Multithread.jl:24` gives workgroup `cld(0, nthreads) = 0` on an empty range → KernelAbstractions `DivideError`; also on main. Check `_dynamic_setup`/`_sized` for the same.
- A3: distributed error paths hang: rank-0 throw in `distribute(reader; dir)` / `distribute(mesh)` (`src/Distribute/Distribute_1_partition.jl:424-438`) leaves ranks in Barrier/recv; `restart_fields!` checks `isfile` per rank (`Distribute_7_io.jl:342`); `FOAMCase` read failure before `Allgather` (`Distribute_8_foam.jl:39,43`), stale `cellProcAddressing` length unchecked.
- A4: `_parts_match` (`Distribute_1_partition.jl:463`) compares only the rank count, so parts of another mesh or `scale` in `dir` are reused silently (`examples/3D_BFS_mpi.jl` writes `./parts`).
- A5: Int32 counters past 2^31 entries fail with the wrong error: `src/FoamMesh/FoamMesh_2_connect.jl:36-43,76-89,123-129,210-217` (`connect_cell_nodes` wraps negative at ~89M hex cells), ASCII `read_faces` in `FoamMesh_1_read.jl`; `src/UNV3/UNV3_2_builder.jl:213-233,296-303,333` has a wrapping `@inbounds` cursor (out-of-bounds write).

## Approach

Each fix states the invariant it restores; no new mechanism beyond these. A3 uses one pattern everywhere: the rank that can fail wraps the work in try/catch, then an `MPI.Bcast`/`Allreduce` of an ok flag, and every rank throws the same error. A4 adds a fingerprint to the part header (mesh counts + a hash of the reader's inputs where available, plus TI/TF), which bumps the part format to 6; `_parts_match` compares it and repartitions on mismatch. A5 sums counts in `Int` and calls `_check_index_capacity` before any `TI` conversion or `@inbounds` loop.

## Configuration space

Threads {1, 4 (`-t 4` and `-t 4,1`)} × call context {top level, inside `@threads`, two concurrent `@spawn`} × system size {< 2^16, ≥ 2^16}; ranks {2, 3} × failure point {reader, partition, restart, FOAMCase}. Gate per step below; the full serial suite and `gate.jl` run once at milestone close.

## Steps

Expected 5 steps.

- [ ] **P1-M34-S1** A1 + A2: `_foreach_chunk`/`_reduce_chunks` use `:static` only when not already in a threaded region (`ccall(:jl_in_threaded_region, Cint, ()) == 0`), else `:dynamic` over the same chunks (the reduction stays chunk-ordered, so results are unchanged); `_setup` AutoTune uses `cld(max(n, 1), nthreads)`. Add both repros to `test/unit_test_xvector.jl` (nested and concurrent `run!`-size solves at `-t 4`; empty-range AutoTune launch) - mechanism: thread affinity is a cache hint, never a correctness need - cost: one ccall per primitive call - verdict: tests pass at 1 and 4 threads; motorBike 1t/8t 20-iteration smoke bitwise vs S0 HEAD and run_s within noise.
- [ ] **P1-M34-S2** A3: ok-flag broadcast around rank-0 reader/partition in `distribute(reader; dir)` and `distribute(mesh)`, `Allreduce` of `isfile` in `restart_fields!`, ok-flag around the per-rank `FOAM3D_mesh` in `FOAMCase` and a length check of `cellProcAddressing` - mechanism: a collective is entered by all ranks or by none - cost: one small collective per call, setup only - verdict: a new `test/distributed/test_failure.jl` (reader that throws on rank 0; partial restart dir; broken processor dir) errors on every rank at n=2,3 within the test timeout instead of hanging; `test_offline.jl`, `test_restart.jl` (`dev/petscenv_stock`), `test_io.jl` pass.
- [ ] **P1-M34-S3** A4: part header gains a mesh fingerprint (cell/face/node counts, a hash of node coordinates or of the reader call where cheaper, TI, TF); `_parts_match` compares it; part format 6 - mechanism: a part is reused only for the mesh it was cut from - cost: one hash at write and read - verdict: `test_offline.jl` extended: parts of mesh A in `dir` are replaced when `distribute(reader_B; dir)` runs; unchanged parts are reused without reading; MPI n=4 smoke bitwise on regenerated parts.
- [ ] **P1-M34-S4** A5: counters in `Int`, `_check_index_capacity` before TI conversions and before the UNV3 `@inbounds` cursor loops - mechanism: a count is checked before it is narrowed - cost: none per element - verdict: `test_mesh_conversion.jl` + unit tests calling the counting helpers with fake totals above `typemax(Int32)` get the capacity `ArgumentError`; readers bitwise on the test grids.
- [ ] **P1-M34-S5** close: full serial suite by file (5 groups, `dev/scripts/suite_file.jl`), `gate.jl` as two commands, docs build; CHANGELOG entries (Fixed) for A1-A5.

## Exit criterion

A1-A5 fixed with a test each, full serial suite and distributed gate green, motorBike smokes bitwise vs the pre-M34 HEAD.

## Open questions

- A4 fingerprint cost on 10^8-cell meshes: hash node coordinates (exact, O(n) once) or the reader arguments (cheap, misses edited files)? Settle at S3 by timing the hash on the 4 mm BFS.
