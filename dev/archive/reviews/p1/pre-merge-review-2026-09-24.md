# Pre-merge review of HM/distributed-draft (2026-09-24)

Three read-only Opus reviews (performance, behaviour vs origin/main, production robustness) at c0ef5101. Findings the user triages into milestones before P1 close.

## Crash / wrong result

- A1 CONFIRMED (re-run): `Threads.@threads :static` in `src/Multithread/xvector.jl` `_foreach_chunk`/`_reduce_chunks` throws when `run!` is called inside another `@threads` loop or concurrently from `@spawn` tasks, once a system touches ≥ 2^16 elements; regression vs main (`@sync/@spawn`). Fix: fall back to `:dynamic`/serial when `jl_in_threaded_region() != 0`.
- A2 `AutoTune` with an empty range: `cld(0, nthreads) = 0` → KernelAbstractions `DivideError` (`Multithread.jl:24`); also on main; hit by any empty per-range launch (e.g. a rank with 0 owned cells). Fix: `cld(max(n,1), nthreads)`.
- A3 Distributed error paths hang instead of failing: rank-0 throw in `distribute(reader; dir)` / `distribute(mesh)` leaves other ranks in Barrier/recv (`Distribute_1_partition.jl:424-438`); per-rank `isfile` in `restart_fields!` (`Distribute_7_io.jl:342`); `FOAMCase` read failure before `Allgather` (`Distribute_8_foam.jl:39,43`).
- A4 `_parts_match` compares only the rank count: parts of a different mesh (or `scale`) in `dir` are silently reused (`Distribute_1_partition.jl:463`, `examples/3D_BFS_mpi.jl`). Fix: mesh fingerprint + TI/TF in the part header.
- A5 Int32 counters that fail with the wrong error past 2^31 entries (`FoamMesh_2_connect.jl:36-43,76-89,123-129,210-217`, ASCII `read_faces`; `connect_cell_nodes` wraps at ~89M hex cells) and a wrapping `@inbounds` cursor in `UNV3_2_builder.jl:213-233,296-303,333`. Fix: sum in `Int`, `_check_index_capacity` before the loops.

## Behaviour changes users would hit

- B1 Hand-written BC functors with `cell::Cell{TF}` and user `scheme!` methods reading `face.weight` break (MethodError); CHANGELOG files it under Changed and says definitions "compile unchanged". Move to Breaking.
- B2 `get_backend(mesh.cells)` throws (ElementArrays lack `KernelAbstractions.get_backend`); `propertynames(mesh)` omits cells/faces/nodes; `cell_nsign` Int8 undocumented.
- B3 `config.postprocess` silently skipped on distributed meshes (`Solvers_1_SIMPLE.jl:268-288`, `Solvers_2_PISO.jl:216-220`): warn + document.
- B4 `activate_multithread` default: origin/main (#161) uses `Threads.nthreads()`; the branch reverts to 1 and deleted main's CHANGELOG line (D198 wrongly called it unreleased). Add a Changed entry.
- B5 Serial OpenFOAM writer writes `dimensions [0 0 0 0 0 0 0]` for every field (`OpenFOAM_writer.jl:333`).
- B6 Internal names exported (`_check_index_capacity`, `_with_index_capacity`, `_sized`, `_index_type`, `passemble!`, `halo_exchange_adjoint!`, `decompose`, `gather`).
- B7 Cosmetic: CHANGELOG `[#160]`/`[#161]` placeholders point at merged PRs (D137); PISO `@time` print removed; `contributor_guide.md:69-73` stale mesh text; docs env pulls PETSc; MPI/Metis now hard deps; `_foam_binary` reads whole ASCII files to check the header (`FoamMesh_1_read.jl:199`); dead `src/precompile.jl` uses 1.11-only `get_bool_env`; no Julia 1.10 CI job.

## Performance (8t motorBike shares from the M30 profile; guesses marked)

- C1 `residual()` extra SpMV + 2 serial sums, 6×/iteration (`Solve_1_api.jl:454-469`): 3-7%. Cheap: threaded partials (~1%); deeper: reuse Krylov's residual (~2.5-3%).
- C2 Dead nzval zeroing before discretise (`Discretise_2_*.jl:41,118`) when no BC extends the pattern: ~2%.
- C3 `implicit_relaxation_diagdom!` re-reads every row 5×/iteration: ~2.7%; off-diagonal Σ|a| could come from discretise (periodic gate).
- C4 Jacobi refresh is a separate `spindex` pass (~1.7%); fusing into relaxation also makes Uy/Uz preconditioners exact (not bitwise).
- C5 Serial main-thread passes (double p copy, `nut = k/ω` broadcast, wall scratch `fill!`): ~1-1.5%.
- C6 GPU: ~31 `KernelAbstractions.synchronize`/iteration (`Discretise_5_apply_bcs.jl:48`, `Solvers_1_SIMPLE.jl:369,377,465,532`) + residual syncs: 2-5% (guess).
- C7 GPU: BC kernels still carry `model` + terms with meshes (depot 2376/1248/720 B): 3-6% (guess); D187 pattern applies.
- C8 Larger levers: fused in-house CG/BiCGStab (fork/join ~2-3%), fused gradient/interpolation (2-4%), PISO p-matrix rebuild per corrector, interactive-thread pinning (guess).
- C9 Serial `update_equation!` full reset (D206/D207 analogue): user decision.
