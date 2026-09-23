# P1-M30 - threaded linear solve on one thread pool

Milestone row: `dev/phaseRoadmap.md`. Source: `dev/archive/reviews/p1/memory-scaling-2026-09-23.md` § 4. Evidence: `dev/telemetry/memory_scaling.md` § diagnosis. Expected 6 steps (D159).

## mechanism

Krylov.jl 0.10 dispatches `Vector{<:BlasFloat}` vector primitives to OpenBLAS, whose `axpby` is single-threaded and whose spinning pool competes with Julia's. A thin CPU-only `XVector{T} <: DenseVector{T}` wrapping a `Vector{T}` plus the static row partition `xmul!` already uses routes `kdot`, `kdotr`, `knorm`, `kscal!`, `kcopy!`, `kaxpy!`, `kaxpby!`, `kfill!`, SpMV and the Jacobi apply onto Julia's threads over ONE partition, so the thread that wrote `y[r]` reads it next. `similar` keeps the partition, so Krylov workspaces built from a wrapped `b` come out wrapped. Wrapping at the `solve_system!` boundary holds a reference, never a copy; field storage stays `Vector`. Reductions use per-chunk partials padded to 64 B, summed serially: deterministic for a fixed partition. GPU keeps device vectors, MPI keeps PETSc. BLAS then defaults to one thread.

## bars (ranked)

- STRICT (restated D197): residuals differ from the pre-M30 revision by no more than the new revision differs from itself between 1 and 2 threads (reduction order only; motorBike stays ≥10 figures); 1t not slower beyond ±5% noise.
- Objective: 8t Krylov vector-op share falls from about 13 s to under 3 s; 8t within 5% of 8-rank MPI on the same revision.

## gate

Unit tests of each primitive against `Vector` at 1 and 8 threads; motorBike 20-iteration smoke at 1t and 8t against the M29 residual files (8 figures); 100-iteration main-thread profile at 8t, one command.

## steps

- [x] P1-M30-S1 DELIVERED by the M28 close profile (D190): Krylov vector work ~22% of 8t main-thread samples, progress-output strings ~8% (not reached by `XVector`). Baseline: 100-iteration 8t main-thread profile, Krylov vector-op totals (`kaxpby!`, `kaxpy!`, `kdot`, `kfill!`, `mulorldiv!`). A flat per-thread profile charges spawned work to workers and the join to `mul!`; read the main thread only. Blast radius: none shipped.
- [x] P1-M30-S2 LANDED (D196; internal, not exported): `XVector` type, partition, `similar`, `size`, `getindex`, `setindex!`, `unsafe_convert`, and the `k*` primitives, in `src/Multithread/`. Blast radius: none until wired. Bar: primitive unit tests.
- [x] P1-M30-S3 LANDED (D197; serial below 2^16 elements touched): wire into `solve_system!` for the CPU backend: wrap `b` and `values`, workspaces from the wrapped `b`, partitioned `mul!` and Jacobi `ldiv!`; `krylov_solve!` still resolves the preconditioner method. DILU stays serial. Blast radius: every CPU linear solve. Bar: strict class.
- [ ] P1-M30-S4 `activate_multithread` defaults BLAS to one thread, keyword kept; docstring and benchmark README BLAS section updated. Blast radius: thread setup. Bar: 8t smoke time not slower.
- [ ] P1-M30-S5 remove the dead `spmvm.jl` methods (`xmul!(A, x)` with undefined `y`, `xmul(y, A, x)` discarding `y`, `Base.:*` calling a missing 2-argument `xmul`) or make them correct. Blast radius: none reachable. Bar: suite files for Multithread.
- [ ] P1-M30-S6 close: 500-iteration per-point 1t/8t/n=8 timings, profile; a fused CG of our own on `XVector` only if vector ops stay above about 3 s at 8t. Persistent thread team only if measured (spawn-per-call scaled 5.6x in isolation).
