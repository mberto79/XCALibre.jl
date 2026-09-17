# P1-M4 - distributed setup interface (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R1, R2, R3, R5, R6. Governing decisions: D4, D5.

## Problem, quantified

A distributed case today needs eleven lines of preamble before any physics: an `MPI.Init`, three communicator queries, an environment-variable assignment, a rank-guarded partition with a barrier, and two more environment reads for the mesh path. Of those, the rank guard is the one that hangs: assigning a path only on rank zero gives the other ranks `nothing`, which selects a different `distribute` method, and the run blocks in `MPI.recv` for a scatter that never comes. The launch line is 190 characters of nested backticks. Three environment variables reach into the run: `PETSC_OPTIONS`, `XCAL_BFS_GRIDS` and `XCAL_BFS_MESH`.

## Approach

One rank-uniform way in, so the deadlock class cannot be written. `distribute` gains a `dir` keyword on its reader method: every rank calls the same expression, the function initialises MPI, partitions on rank zero only when the directory is absent, barriers, and every rank then loads its own part. Nothing in the user's script branches on rank, so no value can differ in type between ranks. The existing positional-directory and bare-mesh methods stay for pre-made decompositions and in-memory meshes.

Output still needs a root guard, and exactly one is enough: a predicate that is true on rank zero and true in serial, so the same line works in both. It is a predicate rather than a block form because a block cannot return a value and users print residuals.

Global PETSc options move onto the existing `petsc_options` string rather than a new knob: it is already threaded from `run!` to the solver, and `PETSc.initialize` takes the same options, so one string configures both initialisation and the Krylov solve. PETSc ignores what it does not consume.

Launching uses MPI.jl's own `mpiexecjl`, which resolves the same binary the package resolves and needs no XCALibre code. It is documented, not wrapped.

## Configuration space

Parameters: mesh source (in-memory, offline directory, reader with `dir`) x rank count (1, 2, 4) x periodicity (none, colocated pairs) x root guard (serial, distributed). The existing `test/distributed` files already cover the first two products; the guard and the reader-with-`dir` form need rows added to `test_offline.jl` and `test_partition.jl`. The gate for this milestone is the distributed suite at ranks 1 and 2 plus the backward-facing-step example run end to end.

## Steps

- [x] **P1-M4-S1** `distribute(reader; dir)` partitions once on rank zero, barriers and loads per rank - mechanism: every rank evaluates the same call with the same argument types, so no rank can select a different method - cost: one barrier per run - verdict: a two-rank run of the reader-with-`dir` form completes and matches the offline form's mesh, and removing the barrier makes it fail. DELIVERED: `test_offline.jl` passes at one and two ranks, covering reader-runs-on-root-only, stale-part replacement and reuse.
- [x] **P1-M4-S2** a root predicate true on rank zero and in serial - mechanism: the predicate reads the communicator only when MPI is initialised, so a serial script needs no MPI - cost: none - verdict: the same script line prints once under one rank, once under four, and once with no MPI at all. DELIVERED as `is_root`, guarded against being called after `MPI.Finalize`.
- [x] **P1-M4-S3** `petsc_options` reaches `PETSc.initialize` as well as the Krylov solver - mechanism: one options string, consumed by whichever PETSc stage recognises each entry - cost: none - verdict: `-log_view` passed through `run!` takes effect with `PETSC_OPTIONS` unset. `-use_gpu_aware_mpi 0` is only testable against a CUDA PETSc, which `PETSc_jll` does not ship, so `-log_view` stands for the mechanism. DELIVERED: a two- and eight-rank run with `petsc_options="-log_view"` printed the PETSc performance summary with `PETSC_OPTIONS` unset.
- [x] **P1-M4-S4** the backward-facing-step example is rewritten on the new interface with its environment variables replaced by script arguments - mechanism: the example is the interface's own acceptance case - cost: none - verdict: it reads as a serial case plus two distributed lines and runs on stock binaries from a clean directory. DELIVERED: the preamble is four lines and takes its mesh from `ARGS`.
- [x] **P1-M4-S5** the other distributed examples and the documentation page follow the same interface - mechanism: one documented way in, so no example teaches the rank-guarded form - cost: none - verdict: no example contains a rank-guarded mesh assignment or an environment variable. DELIVERED for all three examples and the documentation page, which also gained the stock-binary requirements and the `mpiexecjl` launch.

## Exit criterion

The distributed suite is green at one and two ranks, the backward-facing-step example runs to completion on stock binaries with no environment variable set and no preferences file, and no example or documentation page shows a setup that can diverge per rank.

## Open questions

- Whether a wrapper around `mpiexecjl` earns its place, or documentation is enough. Settled by whether the wrapper can do anything `mpiexecjl -n 4 julia --project=. case.jl` cannot; if not, it is not written.
