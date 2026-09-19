# gotchas: XCALibre.jl distributed (MPI) module

One line per trap. Reasoning lives in `dev/decisions.md`; this file is how to WORK here.

## workflow

- Never launch a Julia simulation in the foreground: compilation plus GPU warm-up exceeds the two-minute command timeout every time. Always run in the background and redirect simulation output to a file.
- Smoke-test any new simulation script at one or two iterations before a full run; a typo costs a minute instead of an hour.
- Never launch the whole distributed suite (every file at ranks 1,2,3,5) in one go, and never without `dev/scripts/memguard.sh`: on 2026-09-19 it ran this 14 GB box out of memory and took VS Code down. Run `gate.jl`, then named files one at a time under `MIN_MB=2500 memguard.sh`.
- `test/distributed/runtests_mpi.jl` swallows child standard output when a test passes, so anything printed by the run must be checked with a direct `mpiexec` invocation instead.
- BFS tet meshes at 3, 4, 5 and 10 mm live in `~/Desktop/BFS_GRIDS/` (`bfs_unv_tet_<h>mm.unv`); parts are refused at load unless written at the current part format (2 since P1-M22-S4), so any cached part dated before 2026-09-20 must be regenerated with `mem_probe.jl part`.
- A distributed hang with flat resident memory and no solver banner is almost always ranks dispatching differently: any value that selects a method must be broadcast so it has the same type on every rank.
- `xcalibre-dev check` exits non-zero on an invalid vault, but a status read through a pipe is the pipe's status; run it bare.
- `pgrep -f <pattern>` matches the poller's OWN command line when the pattern appears in it, so `until ! pgrep -f 'Pkg.test'` never exits while any shell mentions `Pkg.test`. Wait on a marker written to a file instead.

## time budget (D101)

- HARD CAP: any gate or experiment whose result decides a verdict finishes within five minutes of wall clock, compilation included. Plan the run to fit before launching it; if it cannot fit, shrink it, never extend the cap.
- Measured durations to plan with: `gate.jl` (n=2,3) about 3.3 min; single suite files at `--ranks=2,3` 0.5-4 min each (`test_invariance.jl` 3.3, `test_gpu.jl` at 1,2 about 4); a 10 mm `mem_probe.jl` worker about 1 min; a 4 mm worker about 2 min; docs build several minutes.
- Levers, cheapest first: the 10 mm mesh instead of 5 or 4 mm; 2-5 iterations; one rank-count pair; only the suite files the change can reach, run as separate commands; one A/B pair per timing run (the `equal_thermal.sh` 25 s warm-up and 15 s settle per point add up); `mem_probe.jl` residual hashes instead of full tests where bitwise equality is the verdict.
- Swapping an extension file to compare versions forces a recompile on each swap (about 1 min); count it against the cap.

## environment and libraries
- `xcalibre-dev` is a Python script without the executable bit: run it as `python3 <skill-dir>/scripts/xcalibre-dev check .`; `bash` misreads it and `resume` blocked past the two-minute command timeout here, so read the `LOAD` records directly instead.
- `pkill -f <pattern>` kills the shell whose command line contains the pattern, which is the shell issuing it; match on a process name or a pid file instead.
- MPI and PETSc API coverage for the M21-M23 plans was checked on 2026-09-18 against the installed packages: PETSc.jl wraps `VecCreateMPIWithArray`, `VecPlaceArray`, `MatUpdateMPIAIJWithArray`, `MatCreateMPIAIJWithSplitArrays`, `MatMPIAIJSetPreallocationCSR`, `MatPartitioningCreate`, `PetscDeviceContextGetStreamHandle`, but NOT the CUDA vector variants (`VecCreateMPICUDAWithArray`, `VecCUDAPlaceArray`), which need a hand-written `@ccall`; MPI.jl has `Send_init`/`Recv_init`/`Start`; Metis.jl takes vertex weights only through a hand-built `Metis.Graph`.

- Julia resolves `Preferences` per project environment and `PETSc.jl`'s low-level wrappers are generated at precompilation for the configured library only, so the scalar precision and the library path are an environment choice and cannot be switched at runtime.
- `--heap-size-hint` on MPI ranks is safe once the caches exist: precompile in a plain session first, because a rank that has to precompile the PETSc extension under the flag asks for images built under other cache flags and fails like cache corruption. With caches built it loads them and lowers the 4 mm BFS n=2 peak 1772 → 1667 MB at 1200M (D96). Always `--startup-file=no`.
- Stock `PETSc_jll` ships hypre for Float64 only and no CUDA in any of its libraries, so GPU-native solves need a custom PETSc build and Float32 users have no hypre.
- `activate_multithread(backend::CPU)` pins BLAS to one thread despite its name; without it BLAS takes every core and oversubscribes the ranks.
- Any keyword `BoomerAMG`/`GAMG` does not know, including a removed `reuse=` or a typo, is forwarded to PETSc as `-pc_<prefix>_<k>` and silently ignored unless `-options_left` is set; it never errors.
- This shell has the custom OpenMPI `mpiexec` first on PATH, which aborts ranks of the stock (MPICH) env with `internal_Init_thread`; launch stock-env runs through `MPI.mpiexec()` or `mpiexecjl`.
- `dev/petscenv_f32` goes stale when XCALibre gains a dependency (precompile fails with "Cannot load module ... into XCALibre"); `Pkg.resolve()` in that env fixes it.
- Julia threads default to one, so the CPU kernel backend is already serial under MPI; passing more threads per rank adds overhead rather than removing it.
- A distributed GPU run that dies with `signal 11` and NO Julia backtrace means Julia's SIGSEGV handler is gone: check `/proc/self/task/*/status` SigCgt for bit 10 (0x400); PETSc's CUDA device init clears it (D85) and `_with_julia_signals` in the PETSc extension puts it back. Julia 1.13 has an interactive thread, so GC safepoint faults happen even with `-t 1`.
- GPU residuals differ run to run in the last bits (reduction order), so GPU equivalence is a tolerance check; bitwise hashes apply to CPU runs only. PETSc's CUDA path runs on the legacy NULL stream, CUDA.jl on a non-blocking task stream (D116).
- `HYPRE_GetMemoryLocation` says device on a CPU-only hypre too (device memory maps to host there); `HYPRE_GetExecutionPolicy` is the query that distinguishes the builds (D79). Both live in libHYPRE, reachable through `dlsym` on the libpetsc handle.

## machine

- `/tmp` is memory-backed on this box: never write mesh partitions there.
- This box has 14 GB: the 4 mm BFS (1.32M cells) at eight ranks with AMG ran it out of memory and the kernel OOM-killed Chrome and VS Code. Run large-mesh sweeps under `dev/scripts/memguard.sh`.
- The memory ceiling is rank zero holding the global mesh, not the rank count: 0.9 KB per cell above the runtime after load and 2.4 GB peak while partitioning the 4 mm BFS (`dev/telemetry/memory_breakdown.md`). Partitioning offline in a separate process removes it.
- Every rank pays about 800 MB before it holds a cell (Julia runtime and packages 546 MB, compiled code the rest), so memory per rank is roughly 790 MB + 1.85 KB per local cell on the BFS at Float64.
- This laptop has EIGHT performance cores (CPUs 0-15, hyperthread siblings paired adjacently, so 0,1 = core 0) and sixteen efficiency cores at 4.1 GHz (CPUs 16-31). More than eight ranks crosses onto the slower cores and any scaling number past that measures core heterogeneity.

## measuring across rank counts

- The package falls from 4400 MHz on one busy core to 3100 MHz on eight, so RANK COUNT AND CLOCK CO-VARY and an uncorrected strong-scaling curve measures the power limit, not the code. This produced a wrong attribution once already (D16, withdrawn by D19).
- Sample the clock DURING a run, never after it: a reading taken once the solver has exited shows idle cores and is worthless (it made the first OpenFOAM comparison unusable, D22).
- Without root, `dev/scripts/equal_thermal.sh` is the substitute for pinning: spin loops occupy every P-core the solver is not using, so all rank counts throttle equally. The two methods agree.

- Pin the clock before measuring and revert afterwards with `dev/scripts/pin_clock.sh pin|revert`.
- conda CUDA PETSc envs: micromamba at `~/.local/micromamba` (envs `petsc-cuda` mpich, `petsc-cuda-ompi` openmpi); Julia envs `dev/petscenv_conda*`; openmpi runs need `OMPI_MCA_opal_cuda_support=true` (D69).
