# gotchas: XCALibre.jl distributed (MPI) module

One line per trap. Reasoning lives in `dev/decisions.md`; this file is how to WORK here.

## workflow

- Never launch a Julia simulation in the foreground: compilation plus GPU warm-up exceeds the two-minute command timeout every time. Always run in the background and redirect simulation output to a file.
- Smoke-test any new simulation script at one or two iterations before a full run; a typo costs a minute instead of an hour.
- `test/distributed/runtests_mpi.jl` swallows child standard output when a test passes, so anything printed by the run must be checked with a direct `mpiexec` invocation instead.
- A distributed hang with flat resident memory and no solver banner is almost always ranks dispatching differently: any value that selects a method must be broadcast so it has the same type on every rank.
- `xcalibre-dev check` exits non-zero on an invalid vault, but a status read through a pipe is the pipe's status; run it bare.
- `pgrep -f <pattern>` matches the poller's OWN command line when the pattern appears in it, so `until ! pgrep -f 'Pkg.test'` never exits while any shell mentions `Pkg.test`. Wait on a marker written to a file instead.

## environment and libraries
- `xcalibre-dev` is a Python script without the executable bit: run it as `python3 <skill-dir>/scripts/xcalibre-dev check .`; `bash` misreads it and `resume` blocked past the two-minute command timeout here, so read the `LOAD` records directly instead.
- `pkill -f <pattern>` kills the shell whose command line contains the pattern, which is the shell issuing it; match on a process name or a pid file instead.
- MPI and PETSc API coverage for the M21-M23 plans was checked on 2026-09-18 against the installed packages: PETSc.jl wraps `VecCreateMPIWithArray`, `VecPlaceArray`, `MatUpdateMPIAIJWithArray`, `MatCreateMPIAIJWithSplitArrays`, `MatMPIAIJSetPreallocationCSR`, `MatPartitioningCreate`, `PetscDeviceContextGetStreamHandle`, but NOT the CUDA vector variants (`VecCreateMPICUDAWithArray`, `VecCUDAPlaceArray`), which need a hand-written `@ccall`; MPI.jl has `Send_init`/`Recv_init`/`Start`; Metis.jl takes vertex weights only through a hand-built `Metis.Graph`.

- Julia resolves `Preferences` per project environment and `PETSc.jl`'s low-level wrappers are generated at precompilation for the configured library only, so the scalar precision and the library path are an environment choice and cannot be switched at runtime.
- `--heap-size-hint` changes the precompilation cache-flags hash, so a child precompiling the PETSc extension asks for an image built under different flags and fails with a message that reads exactly like cache corruption; clearing the compiled cache does not fix it. Use `--startup-file=no` and no heap hint.
- Stock `PETSc_jll` ships hypre for Float64 only and no CUDA in any of its libraries, so GPU-native solves need a custom PETSc build and Float32 users have no hypre.
- `activate_multithread(backend::CPU)` pins BLAS to one thread despite its name; without it BLAS takes every core and oversubscribes the ranks.
- Any keyword `BoomerAMG`/`GAMG` does not know, including a removed `reuse=` or a typo, is forwarded to PETSc as `-pc_<prefix>_<k>` and silently ignored unless `-options_left` is set; it never errors.
- This shell has the custom OpenMPI `mpiexec` first on PATH, which aborts ranks of the stock (MPICH) env with `internal_Init_thread`; launch stock-env runs through `MPI.mpiexec()` or `mpiexecjl`.
- `dev/petscenv_f32` goes stale when XCALibre gains a dependency (precompile fails with "Cannot load module ... into XCALibre"); `Pkg.resolve()` in that env fixes it.
- Julia threads default to one, so the CPU kernel backend is already serial under MPI; passing more threads per rank adds overhead rather than removing it.
- A distributed GPU run that dies with `signal 11` and NO Julia backtrace means Julia's SIGSEGV handler is gone: check `/proc/self/task/*/status` SigCgt for bit 10 (0x400); PETSc's CUDA device init clears it (D85) and `_with_julia_signals` in the PETSc extension puts it back. Julia 1.13 has an interactive thread, so GC safepoint faults happen even with `-t 1`.
- `HYPRE_GetMemoryLocation` says device on a CPU-only hypre too (device memory maps to host there); `HYPRE_GetExecutionPolicy` is the query that distinguishes the builds (D79). Both live in libHYPRE, reachable through `dlsym` on the libpetsc handle.

## machine

- `/tmp` is memory-backed on this box: never write mesh partitions there.
- This box has 14 GB: the 4 mm BFS (1.32M cells) at eight ranks with AMG ran it out of memory and the kernel OOM-killed Chrome and VS Code. Run large-mesh sweeps under `dev/scripts/memguard.sh`.
- The memory ceiling is rank zero holding the global mesh, not the rank count, at roughly 1.6 KB per cell. Partitioning offline in a separate process removes it.
- This laptop has EIGHT performance cores (CPUs 0-15, hyperthread siblings paired adjacently, so 0,1 = core 0) and sixteen efficiency cores at 4.1 GHz (CPUs 16-31). More than eight ranks crosses onto the slower cores and any scaling number past that measures core heterogeneity.

## measuring across rank counts

- The package falls from 4400 MHz on one busy core to 3100 MHz on eight, so RANK COUNT AND CLOCK CO-VARY and an uncorrected strong-scaling curve measures the power limit, not the code. This produced a wrong attribution once already (D16, withdrawn by D19).
- Sample the clock DURING a run, never after it: a reading taken once the solver has exited shows idle cores and is worthless (it made the first OpenFOAM comparison unusable, D22).
- Without root, `dev/scripts/equal_thermal.sh` is the substitute for pinning: spin loops occupy every P-core the solver is not using, so all rank counts throttle equally. The two methods agree.

Pin the clock before measuring, and put it back afterwards:

```bash
# pin at the 2200 MHz base clock on every core
powerprofilesctl set performance
echo 1   | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
echo 100 | sudo tee /sys/devices/system/cpu/intel_pstate/min_perf_pct
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# revert to this machine's normal state
echo 0  | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
echo 15 | sudo tee /sys/devices/system/cpu/intel_pstate/min_perf_pct
echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo balance_performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference
powerprofilesctl set balanced
```
- conda CUDA PETSc envs: micromamba at `~/.local/micromamba` (envs `petsc-cuda` mpich, `petsc-cuda-ompi` openmpi); Julia envs `dev/petscenv_conda*`; openmpi runs need `OMPI_MCA_opal_cuda_support=true` (D69).
