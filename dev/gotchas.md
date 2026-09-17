# gotchas: XCALibre.jl distributed (MPI) module

One line per trap. Reasoning lives in `dev/decisions.md`; this file is how to WORK here.

## workflow

- Never launch a Julia simulation in the foreground: compilation plus GPU warm-up exceeds the two-minute command timeout every time. Always run in the background and redirect simulation output to a file.
- Smoke-test any new simulation script at one or two iterations before a full run; a typo costs a minute instead of an hour.
- `test/distributed/runtests_mpi.jl` swallows child standard output when a test passes, so anything printed by the run must be checked with a direct `mpiexec` invocation instead.
- A distributed hang with flat resident memory and no solver banner is almost always ranks dispatching differently: any value that selects a method must be broadcast so it has the same type on every rank.
- `xcalibre-dev check` exits non-zero on an invalid vault, but a status read through a pipe is the pipe's status; run it bare.

## environment and libraries

- Julia resolves `Preferences` per project environment and `PETSc.jl`'s low-level wrappers are generated at precompilation for the configured library only, so the scalar precision and the library path are an environment choice and cannot be switched at runtime.
- `--heap-size-hint` changes the precompilation cache-flags hash, so a child precompiling the PETSc extension asks for an image built under different flags and fails with a message that reads exactly like cache corruption; clearing the compiled cache does not fix it. Use `--startup-file=no` and no heap hint.
- Stock `PETSc_jll` ships hypre for Float64 only and no CUDA in any of its libraries, so GPU-native solves need a custom PETSc build and Float32 users have no hypre.
- `activate_multithread(backend::CPU)` pins BLAS to one thread despite its name; without it BLAS takes every core and oversubscribes the ranks.
- Julia threads default to one, so the CPU kernel backend is already serial under MPI; passing more threads per rank adds overhead rather than removing it.

## machine

- `/tmp` is memory-backed on this box: never write mesh partitions there.
- The memory ceiling is rank zero holding the global mesh, not the rank count, at roughly 1.6 KB per cell. Partitioning offline in a separate process removes it.
- This laptop has sixteen performance cores and sixteen efficiency cores; more than eight ranks crosses onto the slower cores and any scaling number past that measures core heterogeneity.
- Sustained load throttles the clock to a fraction of peak, which alone can produce a scaling curve of the shape observed, so any scaling measurement must log clock frequency alongside timings.
