# project scripts

Every helper here is specific to this project. One entry per script: what it answers, and the command that runs it.

- `<script>` - <what question it answers>; run: `<command>`.

- `scaling_probe.jl` — strong-scaling probe (`dev=cuda` runs the worker on the GPU; `wait=1` prints per-rank barrier wait before each PETSc assembly, and its per_iter is invalid because the patch recompiles) for the distributed backward-facing-step case: serial, single-rank and multi-rank per-iteration cost with residuals and core clock. Partitions are cached under `~/.cache/xcal_scaling_probe`. Per-iteration cost is `(t100 - t3) / 97`, which cancels compilation and setup.
  - `julia --project=<env> dev/scripts/scaling_probe.jl serial <mesh.unv> <iters>`
  - `julia --project=<env> dev/scripts/scaling_probe.jl drive <mesh.unv> <n>... <iters>`
  - `mpiexec -n <n> julia --project=<env> dev/scripts/scaling_probe.jl worker <partdir> <iters>`
- `heated_control.sh` — single-rank run with the other performance cores loaded by spin loops, so rank count and thermal state stop rising together. The heaters are scalar: they reproduce thermal load but consume no memory bandwidth.
  - `dev/scripts/heated_control.sh <partdir_n1> <iters> [env]`
- `openfoam_scaling.sh` — the OpenFOAM 12 backward-facing-step benchmark at several rank counts, same binding and per-iteration metric as the XCALibre probe. The case is OpenFOAM 12 (`foamRun`, `constant/momentumTransport`) while this shell usually has the ESI build sourced, so re-enter from a clean environment. The serial run is pinned with `taskset` so it matches the binding `mpiexec` gives every other rank count. The case is copied to `~/.cache/xcal_of_scaling`, never `/tmp`, which is memory-backed here.
  - `env -i HOME=$HOME PATH=/usr/bin:/bin bash -lc 'source $HOME/OpenFOAM/OpenFOAM-12/etc/bashrc; cd <repo>; dev/scripts/openfoam_scaling.sh 1 2 4 8'`
- `xcal_of_compare.sh` — runs both of the above back to back on the same 5 mm mesh for a like-for-like curve.

- `stream.jl` - MPI axpy at a cache-resident and a DRAM-resident size; establishes THIS machine's
  bandwidth ceiling so a solver rate can be judged against what the hardware allows rather than
  against a guess. Run under `mpiexecjl -n <n> --bind-to core --map-by core`.
- `fixed_clock.sh <tag> <iters>` - strong-scaling sweep with the clock pinned by hardware
  (`no_turbo=1`); samples the clock during each run so the pin is verified, not assumed. Needs
  the power state from `dev/gotchas.md` applied first.
- `equal_thermal.sh <tag> <iters>` (env `ENVDIR`, `NS`, `OPTS`, `EXTRA` override the project, rank counts, petsc_options and extra probe args such as `pc=gamg`) - the no-root substitute: spin loops occupy every P-core the
  solver is not using, so every rank count runs at the same sustained power limit.
- `memguard.sh <cmd...>` - runs a command and kills its process group when MemAvailable falls below `MIN_MB` (default 1500); wrap any large-mesh run with it.
- `mem_probe.jl` - per-rank memory by setup stage (RSS, HWM, GC-live, PetscMalloc) plus `/proc/self/smaps_rollup` split (Rss, Pss, private and shared clean/dirty) per stage, the top 25 mapped paths by private bytes at `runtime` and `iterations`, and `summarysize` per structure, for the distributed BFS; `gc=1` forces a full collection before each reading, `pre=<Package>` loads a local package of precompile statements before setup, `gcmax=<MB>` sets the GC memory target at runtime, `repeat=<k>` rows carry each `run!` time, `gclog=1` prints rank 0 GC heap stats per stage, `trim=1` calls `malloc_trim(0)` after each forced collection, `malloc=1` starts PETSc with `-malloc_debug` so its exit dump lists live allocations by site. Run: `julia --project=dev/petscenv_stock dev/scripts/mem_probe.jl part <mesh.unv> <n> <dir>`, then `mpiexec -n <n> julia --project=dev/petscenv_stock dev/scripts/mem_probe.jl worker <dir> <iters> [gc=1] [malloc=1]` (launch through `MPI.mpiexec()`, under `memguard.sh`).
- `pkg_mem.jl` - private and shared MB added by loading packages in a fresh process (transitive deps resolved through the manifest); `list` prints the modules and libraries a worker loads. Run: `julia --startup-file=no --project=dev/petscenv_stock dev/scripts/pkg_mem.jl one <Name>...` or `... list`.
- `pin_clock.sh pin|revert` - pins every core at the 2200 MHz base clock for timing, and restores this machine's normal power state; needs sudo.
- `halo_bench.jl` - one halo exchange timed with persistent requests against fresh `Irecv!`/`Isend` on the same schedule, alternated in one process; prints median microseconds and allocations. Run: `julia --project=dev/petscenv_stock dev/scripts/halo_bench.jl part <mesh.unv> <n> <dir>`, then `mpiexec -n <n> julia --project=dev/petscenv_stock dev/scripts/halo_bench.jl run <dir> <reps> [width]` (through `MPI.mpiexec()`).
- `part_load.jl` - loads every `rank_*.xdm` in a directory in one process and reports the second-pass total and slowest file, so compilation is excluded. Run: `julia --startup-file=no --project=dev/petscenv_stock dev/scripts/part_load.jl <dir>`.
- `foam_decomposed.jl` - laminar 3D BFS on a `decomposePar` case: `serial` writes the `FOAM3D_mesh` reference fields into the case, `worker` runs `distribute(FOAMCase)` and prints relative differences, `check_ghosts` and load time; `WRITE=1` writes results into the decomposed case for `reconstructPar`. Run: `julia --project=dev/petscenv_stock dev/scripts/foam_decomposed.jl serial <case> <iters>`, then `<MPI.mpiexec()> -n <n> julia --project=dev/petscenv_stock dev/scripts/foam_decomposed.jl worker <case> <iters> <decomposed case>`.
- `io_bench.jl` - decomposed-writer time and size: the current binary writer against the ASCII writer from 590b30e3 loaded from git under renamed names, on offline parts. Run: `<MPI.mpiexec()> -n <n> julia --project=dev/petscenv_stock dev/scripts/io_bench.jl <partdir> <outdir>`.
