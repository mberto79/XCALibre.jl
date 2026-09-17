# project scripts

Every helper here is specific to this project. One entry per script: what it answers, and the command that runs it.

- `<script>` - <what question it answers>; run: `<command>`.

- `scaling_probe.jl` — strong-scaling probe for the distributed backward-facing-step case: serial, single-rank and multi-rank per-iteration cost with residuals and core clock. Partitions are cached under `~/.cache/xcal_scaling_probe`. Per-iteration cost is `(t100 - t3) / 97`, which cancels compilation and setup.
  - `julia --project=<env> dev/scripts/scaling_probe.jl serial <mesh.unv> <iters>`
  - `julia --project=<env> dev/scripts/scaling_probe.jl drive <mesh.unv> <n>... <iters>`
  - `mpiexec -n <n> julia --project=<env> dev/scripts/scaling_probe.jl worker <partdir> <iters>`
- `heated_control.sh` — single-rank run with the other performance cores loaded by spin loops, so rank count and thermal state stop rising together. The heaters are scalar: they reproduce thermal load but consume no memory bandwidth.
  - `dev/scripts/heated_control.sh <partdir_n1> <iters> [env]`
- `openfoam_scaling.sh` — the OpenFOAM 12 backward-facing-step benchmark at several rank counts, same binding and per-iteration metric as the XCALibre probe. The case is OpenFOAM 12 (`foamRun`, `constant/momentumTransport`) while this shell usually has the ESI build sourced, so re-enter from a clean environment. The serial run is pinned with `taskset` so it matches the binding `mpiexec` gives every other rank count. The case is copied to `~/.cache/xcal_of_scaling`, never `/tmp`, which is memory-backed here.
  - `env -i HOME=$HOME PATH=/usr/bin:/bin bash -lc 'source $HOME/OpenFOAM/OpenFOAM-12/etc/bashrc; cd <repo>; dev/scripts/openfoam_scaling.sh 1 2 4 8'`
- `xcal_of_compare.sh` — runs both of the above back to back on the same 5 mm mesh for a like-for-like curve.
