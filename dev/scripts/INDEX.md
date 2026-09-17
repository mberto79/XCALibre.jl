# project scripts

Every helper here is specific to this project. One entry per script: what it answers, and the command that runs it.

- `<script>` - <what question it answers>; run: `<command>`.

- `scaling_probe.jl` — strong-scaling probe for the distributed backward-facing-step case: serial, single-rank and multi-rank per-iteration cost with residuals and core clock. Partitions are cached under `~/.cache/xcal_scaling_probe`.
- `heated_control.sh` — single-rank run with the other performance cores loaded by spin loops, so rank count and thermal state stop rising together.
- `openfoam_scaling.sh` — the OpenFOAM 12 backward-facing-step benchmark at several rank counts, same binding and per-iteration metric as the XCALibre probe. Needs a clean environment; see the header.
- `xcal_of_compare.sh` — runs both of the above back to back on the same 5 mm mesh for a like-for-like curve.
