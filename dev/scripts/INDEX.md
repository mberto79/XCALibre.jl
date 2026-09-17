# project scripts

Every helper here is specific to this project. One entry per script: what it answers, and the command that runs it.

- `<script>` - <what question it answers>; run: `<command>`.

- `scaling_probe.jl` — strong-scaling probe for the distributed backward-facing-step case: serial, single-rank and multi-rank per-iteration cost with residuals and core clock. Partitions are cached under `~/.cache/xcal_scaling_probe`.
