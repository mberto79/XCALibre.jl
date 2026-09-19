# Active context - distributed module release polish
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md
updated: 2026-09-19T21:45:00+01:00
STATE: IDLE
STEP: P1 exit gate and close (all milestones closed; M15 D136)
HEAD: ad7b4477
BRANCH: HM/distributed-draft
GATE: test/distributed/gate.jl (about 3.3 min) plus only the suite files the change reaches, each a separate command under MIN_MB=2500 memguard.sh; every verdict run under five minutes (D101)
resume: every P1 milestone is closed; the phase exit gate (full serial suite, gate, example on stock binaries, invariance, docs) and the close are the user's call through the xcalibre-close skill; the 108 pre-existing long comment blocks on main are a separate cleanup PR off main (D136)
## binding
- HARD CAP (D101, user): every gate or experiment deciding a verdict finishes in five minutes of wall clock; `dev/gotchas.md` § time budget has the measured durations and the levers.
- M22 closed (D118), M23 closed (D129), M24 closed (D132), M26 closed (D135), M15 closed (D136), M25 closed (D109); M26 (shipped precompile, user-opened D108) runs after M24.
## implementer
- These milestones are implemented by Opus 5 in the current session (D121 amends D75), in roadmap order, committing and pushing each step and not stopping at milestone boundaries (D75). Every plan states mechanism, cost and verdict per step; verdicts are measurable on this machine. Multi-node, multi-GPU and AMD validation is P2 on the HPC (D73), so nothing here waits for hardware that is not present.
- Source of the work: `dev/archive/reviews/p1/audit-2026-09-18.md` (archived at M18 close, D77). Its three structural changes are M20+M23 (preprocessing), M21 (memory), M22 (communication); its release-blocker list is M18.
- Bars that gate every step: residual hashes bitwise equal under Jacobi at n=2 and n=4 on CPU (n=8 with PETSc does not fit beside the language server; GPU residuals are tolerance-checked, D116; R8 binding; `mem_probe.jl` on 10 mm parts fits the cap), `check_ghosts` zero (M19-S2), the M19-S4 round and all-reduce counters, and `gate.jl`. The full serial suite exceeds the five-minute cap and runs only at phase close.
## position
M1-M26 closed (M12 superseded by M13; M22 on 2026-09-19 with S6-S8 withdrawn on bounds, D110-D118). Open: none; plans in `dev/plans/`.
## evidence
- The scaling attribution in D16 was WRONG and is withdrawn. With the clock pinned the module scales at 102/94/71% (n=2/4/8) and mesh size does not move it (D20, D21). Do not reopen this without reading `dev/telemetry/scaling_attribution.md` first.
- This machine throttles 4400 to 3100 MHz as rank count rises. ANY timing comparison across rank counts is meaningless unless the clock is pinned or the package power held constant; `dev/gotchas.md` carries both methods.
- `dev/petscenv_stock` (XCALibre, MPI, PETSc, Test; no preferences) runs the CPU path; `dev/petscenv_conda_ompi` runs the GPU path with a CUDA-aware Open MPI (`OMPI_MCA_opal_cuda_support=true`); `dev/petscenv` is the custom CUDA-hypre build.
- Memory baseline after M21 (D99): 1772 MB peak per rank at 4 mm n=2 (1667 MB with `--heap-size-hint=1200M`); header-checked 5 mm parts for n=1,2,4,8 are in `~/.cache/xcal_mem_probe/` and copied to `~/.cache/xcal_scaling_probe/bfs5h_n*` for `equal_thermal.sh bfs5h`. n=8 timing needs the language server closed (about 7 GB).
## blocked/carried
- Parts are `.xdm` format 3 since M23-S1 (D122, D123); every `.jls` cache is obsolete. Current: `~/.cache/xcal_m23/bfs10_n8`, `bfs4_n8`; regenerate others with `mem_probe.jl part`.
- Memory: 14 GB box. Wrap every large-mesh run in `dev/scripts/memguard.sh`.
- Machine partly reverted at 2026-09-18: turbo on, min_perf 15, governor powersave, but `powerprofilesctl get` still says performance; run `powerprofilesctl set balanced` before timing.
- `xcalibre-dev check` stays INVALID on 108 long comment blocks that already exist on main (`src/`, `test/`); cleanup PR off main (D136).
