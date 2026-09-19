# Active context - distributed module release polish
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m18-release-blockers.md
updated: 2026-09-19T12:00:00+01:00
STATE: GATING
STEP: P1-M18-S2..S10 - release blockers, one diff under test
HEAD: d91c67b8
BRANCH: HM/distributed-draft
GATE: julia --project=. -e 'using Pkg; Pkg.test()'
resume: read the suite log `suite_m18.log` and `gpu_m18.log` in the session scratchpad (or rerun the two commands in `dev/gotchas.md` § workflow); green -> commit S2-S10 as separate commits, then S11 with `gpu_probe.sh` (scratchpad) under compute-sanitizer, record in `dev/telemetry/conda_cuda_petsc.md`, archive `AUDIT.md` (D77), close M18, continue M19
## implementer
- These milestones are for Fable (claude-fable-5-1) to implement in ONE fresh session, in roadmap order, committing and pushing each step and not stopping at milestone boundaries (D75). Every plan states mechanism, cost and verdict per step; verdicts are measurable on this machine. Multi-node, multi-GPU and AMD validation is P2 on the HPC (D73), so nothing here waits for hardware that is not present.
- Source of the work: `AUDIT.md` at the repository root (moves to `dev/archive/reviews/p1/audit-2026-09-18.md` at M18 close, D77). Its three structural changes are M20+M23 (preprocessing), M21 (memory), M22 (communication); its release-blocker list is M18.
- Bars that gate every step: residuals bitwise identical under Jacobi at n=2 and n=8 (R8 is binding), `check_ghosts` zero (M19-S2), the M19-S4 round and all-reduce counters, and the serial suite with no reduction in test count.
## position
M1-M14, M17 closed (M12 superseded by M13). M18-M24 opened by D72 from the audit; M16 restated to measurement only (D76). Plans for all seven live in `dev/plans/`. M18-S1 landed (d91c67b8); S2-S10 patched in the working tree (D78-D82), suite and GPU test running.
## evidence
- The scaling attribution in D16 was WRONG and is withdrawn. With the clock pinned the module scales at 102/94/71% (n=2/4/8) and mesh size does not move it (D20, D21). Do not reopen this without reading `dev/telemetry/scaling_attribution.md` first.
- This machine throttles 4400 to 3100 MHz as rank count rises. ANY timing comparison across rank counts is meaningless unless the clock is pinned or the package power held constant; `dev/gotchas.md` carries both methods.
- `dev/petscenv_stock` (XCALibre, MPI, PETSc, Test; no preferences) runs the CPU path; `dev/petscenv_conda_ompi` runs the GPU path with a CUDA-aware Open MPI (`OMPI_MCA_opal_cuda_support=true`); `dev/petscenv` is the custom CUDA-hypre build.
- Per-rank peak RSS 2.79/2.67 GB at 660k cells (4.1 KB/cell); rank-0 global mesh 1.6 KB/cell; the M16 breakdown must precede M21's cures.
## blocked/carried
- Intermittent segfault once in 4 runs on the conda openmpi CUDA-aware direct path, n=2: root-cause is P1-M18-S11.
- `BoomerAMG()` on GPU fields with a host-only hypre SEGFAULTS instead of erroring (D70): P1-M18-S2.
- Memory: 14 GB box. Wrap every large-mesh run in `dev/scripts/memguard.sh`.
- Machine partly reverted at 2026-09-18: turbo on, min_perf 15, governor powersave, but `powerprofilesctl get` still says performance; run `powerprofilesctl set balanced` before timing.
- `xcalibre-dev check` reports pre-existing multi-line comment blocks in `examples/*.jl`; that is P1-M15.
