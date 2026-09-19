# Active context - distributed module release polish
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m19-gate-widening.md
updated: 2026-09-19T12:00:00+01:00
STATE: BUILDING
STEP: P1-M19-S1..S4 - gate widening
HEAD: 9f9dec5e
BRANCH: HM/distributed-draft
GATE: julia --project=. -e 'using Pkg; Pkg.test()'
resume: M18 is closed pending its commit; implement `dev/plans/p1-m19-gate-widening.md` S1-S4 (gate at n=2,3; `check_ghosts` + `test_ghosts.jl`; `test_invariance.jl`; exchange/all-reduce counters in `test_perf.jl`), gate each with `julia --project=dev/petscenv_stock --startup-file=no test/distributed/runtests_mpi.jl --ranks=1,2,3,5 <files>`, land, then M20
## implementer
- These milestones are for Fable (claude-fable-5-1) to implement in ONE fresh session, in roadmap order, committing and pushing each step and not stopping at milestone boundaries (D75). Every plan states mechanism, cost and verdict per step; verdicts are measurable on this machine. Multi-node, multi-GPU and AMD validation is P2 on the HPC (D73), so nothing here waits for hardware that is not present.
- Source of the work: `dev/archive/reviews/p1/audit-2026-09-18.md` (archived at M18 close, D77). Its three structural changes are M20+M23 (preprocessing), M21 (memory), M22 (communication); its release-blocker list is M18.
- Bars that gate every step: residuals bitwise identical under Jacobi at n=2 and n=8 (R8 is binding), `check_ghosts` zero (M19-S2), the M19-S4 round and all-reduce counters, and the serial suite with no reduction in test count.
## position
M1-M14, M17 closed (M12 superseded by M13). M18-M24 opened by D72 from the audit; M16 restated to measurement only (D76). Plans for all seven live in `dev/plans/`. M18 DELIVERED 2026-09-19 (D78-D85): suite 12/12 at n=1,2, `test_gpu.jl` n=1,2 on both CUDA envs, the segfault was PETSc resetting Julia's signal handlers (D85).
## evidence
- The scaling attribution in D16 was WRONG and is withdrawn. With the clock pinned the module scales at 102/94/71% (n=2/4/8) and mesh size does not move it (D20, D21). Do not reopen this without reading `dev/telemetry/scaling_attribution.md` first.
- This machine throttles 4400 to 3100 MHz as rank count rises. ANY timing comparison across rank counts is meaningless unless the clock is pinned or the package power held constant; `dev/gotchas.md` carries both methods.
- `dev/petscenv_stock` (XCALibre, MPI, PETSc, Test; no preferences) runs the CPU path; `dev/petscenv_conda_ompi` runs the GPU path with a CUDA-aware Open MPI (`OMPI_MCA_opal_cuda_support=true`); `dev/petscenv` is the custom CUDA-hypre build.
- Per-rank peak RSS 2.79/2.67 GB at 660k cells (4.1 KB/cell); rank-0 global mesh 1.6 KB/cell; the M16 breakdown must precede M21's cures.
## blocked/carried
- Memory: 14 GB box. Wrap every large-mesh run in `dev/scripts/memguard.sh`.
- Machine partly reverted at 2026-09-18: turbo on, min_perf 15, governor powersave, but `powerprofilesctl get` still says performance; run `powerprofilesctl set balanced` before timing.
- `xcalibre-dev check` reports pre-existing multi-line comment blocks in `examples/*.jl`; that is P1-M15.
