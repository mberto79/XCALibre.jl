# Active context - distributed module release polish
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m21-owned-row-system.md
updated: 2026-09-19T17:30:00+01:00
STATE: IDLE
STEP: P1-M21-S1 - n_rows seam, owned-row CSR
HEAD: 7cf85354
BRANCH: HM/distributed-draft
GATE: julia --project=. -e 'using Pkg; Pkg.test()'
resume: start P1-M21-S1 per `dev/plans/p1-m21-owned-row-system.md`; then M21 S3-S6, M22, M23, M24, M15 in `dev/phaseRoadmap.md` order
## implementer
- These milestones are for Fable (claude-fable-5-1) to implement in ONE fresh session, in roadmap order, committing and pushing each step and not stopping at milestone boundaries (D75). Every plan states mechanism, cost and verdict per step; verdicts are measurable on this machine. Multi-node, multi-GPU and AMD validation is P2 on the HPC (D73), so nothing here waits for hardware that is not present.
- Source of the work: `dev/archive/reviews/p1/audit-2026-09-18.md` (archived at M18 close, D77). Its three structural changes are M20+M23 (preprocessing), M21 (memory), M22 (communication); its release-blocker list is M18.
- Bars that gate every step: residuals bitwise identical under Jacobi at n=2 and n=8 (R8 is binding), `check_ghosts` zero (M19-S2), the M19-S4 round and all-reduce counters, and the serial suite with no reduction in test count.
## position
M1-M14, M16, M17 closed (M12 superseded by M13). M18-M24 opened by D72 from the audit; M16 restated to measurement only (D76). Plans for the open ones live in `dev/plans/`. M18 DELIVERED 2026-09-19 (D78-D85): suite 12/12 at n=1,2, `test_gpu.jl` n=1,2 on both CUDA envs, the segfault was PETSc resetting Julia's signal handlers (D85). M19 DELIVERED (83d10833, D87-D89) and M20 DELIVERED (D86) 2026-09-19.
## evidence
- The scaling attribution in D16 was WRONG and is withdrawn. With the clock pinned the module scales at 102/94/71% (n=2/4/8) and mesh size does not move it (D20, D21). Do not reopen this without reading `dev/telemetry/scaling_attribution.md` first.
- This machine throttles 4400 to 3100 MHz as rank count rises. ANY timing comparison across rank counts is meaningless unless the clock is pinned or the package power held constant; `dev/gotchas.md` carries both methods.
- `dev/petscenv_stock` (XCALibre, MPI, PETSc, Test; no preferences) runs the CPU path; `dev/petscenv_conda_ompi` runs the GPU path with a CUDA-aware Open MPI (`OMPI_MCA_opal_cuda_support=true`); `dev/petscenv` is the custom CUDA-hypre build.
- Memory baseline (D90): 1950 MB peak per rank at 4 mm n=2 = ~790 MB fixed + 1.85 KB per local cell; M21 measures against `dev/telemetry/memory_breakdown.md`. Parts for 5 and 4 mm n=2 with the checked header are in `~/.cache/xcal_mem_probe/`.
## blocked/carried
- Memory: 14 GB box. Wrap every large-mesh run in `dev/scripts/memguard.sh`.
- Machine partly reverted at 2026-09-18: turbo on, min_perf 15, governor powersave, but `powerprofilesctl get` still says performance; run `powerprofilesctl set balanced` before timing.
- `xcalibre-dev check` reports pre-existing multi-line comment blocks in `examples/*.jl`; that is P1-M15.
