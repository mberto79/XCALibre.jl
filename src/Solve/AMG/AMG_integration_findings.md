# AMG Integration Findings

Numbers -> AMG_integration_results.json. This file = decisions/gotchas only, one-liners, ## per phase.
Pre-unification history: Archived/AMG_perf_findings.md (P1-P13), Archived/Implementation/arch_findings.md.

## Carried-over constraints (do not re-derive)
- 9 invariants listed in AMG_integration_plan.md must survive refactor verbatim.
- ext/ import list = rename freeze (plan §renames).
- P13: Jacobi kernels stay residual-form; GS/SOR host sweeps still old form (CPU/F64 only, acceptable).
