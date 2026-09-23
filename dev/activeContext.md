# Active context - memory-traffic scaling (P1-M28..M30)
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m28-mesh-soa.md, dev/plans/p1-m31-column-reads.md, dev/plans/p1-m29-index-widths.md, dev/plans/p1-m30-threaded-krylov.md
updated: 2026-09-23T18:00:00+01:00
STATE: PLANNING
STEP: P1-M31 route after D176 (user decision)
HEAD: 49c1ff23
BRANCH: HM/distributed-draft
GATE: `~/.cache/xcal_m28/chain.sh <dir> cpu1 cpu8 2d gpu` then `~/.cache/xcal_m28/mpi.sh <dir>/mpi4 <parts> 4` (parts via `motorbike_smoke.jl part 4 <dir>`), compared to `dev/telemetry/m28_baseline/` with `dev/scripts/cmpres.jl`; compile A/B with `XENV=env_base` (875c16bb worktree `~/.cache/xcal_m28/wt_base`), two samples each; bars in plan p1-m28 (D163, D168)
resume: ask the user (D176): (1) accept +22-33% compile for −11-13% runtime, withdraw M31, close M28 (S5, S6), optionally land the S1 diff as a code-quality change; (2) revert to AoS; then act on the answer

## binding
- User rule (D168): adopt option B if first-run compile is within +10% of 875c16bb on motorBike 1t and 2D; else option A (whole mesh as plain columns, short accessors allowed at ≤1-2% runtime cost). The D164 band is withdrawn.
- HARD CAP (D101): every verdict run under five minutes; 500-iteration timings one point per command at milestone close (D160).
- Order: P1-M31 (S1 screen, S2, S3), then M28 S5 re-scoped and S6 close, then M29, then M30 (D159). Commit and push each step.

## position
M28 S1-S4, S7 landed (in-house containers, bitwise, runtime −11-13%, D165, D170). Compile +22-33% vs same-session base; every access-side cure refuted (S9 D172, wrapper/boundscheck D174, M31-S1 D176): the cost is carrying the column arrays' types. Route back with the user.

## carried
- Q2 regression: distributed gate 5m04s after S2 (D167); run its n=6 part as a separate command; expected to recover with the compile fix.
- `chain.sh` labels: `2d*` now matches `2db` (before, `2db` silently reran the previous label).
- Envs: `~/.cache/xcal_m28/env` (develops this checkout), `env_base` (875c16bb worktree), `env_test` (suite files), `*_snoop` (SnoopCompile). A dependency change needs `Pkg.resolve()` in env, env_test and every `dev/petscenv*`.
- Source edits must wait until any running smoke or test process has loaded XCALibre (the envs develop this checkout).
- CHANGELOG cites `[#160](@ref)` 12 times; replace with the distributed PR number when opened (D137). No CHANGELOG entry yet for M28; add at milestone close.
- Memory: 14 GB box, language server ~7 GB: MPI smoke is n=4, wrap in `dev/scripts/memguard.sh`.
