# Active context - P1 complete, pre-merge review
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md
updated: 2026-09-24T12:00:00+01:00
STATE: REVIEWING
STEP: pre-merge review of HM/distributed-draft (user request; every P1 milestone closed, last D207)
HEAD: c925df98
BRANCH: HM/distributed-draft
GATE: `dev/phaseRoadmap.md` § Exit gate: full serial suite, distributed gate within Q2, BFS example on stock binaries from a clean checkout, rank invariance at 1,2,4 (Q1), scaling telemetry (Q3), docs build
resume: collect the review agents' findings (performance, behaviour changes vs main, production robustness), report them to the user, who decides which become milestones before `xcalibre-close`

## binding
- User rulings: flat layout adopted, discretisation kernels to read arrays directly as follow-up (D179); compile bar +10% of same-session AoS base (D168); scheme/BC signature change accepted (D175).
- HARD CAP (D101): every verdict run under five minutes; 500-iteration timings one point per command at milestone close (D160).
- Order: review findings triaged by the user, then P1 phase gate via `xcalibre-close`, then the distributed PR (CHANGELOG `[#160](@ref)` placeholders, D137).

## position
All P1 milestones closed. motorBike 500 iterations: 8t 48.1 s, 8-rank MPI 54.1 s, GPU 17.2 s, 1t 137.6 s (`dev/telemetry/memory_scaling.md` § P1-M30-S6 close). Open findings in `dev/roadmap.md` § flagged (D202, D205). User rulings this round: stock Int64 PETSc stays (D205), diagonal-only PETSc writes refused (D207).

## carried
- Read GPU `run_s` and the `faces=` tag on every `.time` line, not only residuals (D178).
- GPU kernel times: `dev/scripts/gpu_profile.jl`; per-thread local memory: `dev/scripts/ptx_dump.jl` (D187).
- `chain.sh` labels: `2d*` now matches `2db` (before, `2db` silently reran the previous label).
- Envs: `~/.cache/xcal_m28/env` (develops this checkout), `env_base` (875c16bb worktree), `env_test` (suite files), `env_p322` (PETSc_jll 3.22.2, Int32 PETSc), `*_snoop` (SnoopCompile). Timing drivers: `~/.cache/xcal_m28/close/`. A dependency change needs `Pkg.resolve()` in env, env_test and every `dev/petscenv*`.
- Source edits must wait until any running smoke or test process has loaded XCALibre (the envs develop this checkout).
- CHANGELOG cites `[#160](@ref)` as a placeholder throughout; replace with the distributed PR number when opened (D137).
- Memory: 14 GB box, language server ~7 GB: MPI smoke is n=4, wrap in `dev/scripts/memguard.sh`.
