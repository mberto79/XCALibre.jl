# Active context - P1 complete, exit gate pending
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md
updated: 2026-09-25T06:00:00+01:00
STATE: IDLE
STEP: P1 exit gate (every P1 milestone closed; M30 D200)
HEAD: 9f731499
BRANCH: HM/distributed-draft
GATE: `dev/phaseRoadmap.md` § Exit gate: full serial suite, distributed gate within Q2, BFS example on stock binaries from a clean checkout, rank invariance at 1,2,4 (Q1), scaling telemetry (Q3), docs build
resume: user runs the phase close (`xcalibre-close`, direct invocation only); until then nothing is open in P1

## binding
- User rulings: flat layout adopted, discretisation kernels to read arrays directly as follow-up (D179); compile bar +10% of same-session AoS base (D168); scheme/BC signature change accepted (D175).
- HARD CAP (D101): every verdict run under five minutes; 500-iteration timings one point per command at milestone close (D160).
- Order: P1 phase gate via `xcalibre-close`, then the distributed PR (CHANGELOG `[#160](@ref)` placeholders, D137).

## position
All P1 milestones closed. motorBike 500 iterations: 8t 48.1 s, 8-rank MPI 54.1 s, GPU 17.2 s, 1t 137.6 s (`dev/telemetry/memory_scaling.md` § P1-M30-S6 close). Open finding for a later milestone: progress-output strings ~8% of 8t main thread (D200).

## carried
- Read GPU `run_s` and the `faces=` tag on every `.time` line, not only residuals (D178).
- GPU kernel times: `dev/scripts/gpu_profile.jl`; per-thread local memory: `dev/scripts/ptx_dump.jl` (D187).
- `chain.sh` labels: `2d*` now matches `2db` (before, `2db` silently reran the previous label).
- Envs: `~/.cache/xcal_m28/env` (develops this checkout), `env_base` (875c16bb worktree), `env_test` (suite files), `*_snoop` (SnoopCompile). A dependency change needs `Pkg.resolve()` in env, env_test and every `dev/petscenv*`.
- Source edits must wait until any running smoke or test process has loaded XCALibre (the envs develop this checkout).
- CHANGELOG cites `[#160](@ref)` 14 times; replace with the distributed PR number when opened (D137). M28/M31 layout and BC-signature entries added at S6.
- Memory: 14 GB box, language server ~7 GB: MPI smoke is n=4, wrap in `dev/scripts/memguard.sh`.
