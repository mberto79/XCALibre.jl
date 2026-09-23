# Active context - memory-traffic scaling (P1-M30)
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m30-threaded-krylov.md
updated: 2026-09-24T22:00:00+01:00
STATE: IDLE
STEP: P1-M30-S2 XVector type and k* primitives (not started; M29 closed, D195)
HEAD: 15c1e8e0
BRANCH: HM/distributed-draft
GATE: per plan p1-m30 (strict: residuals to 8 figures vs pre-M30, 1t not slower beyond ±5%); primitive unit tests at 1 and 8 threads; motorBike 20-iteration smoke 1t/8t via `~/.cache/xcal_m28/chain.sh` vs `dev/telemetry/m28_baseline/`
resume: plan p1-m30 S2 row: `XVector{T} <: DenseVector{T}` in `src/Multithread/` wrapping a `Vector` plus the static row partition of `xmul!`, with `similar`, `size`, `getindex`, `setindex!`, `unsafe_convert` and Krylov's `k*` primitives; unit tests vs `Vector` at 1 and 8 threads

## binding
- User rulings: flat layout adopted, discretisation kernels to read arrays directly as follow-up (D179); compile bar +10% of same-session AoS base (D168); scheme/BC signature change accepted (D175).
- HARD CAP (D101): every verdict run under five minutes; 500-iteration timings one point per command at milestone close (D160).
- Order: M30 (D159), then the P1 phase gate. Commit and push each step.

## position
P1-M28, M29, M31 closed (D189, D195, D188): motorBike 500 iterations 8-rank 53.6 s, 8t 61.4 s, GPU 17.2 s, 1t 135.6 s; readers default to Int32 (D194). 8t main-thread profile: Krylov vector work ~22%, progress strings ~8% (D190, `~/.cache/xcal_m28/close/profile_close_8t.txt`). Timing drivers: `~/.cache/xcal_m28/close/`.

## carried
- Read GPU `run_s` and the `faces=` tag on every `.time` line, not only residuals (D178).
- GPU kernel times: `dev/scripts/gpu_profile.jl`; per-thread local memory: `dev/scripts/ptx_dump.jl` (D187).
- `chain.sh` labels: `2d*` now matches `2db` (before, `2db` silently reran the previous label).
- Envs: `~/.cache/xcal_m28/env` (develops this checkout), `env_base` (875c16bb worktree), `env_test` (suite files), `*_snoop` (SnoopCompile). A dependency change needs `Pkg.resolve()` in env, env_test and every `dev/petscenv*`.
- Source edits must wait until any running smoke or test process has loaded XCALibre (the envs develop this checkout).
- CHANGELOG cites `[#160](@ref)` 14 times; replace with the distributed PR number when opened (D137). M28/M31 layout and BC-signature entries added at S6.
- Memory: 14 GB box, language server ~7 GB: MPI smoke is n=4, wrap in `dev/scripts/memguard.sh`.
