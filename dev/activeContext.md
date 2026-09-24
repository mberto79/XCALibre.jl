# Active context - P1-M34..M36 pre-merge fixes
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m34-robustness.md, dev/plans/p1-m35-behaviour-docs.md, dev/plans/p1-m36-cheap-perf.md
updated: 2026-09-24T16:00:00+01:00
STATE: BUILDING
STEP: P1-M34-S2 A3 distributed error paths (gating) then S3 A4 part fingerprint
HEAD: pending-S2
BRANCH: HM/distributed-draft
GATE: per plan row; smokes `~/.cache/xcal_m28/chain.sh <dir> cpu1 cpu8 2d gpu` and `~/.cache/xcal_m28/mpi.sh <dir>/mpi4 ~/.cache/xcal_m28/parts_m29_4 4` with `dev/scripts/cmpres.jl` (1t bitwise vs `~/.cache/xcal_m28/m30s4/cpu1.res`, not `m28_baseline` which is 13 figures off since M30; MPI vs `m33/`; 2d 7.6 figures is its R13 band, D197); suite files via `dev/scripts/suite_file.jl` in `~/.cache/xcal_m28/env_test`; distributed via `test/distributed/runtests_mpi.jl` under `dev/scripts/memguard.sh`
resume: S2 diff in tree (`_on_all_ranks`, `test_failure.jl`, driver timeout): if uncommitted, rerun `test_failure.jl` and `test_restart.jl` (petscenv_stock) and land; then S3 per plan (fingerprint at write + `key=` on `distribute(reader; dir)`, advisor 2026-09-24)

## binding
- User rulings: stock Int64 PETSc stays (D205); diagonal-only PETSc writes refused (D207); flat layout adopted, discretisation kernels to read arrays directly as follow-up (D179); compile bar +10% of same-session AoS base (D168); scheme/BC signature change accepted (D175).
- HARD CAP (D101): every verdict run under five minutes; 500-iteration timings one point per command at milestone close (D160).
- Order: M34, M35 (merge blockers), M36, then P1 exit gate via `xcalibre-close`, then the distributed PR (D209). Commit and push each step. The user expects all three in one fresh session.

## position
All P1 milestones through M33 closed; pre-merge review done (D208, `dev/archive/reviews/p1/pre-merge-review-2026-09-24.md`), its findings are M34 (A1-A5), M35 (B1-B7), M36 (C1, C2, C5, C6, D202). motorBike 500 iterations: 8t 48.1 s, 8-rank MPI 54.1 s, GPU 17.2 s, 1t 137.6 s.

## carried
- Read GPU `run_s` and the `faces=` tag on every `.time` line, not only residuals (D178).
- GPU kernel times: `dev/scripts/gpu_profile.jl`; per-thread local memory: `dev/scripts/ptx_dump.jl` (D187).
- `chain.sh` labels: `2d*` now matches `2db` (before, `2db` silently reran the previous label).
- Envs: `~/.cache/xcal_m28/env` (develops this checkout), `env_base` (875c16bb worktree), `env_test` (suite files), `env_p322` (PETSc_jll 3.22.2, Int32 PETSc), `*_snoop` (SnoopCompile). Timing drivers: `~/.cache/xcal_m28/close/`. A dependency change needs `Pkg.resolve()` in env, env_test and every `dev/petscenv*`.
- Source edits must wait until any running smoke or test process has loaded XCALibre (the envs develop this checkout).
- CHANGELOG cites `[#160](@ref)` as a placeholder throughout; replace with the distributed PR number when opened (D137).
- Memory: 14 GB box, language server ~7 GB: MPI smoke is n=4, wrap in `dev/scripts/memguard.sh`.
