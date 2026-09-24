# Active context - P1-M34..M36 pre-merge fixes
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md
updated: 2026-09-24T16:00:00+01:00
STATE: IDLE
STEP: none (P1-M34..M36 closed); next is the P1 exit gate
HEAD: 0259d4c0
BRANCH: HM/distributed-draft
GATE: per plan row; smokes `~/.cache/xcal_m28/chain.sh <dir> cpu1 cpu8 2d gpu` and `~/.cache/xcal_m28/mpi.sh <dir>/mpi4 ~/.cache/xcal_m28/parts_m34_4 4` with `dev/scripts/cmpres.jl` (1t bitwise vs `~/.cache/xcal_m28/m30s4/cpu1.res`, not `m28_baseline` which is 13 figures off since M30; MPI vs `m33/`; 2d 7.6 figures is its R13 band, D197); suite files via `dev/scripts/suite_file.jl` in `~/.cache/xcal_m28/env_test`; distributed via `test/distributed/runtests_mpi.jl` under `dev/scripts/memguard.sh`
resume: user runs `xcalibre-close` for the P1 exit gate (direct invocation only), then the distributed PR (D209); CHANGELOG `[#160]` placeholders get the PR number when it opens (D137)

## binding
- User rulings: stock Int64 PETSc stays (D205); diagonal-only PETSc writes refused (D207); flat layout adopted, discretisation kernels to read arrays directly as follow-up (D179); compile bar +10% of same-session AoS base (D168); scheme/BC signature change accepted (D175).
- HARD CAP (D101): every verdict run under five minutes; 500-iteration timings one point per command at milestone close (D160).
- Order: P1 exit gate via `xcalibre-close`, then the distributed PR (D209). Commit and push each step.

## position
All P1 milestones closed through M36 (D214, D218, D222); pre-merge review findings A1-A5, B1-B7 and C1/C2/C5/C6/D202 resolved (C1 refused, D220). motorBike 500 iterations: 8t 45.6 s, 8-rank MPI 54.6 s, GPU 14.0 s, 1t 137.3 s. Part format is 6: regenerate any older parts.

## carried
- Read GPU `run_s` and the `faces=` tag on every `.time` line, not only residuals (D178).
- GPU kernel times: `dev/scripts/gpu_profile.jl`; per-thread local memory: `dev/scripts/ptx_dump.jl` (D187).
- `chain.sh` labels: `2d*` now matches `2db` (before, `2db` silently reran the previous label).
- Envs: `~/.cache/xcal_m28/env` (develops this checkout), `env_base` (875c16bb worktree), `env_test` (suite files), `env_p322` (PETSc_jll 3.22.2, Int32 PETSc), `*_snoop` (SnoopCompile). Timing drivers: `~/.cache/xcal_m28/close/`. A dependency change needs `Pkg.resolve()` in env, env_test and every `dev/petscenv*`.
- Source edits must wait until any running smoke or test process has loaded XCALibre (the envs develop this checkout).
- CHANGELOG cites `[#160](@ref)` as a placeholder throughout; replace with the distributed PR number when opened (D137).
- Memory: 14 GB box, language server ~7 GB: MPI smoke is n=4, wrap in `dev/scripts/memguard.sh`.
