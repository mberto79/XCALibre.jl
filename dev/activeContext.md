# Active context - memory-traffic scaling (P1-M29..M30)
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m29-index-widths.md, dev/plans/p1-m30-threaded-krylov.md
updated: 2026-09-24T18:00:00+01:00
STATE: IDLE
STEP: P1-M29-S3 periodic BC maps follow TI + src sweep for stored Int arrays (not started)
HEAD: adb32dbf
BRANCH: HM/distributed-draft
GATE: per plan p1-m29 (strict: CPU residuals bitwise vs preceding step at fixed threads, distributed hashes bitwise at n=2,4); `gate.jl` as two commands under memguard (n=2,3 five files, then n=6 `test_turbulence_sst_wallfn.jl`) via `runtests_mpi.jl` in `~/.cache/xcal_m28/env_test`; `test_restart.jl` under `dev/petscenv_stock` (D183)
resume: plan p1-m29 S3 row: periodic `face_map`, `faceAddress1/2`, `i`/`j` in the mesh `TI`; sweep `src/` for `zeros(Int`, `Int64[`, `Int[`, `Vector{Int}` stored on mesh/equation/solver structs; gate periodic tests bitwise (`3d_incompressible_laminar_cascade_periodic.jl`, distributed `test_periodic.jl`)

## binding
- User rulings: flat layout adopted, discretisation kernels to read arrays directly as follow-up (D179); compile bar +10% of same-session AoS base (D168); scheme/BC signature change accepted (D175).
- HARD CAP (D101): every verdict run under five minutes; 500-iteration timings one point per command at milestone close (D160).
- Order: M29, then M30 (D159). Commit and push each step.

## position
P1-M29-S1, S2 landed (D191, D192; AMG bitwise check: `PSOLVER=amg motorbike_smoke.jl`; part format 5, parts in `~/.cache/xcal_m28/parts_m29_4`). P1-M28 and P1-M31 closed (D188, D189): motorBike 500 iterations 8-rank 53.6 s, 8t 61.4 s, GPU 17.2 s, 1t 135.6 s (`dev/telemetry/memory_scaling.md` § P1-M28-S6 close); scratch drivers for such timings in `~/.cache/xcal_m28/close/`.

## carried
- Read GPU `run_s` and the `faces=` tag on every `.time` line, not only residuals (D178).
- GPU kernel times: `dev/scripts/gpu_profile.jl`; per-thread local memory: `dev/scripts/ptx_dump.jl` (D187).
- `chain.sh` labels: `2d*` now matches `2db` (before, `2db` silently reran the previous label).
- Envs: `~/.cache/xcal_m28/env` (develops this checkout), `env_base` (875c16bb worktree), `env_test` (suite files), `*_snoop` (SnoopCompile). A dependency change needs `Pkg.resolve()` in env, env_test and every `dev/petscenv*`.
- Source edits must wait until any running smoke or test process has loaded XCALibre (the envs develop this checkout).
- CHANGELOG cites `[#160](@ref)` 14 times; replace with the distributed PR number when opened (D137). M28/M31 layout and BC-signature entries added at S6.
- Memory: 14 GB box, language server ~7 GB: MPI smoke is n=4, wrap in `dev/scripts/memguard.sh`.
