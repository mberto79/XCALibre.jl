# Active context - memory-traffic scaling (P1-M28..M30)
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m28-mesh-soa.md, dev/plans/p1-m31-column-reads.md, dev/plans/p1-m29-index-widths.md, dev/plans/p1-m30-threaded-krylov.md
updated: 2026-09-23T19:30:00+01:00
STATE: BUILDING
STEP: P1-M31-S5 apply the flat mesh layout (adopted by the user, D179)
HEAD: 61751a58
BRANCH: HM/distributed-draft
GATE: `~/.cache/xcal_m28/chain.sh <dir> cpu1 cpu8 2d gpu` then `~/.cache/xcal_m28/mpi.sh <dir>/mpi4 <parts> 4` (parts via `motorbike_smoke.jl part 4 <dir>`), compared to `dev/telemetry/m28_baseline/` with `dev/scripts/cmpres.jl`; compile A/B with `XENV=env_base` (875c16bb worktree `~/.cache/xcal_m28/wt_base`), two samples each; bars in plan p1-m28 (D163, D168)
resume: `git apply dev/archive/patches/p1-m31-flat-mesh-columns.diff`, fix `test/distributed/test_offline.jl` (no `getfield(mesh, :faces)` any more), update architecture § mesh storage, then gate per plan p1-m31 S5 row: one chain `chain.sh s5 cpu1 cpu8 2d gpu cpu1b 2db gpub` + `XENV=env_base chain.sh b5 cpu1 2d gpu cpu1b 2db gpub`, fresh parts + `mpi.sh`, offline/partition tests, 8t pinned 100 iterations; then S6

## binding
- User rulings: flat layout adopted, discretisation kernels to read arrays directly as follow-up (D179); compile bar +10% of same-session AoS base (D168); scheme/BC signature change accepted (D175).
- HARD CAP (D101): every verdict run under five minutes; 500-iteration timings one point per command at milestone close (D160).
- Order: P1-M31 S5 (flat patch), S6 (discretise column reads), then P1-M28-S6 close, then M29, then M30 (D159). Commit and push each step.

## position
Branch source is P1-M28-S7 (nested containers): bitwise, runtime −11-13%, but compile +22-33% and GPU 20-iteration run 7.5 vs 4.3 s from a type-unstable `adapt` (D178). Both are fixed by the flat patch (D177, D178), measured but not yet landed; S1 column-read diff stacks on it for S6.

## carried
- Read GPU `run_s` and the `faces=` tag on every `.time` line, not only residuals (D178).
- GPU kernel times: `dev/scripts/gpu_profile.jl` (S6 bar reads `gpu__discretise_*` per call from it).
- Q2 regression: distributed gate 5m04s after S2 (D167); run its n=6 part as a separate command; expected to recover with the compile fix.
- `chain.sh` labels: `2d*` now matches `2db` (before, `2db` silently reran the previous label).
- Envs: `~/.cache/xcal_m28/env` (develops this checkout), `env_base` (875c16bb worktree), `env_test` (suite files), `*_snoop` (SnoopCompile). A dependency change needs `Pkg.resolve()` in env, env_test and every `dev/petscenv*`.
- Source edits must wait until any running smoke or test process has loaded XCALibre (the envs develop this checkout).
- CHANGELOG cites `[#160](@ref)` 12 times; replace with the distributed PR number when opened (D137). No CHANGELOG entry yet for M28; add at milestone close.
- Memory: 14 GB box, language server ~7 GB: MPI smoke is n=4, wrap in `dev/scripts/memguard.sh`.
