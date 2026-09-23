# Active context - memory-traffic scaling (P1-M28..M30)
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m28-mesh-soa.md, dev/plans/p1-m31-column-reads.md, dev/plans/p1-m29-index-widths.md, dev/plans/p1-m30-threaded-krylov.md
updated: 2026-09-24T00:30:00+01:00
STATE: BUILDING
STEP: P1-M31-S6 discretisation path reads arrays directly
HEAD: bdfc8606
BRANCH: HM/distributed-draft
GATE: `~/.cache/xcal_m28/chain.sh <dir> cpu1 cpu8 2d gpu` then `~/.cache/xcal_m28/mpi.sh <dir>/mpi4 <parts> 4` (parts via `motorbike_smoke.jl part 4 <dir>`), compared to `dev/telemetry/m28_baseline/` with `dev/scripts/cmpres.jl`; compile A/B with `XENV=env_base` (875c16bb worktree `~/.cache/xcal_m28/wt_base`), two samples each; bars in plan p1-m28 (D163, D168)
resume: apply `dev/archive/patches/p1-m31-s1-column-reads.diff` on HEAD (expect conflicts in `Mesh_0_types.jl`-adjacent code), then per plan p1-m31 S6 row

## binding
- User rulings: flat layout adopted, discretisation kernels to read arrays directly as follow-up (D179); compile bar +10% of same-session AoS base (D168); scheme/BC signature change accepted (D175).
- HARD CAP (D101): every verdict run under five minutes; 500-iteration timings one point per command at milestone close (D160).
- Order: P1-M31 S6 (discretise column reads), then P1-M28-S6 close, then M29, then M30 (D159). Commit and push each step.

## position
Flat mesh layout landed (P1-M31-S5, D180), type tags as one-element vectors (S7, D182); S1 column-read diff stacks on it for S6.

## carried
- Read GPU `run_s` and the `faces=` tag on every `.time` line, not only residuals (D178).
- GPU kernel times: `dev/scripts/gpu_profile.jl` (S6 bar reads `gpu__discretise_*` per call from it).
- Q2 regression: distributed gate 5m04s after S2 (D167); run its n=6 part as a separate command; expected to recover with the compile fix.
- `chain.sh` labels: `2d*` now matches `2db` (before, `2db` silently reran the previous label).
- Envs: `~/.cache/xcal_m28/env` (develops this checkout), `env_base` (875c16bb worktree), `env_test` (suite files), `*_snoop` (SnoopCompile). A dependency change needs `Pkg.resolve()` in env, env_test and every `dev/petscenv*`.
- Source edits must wait until any running smoke or test process has loaded XCALibre (the envs develop this checkout).
- CHANGELOG cites `[#160](@ref)` 12 times; replace with the distributed PR number when opened (D137). No CHANGELOG entry yet for M28; add at milestone close.
- Memory: 14 GB box, language server ~7 GB: MPI smoke is n=4, wrap in `dev/scripts/memguard.sh`.
