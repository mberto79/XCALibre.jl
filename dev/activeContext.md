# Active context - memory-traffic scaling (P1-M28..M30)
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m28-mesh-soa.md, dev/plans/p1-m29-index-widths.md, dev/plans/p1-m30-threaded-krylov.md
updated: 2026-09-23T12:00:00+01:00
STATE: BUILDING
STEP: P1-M28-S4 cell_nsign as Int8
HEAD: 875c16bb
BRANCH: HM/distributed-draft
GATE: `~/.cache/xcal_m28/chain.sh <dir> cpu1 cpu8 2d gpu` (ITER, XENV=env_base for the 875c16bb worktree) then `~/.cache/xcal_m28/mpi.sh <dir>/mpi4 <parts> 4`, compared to `dev/telemetry/m28_baseline/` with `dev/scripts/cmpres.jl` (bars D163)
resume: convert `cell_nsign` to Int8 once in the Mesh2/Mesh3 inner constructor (own type parameter), write/read it as Int8 in `.xdm`, bump `_XDM_FORMAT` to 4; gate with the smoke chain, fresh parts, `test_offline.jl`

## binding
- HARD CAP (D101): every verdict run finishes in five minutes; 500-iteration timings run one point per command at milestone close only (D160).
- Order M28 then M29 then M30 (D159); each step committed and pushed, no stop at milestone boundaries. The user expects all three in one session.
- Baselines are text files in `dev/telemetry/m28_baseline/`; `mesh_*.jld2` and `parts_*` die with the mesh type change. Source edits must wait until any running smoke process has loaded XCALibre (the env develops this checkout).

## position
M1-M27 closed; P1 exit gate and close remain the user's call (xcalibre-close). M28-M30 opened by D159.

## carried
- Q2 regression: distributed gate 5m04s after S2 (D167); run its n=6 part as a separate command until resolved; FIRST line of the report to the user.
- CHANGELOG cites `[#160](@ref)` 12 times for the distributed feature; #160 is the comment-blocks PR (D137); replace with the distributed PR number when opened.
- Parts are `.xdm` format 3; M28-S4/S5 bump it.
- Memory: 14 GB box; wrap large-mesh and MPI runs in `dev/scripts/memguard.sh`. Run `powerprofilesctl set balanced` before timing.
- Benchmark case: `~/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS`; its stock drivers append to recorded datasets, so use scratch copies (gotchas).
