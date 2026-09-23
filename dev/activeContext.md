# Active context - memory-traffic scaling (P1-M28..M30)
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m28-mesh-soa.md, dev/plans/p1-m29-index-widths.md, dev/plans/p1-m30-threaded-krylov.md
updated: 2026-09-23T12:00:00+01:00
STATE: BUILDING
STEP: P1-M28-S1 baselines at 875c16bb
HEAD: 875c16bb
BRANCH: HM/distributed-draft
GATE: M28 smoke set (plan p1-m28 § gate): 20-iteration motorBike residuals at 1t and 8t, one 2D example, GPU, n=8 MPI, each a separate command under five minutes (D160)
resume: write the smoke script in `dev/scripts/` and record 20-iteration residual baselines at 875c16bb (1t, 8t, 2D, GPU, n=8) before any source change

## binding
- HARD CAP (D101): every verdict run finishes in five minutes; 500-iteration timings run one point per command at milestone close only (D160).
- Order M28 then M29 then M30 (D159); each step committed and pushed, no stop at milestone boundaries. The user expects all three in one session.
- Baselines must be text residual files, not `mesh_*.jld2` or `parts_*`: both die with the mesh type change.

## position
M1-M27 closed; P1 exit gate and close remain the user's call (xcalibre-close). M28-M30 opened by D159.

## carried
- CHANGELOG cites `[#160](@ref)` 12 times for the distributed feature; #160 is the comment-blocks PR (D137); replace with the distributed PR number when opened.
- Parts are `.xdm` format 3; M28-S4/S5 bump it.
- Memory: 14 GB box; wrap large-mesh and MPI runs in `dev/scripts/memguard.sh`. Run `powerprofilesctl set balanced` before timing.
- Benchmark case: `~/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS`; its stock drivers append to recorded datasets, so use scratch copies (gotchas).
