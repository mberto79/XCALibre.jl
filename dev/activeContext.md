# Active context - distributed module release polish
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md
updated: 2026-09-18T18:00:00+01:00
STATE: PLANNING
STEP: P1-M11 - reduction count above four ranks
HEAD: d3d7c947
BRANCH: HM/distributed-draft
GATE: julia --project=. -e 'using Pkg; Pkg.test()'
resume: P1-M11: measured 272 PETSc reductions/outer iter at n=2 (CG ~85 its x 3); candidate 1 = CG natural norm (matches Krylov.jl sqrt(r'Mr) stopping), candidate 2 = + -ksp_cg_single_reduction; measure n=2 and n=8 under dev/scripts/equal_thermal.sh
## position
M1-M8 closed. User ruled P1 defers nothing (D48) and GPU runs never fall back to host (D47): M9-M12 opened in `dev/phaseRoadmap.md`.
## evidence
- The scaling attribution in D16 was WRONG and is withdrawn. With the clock pinned the module scales at 102/94/71% (n=2/4/8) and mesh size does not move it (D20, D21). Do not reopen this without reading `dev/telemetry/scaling_attribution.md` first.
- This machine throttles 4400 to 3100 MHz as rank count rises. ANY timing comparison across rank counts is meaningless unless the clock is pinned or the package power held constant; `dev/gotchas.md` carries both methods.
- A project environment holding only XCALibre, MPI and PETSc, with no preferences file and no shell configuration, runs the distributed path on stock binaries (D4). `dev/petscenv_stock` is that environment; it is gitignored, and rebuilding it is `Pkg.develop` of this repository plus MPI, PETSc and Test.
- All scaling numbers, both meshes, both codes, live in `dev/telemetry/scaling.csv`; plots come from `julia dev/scripts/plot_scaling.jl` and `SCALING_SUMMARY.md` at the repository root is the readable version. The probe takes `pc=<jacobi|boomeramg|gamg>` and `reuse=<N>` (its CLI name; it passes `freeze=N`).
## blocked/carried
- Machine partly reverted at 2026-09-18: turbo on, min_perf 15, governor powersave, but `powerprofilesctl get` still says performance; the user must run `powerprofilesctl set balanced`.
- `xcalibre-dev check` is INVALID only on pre-existing multi-line comments in 12 `examples/*.jl` headers; out of M7 scope, left for xcalibre-close.
