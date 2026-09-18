# Active context - distributed module release polish
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m7-preconditioner-api.md
updated: 2026-09-18T12:00:00+01:00
STATE: IDLE
STEP: P1-M7-S2 retune the freeze defaults (S1 rename landed, D38)
HEAD: c8b6e6e4
BRANCH: HM/distributed-draft
GATE: julia --project=. -e 'using Pkg; Pkg.test()'
resume: take P1-M7-S2 per `dev/plans/p1-m7-preconditioner-api.md`; first check whether the probe logs record a final pressure residual for GAMG freeze=1 vs 25, since s/iter cannot see outer-convergence cost of a stale hierarchy; any new row needs the clock pinned; do NOT open a PR or close the phase
## position
M1-M6 are closed and their exit gate PASSED on 2026-09-18 at 2374a1e7; the evidence rows are in `dev/telemetry/gate_results.md` and need no rerun. M3 was reopened during the phase and its attribution corrected (D19-D23). M7 is the only open milestone, with its scope measured rather than speculative (D34-D37) and its own exit criterion in its plan.
## evidence
- The scaling attribution in D16 was WRONG and is withdrawn. With the clock pinned the module scales at 102/94/71% (n=2/4/8) and mesh size does not move it (D20, D21). Do not reopen this without reading `dev/telemetry/scaling_attribution.md` first.
- This machine throttles 4400 to 3100 MHz as rank count rises. ANY timing comparison across rank counts is meaningless unless the clock is pinned or the package power held constant; `dev/gotchas.md` carries both methods.
- A project environment holding only XCALibre, MPI and PETSc, with no preferences file and no shell configuration, runs the distributed path on stock binaries (D4). `dev/petscenv_stock` is that environment; it is gitignored, and rebuilding it is `Pkg.develop` of this repository plus MPI, PETSc and Test.
- All scaling numbers, both meshes, both codes, live in `dev/telemetry/scaling.csv`; plots come from `julia dev/scripts/plot_scaling.jl` and `SCALING_SUMMARY.md` at the repository root is the readable version. The probe takes `pc=<jacobi|boomeramg|gamg>` and `reuse=<N>` (its CLI name; it passes `freeze=N`).
## blocked/carried
- Machine partly reverted at 2026-09-18: turbo on, min_perf 15, governor powersave, but `powerprofilesctl get` still says performance; the user must run `powerprofilesctl set balanced`.
- `xcalibre-dev check` is INVALID only on pre-existing multi-line comments in 12 `examples/*.jl` headers; out of M7 scope, left for xcalibre-close.
