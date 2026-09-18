# Active context - distributed module release polish
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m7-preconditioner-api.md
updated: 2026-09-18T04:00:00+01:00
STATE: BUILDING
STEP: P1-M7 preconditioner API, opened on the scaling findings (D25-D37)
HEAD: c61338d6
BRANCH: HM/distributed-draft
GATE: julia --project=. -e 'using Pkg; Pkg.test()'
resume: take P1-M7-S1, the `reuse` rename, per `dev/plans/p1-m7-preconditioner-api.md`; the M1-M6 exit gate PASSED at 2374a1e7 and needs no rerun; do NOT open a PR or close the phase
## position
M1-M6 are closed and their exit gate PASSED on 2026-09-18 at 2374a1e7; the evidence rows are in `dev/telemetry/gate_results.md` and need no rerun. M3 was reopened during the phase and its attribution corrected (D19-D23). M7 is the only open milestone, with its scope measured rather than speculative (D34-D37) and its own exit criterion in its plan.
## evidence
- The scaling attribution in D16 was WRONG and is withdrawn. With the clock pinned the module scales at 102/94/71% (n=2/4/8) and mesh size does not move it (D20, D21). Do not reopen this without reading `dev/telemetry/scaling_attribution.md` first.
- This machine throttles 4400 to 3100 MHz as rank count rises. ANY timing comparison across rank counts is meaningless unless the clock is pinned or the package power held constant; `dev/gotchas.md` carries both methods.
- A project environment holding only XCALibre, MPI and PETSc, with no preferences file and no shell configuration, runs the distributed path on stock binaries (D4). `dev/petscenv_stock` is that environment; it is gitignored, and rebuilding it is `Pkg.develop` of this repository plus MPI, PETSc and Test.
- All scaling numbers, both meshes, both codes, live in `dev/telemetry/scaling.csv`; plots come from `julia dev/scripts/plot_scaling.jl` and `SCALING_SUMMARY.md` at the repository root is the readable version. The probe takes `pc=<jacobi|boomeramg|gamg>` and `reuse=<N>`.
## blocked/carried
- The machine is in performance mode with turbo disabled for the measurements and MUST be returned to balanced: the revert block is in `dev/gotchas.md`. The user has been given the commands and has not yet confirmed running them.
