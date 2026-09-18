# Active context - distributed module release polish
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md
updated: 2026-09-18T03:05:00+01:00
STATE: BUILDING
STEP: P1-M7 preconditioner API, opened on the scaling findings (D25-D30)
HEAD: e67b898f
BRANCH: HM/distributed-draft
GATE: julia --project=. -e 'using Pkg; Pkg.test()'
resume: take P1-M7 step 1, the `reuse` rename, on the measured recommendation that GAMG is the AMG to document (D31-D33); the serial suite still needs one green rerun after the `Random` fix (D24); do NOT open a PR or close the phase
## position
M1-M6 closed; M3 was reopened and its attribution corrected (D19-D23). M7 is OPEN: the
preconditioner API, opened because the measurements exposed that `reuse` freezes rather than
updates (D28) and that our AMG is the weak configuration (D30). Exit gate otherwise green:
distributed gate 89 s, documentation build, example acceptance at n=2 and n=4 on stock binaries
from clean directories. The serial suite failed once on an undeclared test dependency (D24),
fixed, rerun outstanding.
## evidence
- The scaling attribution in D16 was WRONG and is withdrawn. With the clock pinned the module
  scales at 102/94/71% (n=2/4/8) and mesh size does not move it (D20, D21). Do not reopen this
  without reading `dev/telemetry/scaling_attribution.md` first.
- This machine throttles 4400 to 3100 MHz as rank count rises. ANY timing comparison across
  rank counts is meaningless unless the clock is pinned or the package power held constant.
- A project environment holding only XCALibre, MPI and PETSc, with no preferences file and no
  shell configuration, runs the distributed path on stock binaries (D4).
- `dev/petscenv_stock` is the environment to measure and gate in; it is gitignored, and
  rebuilding it is `Pkg.develop` of this repository plus MPI, PETSc and Test.
- All scaling numbers, both meshes, both codes, live in `dev/telemetry/scaling.csv`; plots come
  from `julia dev/scripts/plot_scaling.jl`. `SCALING_SUMMARY.md` at the repository root is the
  readable version. The probe takes `pc=<jacobi|boomeramg|gamg>` and `reuse=<N>`.
## blocked/carried
- The machine is currently in performance mode with turbo disabled for the measurements. It
  MUST be returned to balanced: see the revert block in `dev/gotchas.md`.
