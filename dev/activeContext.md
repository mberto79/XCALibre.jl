# Active context - distributed module release polish
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m4-distributed-setup-interface.md
updated: 2026-09-17T19:40:00+01:00
STATE: BUILDING
STEP: P1-M4-S5 - bring the remaining examples and the documentation onto the new interface
HEAD: 99d03e04
BRANCH: HM/distributed-draft
GATE: julia --project=dev/petscenv_stock test/distributed/gate.jl
resume: rewrite the cylinder and cascade examples on `distribute(reader; dir)` and `is_root`, then run the distributed gate and time it against Q2
## position
P1-M1 closed. P1-M3's measurement is done and attributed (D9, D10); only the PETSc stage split is outstanding. P1-M4 steps S1 to S4 are written but not yet run. P1-M2's gate has never been executed: it is the next thing to verify.
## evidence
- A project environment holding only XCALibre, MPI and PETSc, with no preferences file and no shell configuration, runs the distributed path on stock binaries: eight PETSc libraries and a bundled OpenMPI (D4). Every claim in P1-M4 about retiring the custom stack rests on this.
- `PETSc.initialize` accepts `options` and `log_view` keywords, so the `PETSC_OPTIONS` environment variable has a direct interface replacement (D5).
- Residuals at one and two ranks agree to thirteen significant figures with `Cg` and `Jacobi` at Float64, so Q1 is met and any scaling loss is cost, not drift (D6).
- The backward-facing-step example's hang was ranks dispatching differently on a value only rank zero held; the fix is committed and P1-M4 makes that bug class unrepresentable.
## blocked/carried
- Nothing. The AMG coarse-solve environment variables were removed on `HM/amg-remove-env-vars` (PR #159) and cherry-picked here (D11).
