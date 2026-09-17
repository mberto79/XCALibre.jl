# Active context - distributed module release polish
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md
updated: 2026-09-17T19:40:00+01:00
STATE: BUILDING
STEP: P1-M2-S1 - wire a fast distributed gate into the prescribed test command
HEAD: b0d475bc
BRANCH: HM/distributed-draft
GATE: julia --project=dev/petscenv_stock test/distributed/gate.jl
resume: run the distributed gate once the scaling probe releases the cores, time it against Q2, then commit and push P1-M2
## position
P1-M1 closed. P1-M2's driver split and gate file are written but not yet run. P1-M3's probe is measuring in the background on the packaged 10 mm mesh.
## evidence
- A project environment holding only XCALibre, MPI and PETSc, with no preferences file and no shell configuration, runs the distributed path on stock binaries: eight PETSc libraries and a bundled OpenMPI (D4). Every claim in P1-M4 about retiring the custom stack rests on this.
- `PETSc.initialize` accepts `options` and `log_view` keywords, so the `PETSC_OPTIONS` environment variable has a direct interface replacement (D5).
- Residuals at one and two ranks agree to thirteen significant figures with `Cg` and `Jacobi` at Float64, so Q1 is met and any scaling loss is cost, not drift (D6).
- The backward-facing-step example's hang was ranks dispatching differently on a value only rank zero held; the fix is committed and P1-M4 makes that bug class unrepresentable.
## blocked/carried
- Whether the AMG device coarse-solve environment variables in `src/Solve/AMG/` belong to this phase is the user's call; they arrived with a different feature.
