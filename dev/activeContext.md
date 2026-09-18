# Active context - distributed module release polish
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md, dev/plans/p1-m17-setup-guide.md
updated: 2026-09-18T19:30:00+01:00
STATE: PLANNING
STEP: P1-M17 - MPI and PETSc setup guide (then P1-M15, P1-M16)
HEAD: 54546b72
BRANCH: HM/distributed-draft
GATE: julia --project=. -e 'using Pkg; Pkg.test()'
resume: start P1-M17 from `dev/plans/p1-m17-setup-guide.md` (research findings on MPIPreferences/MPItrampoline, PETSc.jl custom library, prebuilt CUDA PETSc): write the setup section of docs/src/user_guide/6_distributed_mpi.md, verify each procedure on this machine, build docs; then P1-M15 (example comments, check until VALID) and P1-M16 (memory breakdown first; shared-code cures on one branch off main, D67)
## position
M1-M14 closed (M12 superseded by M13). This session: M9 no host fallback, M10 device-resident PETSc, M11 tolerances match Krylov.jl, M13 preconditioner guidance, M14 Int32 PETSc (D47-D66).
## evidence
- The scaling attribution in D16 was WRONG and is withdrawn. With the clock pinned the module scales at 102/94/71% (n=2/4/8) and mesh size does not move it (D20, D21). Do not reopen this without reading `dev/telemetry/scaling_attribution.md` first.
- This machine throttles 4400 to 3100 MHz as rank count rises. ANY timing comparison across rank counts is meaningless unless the clock is pinned or the package power held constant; `dev/gotchas.md` carries both methods.
- A project environment holding only XCALibre, MPI and PETSc, with no preferences file and no shell configuration, runs the distributed path on stock binaries (D4). `dev/petscenv_stock` is that environment; it is gitignored, and rebuilding it is `Pkg.develop` of this repository plus MPI, PETSc and Test.
- All scaling numbers, both meshes, both codes, live in `dev/telemetry/scaling.csv`; plots come from `julia dev/scripts/plot_scaling.jl` and `SCALING_SUMMARY.md` at the repository root is the readable version. The probe takes `pc=<jacobi|boomeramg|gamg>` and `reuse=<N>` (its CLI name; it passes `freeze=N`).
## blocked/carried
- Memory: 14 GB box; the 4 mm BFS with AMG at 8 ranks OOM-killed VS Code. Wrap every large-mesh run in `dev/scripts/memguard.sh` (P1-M16 measures under it too).
- Machine partly reverted at 2026-09-18: turbo on, min_perf 15, governor powersave, but `powerprofilesctl get` still says performance; the user must run `powerprofilesctl set balanced`.
- `xcalibre-dev check` is INVALID only on multi-line comment blocks in `examples/*.jl`; that is P1-M15.
