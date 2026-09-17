# Active context - distributed module release polish
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md
updated: 2026-09-17T19:40:00+01:00
STATE: BUILDING
STEP: P1-M5-S1 - measure what MPI and Metis cost a serial user
HEAD: 81a63b6b
BRANCH: HM/distributed-draft
GATE: julia --project=dev/petscenv_stock test/distributed/gate.jl
resume: measure load time, precompilation and artifact size for MPI and Metis, then decide whether R4 is met by a recorded cost or needs the six-step extension refactor
## position
P1-M1 to P1-M4 closed. P1-M5 is next and is a measurement before it is a refactor. P1-M6 owes a CHANGELOG entry for the whole feature and a documented-scope section; the documentation page itself is already rewritten.
## evidence
- A project environment holding only XCALibre, MPI and PETSc, with no preferences file and no shell configuration, runs the distributed path on stock binaries (D4). Every claim about retiring the custom stack rests on this.
- The scaling ceiling on this machine is memory bandwidth, not anything in the module (D16). Do not re-open it here; it needs a node with more memory channels.
- `dev/petscenv_stock` is the environment to measure and gate in; it is gitignored, and rebuilding it is `Pkg.develop` of this repository plus MPI, PETSc and Test.
## blocked/carried
- Nothing. The AMG coarse-solve environment variables were removed on `HM/amg-remove-env-vars` (PR #159) and cherry-picked here (D11).
