# Active phase roadmap

phase: P1 - distributed module release polish

## Outcome

The distributed module becomes an ordinary, documented XCALibre feature: it installs and runs on stock binaries, is configured only through the Julia interface, cannot be set up rank-divergently, carries a fast regression gate, has its parallel efficiency attributed to a measured cause, and costs serial users nothing. Serves R1 through R10.

## Ordered work

- [x] **P1-M1 - vault and record reconciliation** The development record is one tracked, unmixed vault; `dev/` leaves `.gitignore` while machine-specific state stays out of Git; the working backward-facing-step example fix lands. Exit: `xcalibre-dev check` valid, `dev/` tracked, example committed and pushed. Expected 3 steps.
- [x] **P1-M2 - fast distributed gate** The distributed suite runs from the repository's prescribed test command in a subset chosen to meet Q2, with the rank count passed as a test argument rather than an environment variable. Exit: gate green from `Pkg.test`, timed, recorded in telemetry. Serves R9, R2. DELIVERED: five files at two ranks, 5/5 in 89 s against the 300 s bar.
- [x] **P1-M3 - scaling attribution** The parallel efficiency loss is split across named stages on the backward-facing-step case at Float64 on CPU with a partition-invariant pressure preconditioner, so that the cause is measured rather than listed. Exit: telemetry naming the dominant cost and the rank-invariance check. Serves R7, R8, Q1, Q3. DELIVERED: the ceiling is this machine's memory bandwidth (D16); residuals agree to sixteen significant figures across rank counts; XCALibre is 3 to 5 times faster than OpenFOAM per iteration on the same case while scaling less well above two ranks, which is partly the arithmetic of a faster baseline (D12).
- [x] **P1-M4 - distributed setup interface** One rank-uniform way to obtain a distributed mesh, a root guard for output, options that were environment variables moved onto the interface, and a single documented launch command. Exit: the example reads as a serial case plus mesh and launch; no environment variable is required. Serves R1, R2, R3, R5, R6. DELIVERED in five steps. Plan: `dev/archive/plans/p1/p1-m4-distributed-setup-interface.md`.
- [x] **P1-M5 - distributed dependencies become optional** Decide, on a measurement rather than a principle, whether MPI and Metis stay hard dependencies. The cost to a serial user is measured first: install size, precompilation and load time. Only if that cost is material does the module's MPI surface move behind an extension. Exit: the measurement recorded and the decision taken, plus the refactor if it is warranted. Serves R4. CLOSED WITHOUT THE REFACTOR: the cost is 0.17 s of an 0.809 s load and 27 MB, so the dependencies stay and R4 is amended to a bounded measured cost (D18).
- [x] **P1-M6 - documentation and messages** A distributed user-documentation page, a CHANGELOG entry for the feature, corrected diagnostic messages that point at the documentation, and the documented scope of what is and is not supported. Exit: documentation builds, scope matches the spec's non-requirements. Serves R10. DELIVERED: the distributed page now covers stock-binary requirements, the rank-uniform mesh call, `mpiexecjl`, the supported and unsupported scope, and what to expect of parallel performance; the CHANGELOG records the feature; the hypre and GPU-solve errors point at the documentation.

## Flagged for later

- The distributed module's own kernels scale superlinearly and the loss is entirely in the Krylov solve, which sits under a memory-bandwidth ceiling on this hardware (D14, D16). Whether a node with more memory channels closes the gap with OpenFOAM is unmeasured and needs hardware this project does not have.
- `VecNorm` costs 1.07 ms per call at eight ranks against 0.053 ms at two, and the Krylov solves issue roughly seventy-five reductions per outer iteration. Reducing that count is a real optimisation and is not part of this phase.

## Exit gate

Full serial suite green with no reduction in test count, distributed gate green within Q2, the backward-facing-step example run on stock binaries from a clean checkout, rank invariance at one, two and four ranks meeting Q1, scaling telemetry meeting Q3, and the documentation build passing.
