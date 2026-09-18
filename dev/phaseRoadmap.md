# Active phase roadmap

phase: P1 - distributed module release polish

## Outcome

The distributed module becomes an ordinary, documented XCALibre feature: it installs and runs on stock binaries, is configured only through the Julia interface, cannot be set up rank-divergently, carries a fast regression gate, has its parallel efficiency attributed to a measured cause, and costs serial users nothing. Serves R1 through R10.

## Ordered work

- [x] **P1-M1 - vault and record reconciliation** The development record is one tracked, unmixed vault; `dev/` leaves `.gitignore` while machine-specific state stays out of Git; the working backward-facing-step example fix lands. Exit: `xcalibre-dev check` valid, `dev/` tracked, example committed and pushed. Expected 3 steps.
- [x] **P1-M2 - fast distributed gate** The distributed suite runs from the repository's prescribed test command in a subset chosen to meet Q2, with the rank count passed as a test argument rather than an environment variable. Exit: gate green from `Pkg.test`, timed, recorded in telemetry. Serves R9, R2. DELIVERED: five files at two ranks, 5/5 in 89 s against the 300 s bar.
- [x] **P1-M3 - scaling attribution** The parallel efficiency loss is split across named stages on the backward-facing-step case at Float64 on CPU with a partition-invariant pressure preconditioner, so that the cause is measured rather than listed. Exit: telemetry naming the dominant cost and the rank-invariance check. Serves R7, R8, Q1, Q3. DELIVERED, then REOPENED and CORRECTED: the first attribution blamed memory bandwidth and was wrong (D16 withdrawn by D19). With the CPU clock pinned, efficiency is 102/94/71% at n=2/4/8 and is unchanged by a 2.6x larger mesh (D20, D21); the apparent loss was this laptop throttling 4400 to 3100 MHz as rank count rose. XCALibre ties OpenFOAM's production GAMG per iteration using only Jacobi and scales better at every rank count (D22). Residuals agree to 15 significant figures at every rank count. The residual above four ranks is `VecNorm` reduction cost and is deferred (D23). Telemetry: `dev/telemetry/scaling_attribution.md`.
- [x] **P1-M4 - distributed setup interface** One rank-uniform way to obtain a distributed mesh, a root guard for output, options that were environment variables moved onto the interface, and a single documented launch command. Exit: the example reads as a serial case plus mesh and launch; no environment variable is required. Serves R1, R2, R3, R5, R6. DELIVERED in five steps. Plan: `dev/archive/plans/p1/p1-m4-distributed-setup-interface.md`.
- [x] **P1-M5 - distributed dependencies become optional** Decide, on a measurement rather than a principle, whether MPI and Metis stay hard dependencies. The cost to a serial user is measured first: install size, precompilation and load time. Only if that cost is material does the module's MPI surface move behind an extension. Exit: the measurement recorded and the decision taken, plus the refactor if it is warranted. Serves R4. CLOSED WITHOUT THE REFACTOR: the cost is 0.17 s of an 0.809 s load and 27 MB, so the dependencies stay and R4 is amended to a bounded measured cost (D18).
- [x] **P1-M6 - documentation and messages** A distributed user-documentation page, a CHANGELOG entry for the feature, corrected diagnostic messages that point at the documentation, and the documented scope of what is and is not supported. Exit: documentation builds, scope matches the spec's non-requirements. Serves R10. DELIVERED: the distributed page now covers stock-binary requirements, the rank-uniform mesh call, `mpiexecjl`, the supported and unsupported scope, and what to expect of parallel performance; the CHANGELOG records the feature; the hypre and GPU-solve errors point at the documentation.

- [x] **P1-M7 - preconditioner API, informed by the scaling measurements** The distributed preconditioner API says what it does to the hierarchy, and the documented default is the one that wins. Delivered: `reuse` renamed `freeze` (D38, D41), GAMG default freeze 25 with BoomerAMG kept at 10 (D39), GAMG recommended with the rank-invariance caveat documented (D40). Plan: `dev/archive/plans/p1/p1-m7-preconditioner-api.md`. Exit: a measured recommendation for the distributed pressure preconditioner, the naming settled, and the docstrings matching the semantics. Serves R5, R10. Expected 4 steps.

## Flagged for later

- BoomerAMG and pipelined CG were both tested as remedies for the high-rank loss and both are slower here (D26, D27). AMG deserves a retest on a stiffer, larger problem, where its setup can amortise; that needs a case this phase does not have.
- Above four ranks the residual loss is `VecNorm` reduction cost: 53% of the eight-rank solve, 38.2x rank imbalance, roughly seventy-five reductions per outer iteration (D23). Cutting that count is a real optimisation and belongs to a later phase.

- The AMG freeze optimum was swept on 499,503 cells only; whether it moves with mesh size, and so whether the interval should be adaptive, is open (`dev/archive/plans/p1/p1-m7-preconditioner-api.md`, open questions).

## Exit gate

PASSED 2026-09-18 for M1-M6, at commit 2374a1e7. M7 closed 2026-09-18 on serial suite 1555/1555 (incl. distributed gate 5/5) and documentation build 0 errors; `dev/telemetry/gate_results.md`. Evidence in `dev/telemetry/gate_results.md`: serial suite 1549/1549, distributed gate 5/5 in 187 s, documentation build 0 errors, example at n=2 and n=4 on stock binaries from clean directories, rank invariance to 15 significant figures under Jacobi, scaling telemetry in `scaling_attribution.md`.

Full serial suite green with no reduction in test count, distributed gate green within Q2, the backward-facing-step example run on stock binaries from a clean checkout, rank invariance at one, two and four ranks meeting Q1, scaling telemetry meeting Q3, and the documentation build passing.
