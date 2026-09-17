# Active phase roadmap

phase: P1 - distributed module release polish

## Outcome

The distributed module becomes an ordinary, documented XCALibre feature: it installs and runs on stock binaries, is configured only through the Julia interface, cannot be set up rank-divergently, carries a fast regression gate, has its parallel efficiency attributed to a measured cause, and costs serial users nothing. Serves R1 through R10.

## Ordered work

- [ ] **P1-M1 - vault and record reconciliation** The development record is one tracked, unmixed vault; `dev/` leaves `.gitignore` while machine-specific state stays out of Git; the working backward-facing-step example fix lands. Exit: `xcalibre-dev check` valid, `dev/` tracked, example committed and pushed. Expected 3 steps.
- [ ] **P1-M2 - fast distributed gate** The distributed suite runs from the repository's prescribed test command in a subset chosen to meet Q2, with the rank count passed as a test argument rather than an environment variable. Exit: gate green from `Pkg.test`, timed, recorded in telemetry. Serves R9, R2. Expected 3 steps.
- [ ] **P1-M3 - scaling attribution** The parallel efficiency loss is split across named stages on the backward-facing-step case at Float64 on CPU with a partition-invariant pressure preconditioner, so that the cause is measured rather than listed. Exit: telemetry naming the dominant cost and the rank-invariance check. Serves R7, R8, Q1, Q3. Expected 4 steps.
- [ ] **P1-M4 - distributed setup interface** One rank-uniform way to obtain a distributed mesh, a root guard for output, options that were environment variables moved onto the interface, and a single documented launch command. Exit: the example reads as a serial case plus mesh and launch; no environment variable is required. Serves R1, R2, R3, R5, R6. Expected 5 steps. Plan: `dev/plans/p1-m4-distributed-setup-interface.md`.
- [ ] **P1-M5 - distributed dependencies become optional** MPI and Metis stop being hard dependencies of a serial install while the names, types and solver seams the rest of the package dispatches on stay where they are. Exit: serial suite green without the distributed dependencies loaded; distributed gate green with them. Serves R4. Expected 4 steps.
- [ ] **P1-M6 - documentation and messages** A distributed user-documentation page, a CHANGELOG entry for the feature, corrected diagnostic messages that point at the documentation, and the documented scope of what is and is not supported. Exit: documentation builds, scope matches the spec's non-requirements. Serves R10. Expected 3 steps.

## Exit gate

Full serial suite green with no reduction in test count, distributed gate green within Q2, the backward-facing-step example run on stock binaries from a clean checkout, rank invariance at one, two and four ranks meeting Q1, scaling telemetry meeting Q3, and the documentation build passing.
