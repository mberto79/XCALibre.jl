# XCALibre.jl distributed (MPI) module - requirements

WHAT THE DELIVERED MODULE MUST BE TRUE OF. Mechanisms, names and call order are `dev/architecture.md`; how to build, run, gate and measure is `dev/gotchas.md`; why a mechanism was chosen or refused is `dev/decisions.md`.

BINDING CORE: R3 and R8. No session relaxes rank-uniform setup or rank-invariant results. Every other clause is amendable under its own ID with a decision saying why.

## goal

A user who already writes serial XCALibre cases can run the same case across MPI ranks on a stock Julia install, launch it with one command, and read the same results. Correctness across rank counts outranks performance, and performance outranks interface convenience.

## vocabulary

- **Rank-uniform** - every MPI rank executes the same statements with the same types, so no rank can take a branch the others do not.
- **Stock binaries** - the MPI and PETSc libraries shipped by `MPICH_jll` and `PETSc_jll` with no `LocalPreferences.toml` and no shell configuration.
- **Distributed gate** - the automated test set that must pass before distributed changes land.
- **Attributed** - a measured cost split across named stages, not a hypothesis list.

## priority

Correctness (R3, R8) outranks everything. Then the stock-binary and serial-cost promises (R1, R4), then interface and launch (R2, R5, R6), then measured speedup (R7). Where several numbers move at once, a regression in R8 refuses the change outright; every other quantity is read as a band with an attribution.

## quality bars

- **Q1 rank invariance** - converged residual histories and final residuals agree to at least four significant figures across rank counts at Float64.
- **Q2 gate cost** - the distributed gate completes within five minutes of wall-clock time on a four-core developer machine, including Julia compilation.
- **Q3 scaling** - strong-scaling efficiency is recorded for each measured rank count, and any efficiency below eighty percent names the dominant cost that produced it.

## requirements

R1 STOCK BINARIES SUFFICE - a distributed Float64 CPU simulation runs on the packages' own bundled MPI and PETSc binaries with no machine-specific configuration and no separate project environment.
R2 NO ENVIRONMENT-VARIABLE CONTROL - every option that changes what a distributed simulation computes is reachable from the documented Julia interface, and no environment variable must be set for a supported configuration to behave correctly.
R3 RANK-UNIFORM SETUP - the documented way to set up a distributed case cannot be written so that ranks follow different paths through it, and a case that runs on one rank runs unchanged on many.
R4 THE SERIAL COST OF DISTRIBUTED SUPPORT IS SMALL AND KNOWN - what a serial install pays in load time and download size for dependencies only distributed runs call is measured, recorded and a small fraction of the package's own, rather than assumed to be zero.
R5 FAMILIAR INTERFACE - a distributed case differs from its serial counterpart only in how the mesh is obtained and how the run is launched; physics, boundary, scheme, solver and runtime setup are written identically.
R6 LAUNCH IS ONE COMMAND - starting a distributed simulation from a shell is a single documented command naming the script, the rank count and the environment.
R7 PARALLEL SPEEDUP - wall-clock time per iteration falls as ranks are added over the supported range, and any departure from ideal is attributed to a measured cause.
R8 RESULTS ARE RANK-INVARIANT - converged fields and residual histories agree across rank counts to solver tolerance.
R9 REGRESSION NET - the distributed feature has an automated gate that runs in ordinary developer and continuous-integration time.
R10 DOCUMENTED SCOPE - the supported and unsupported distributed physics models, floating-point precisions and hardware paths are stated in user documentation.
R11 NO RANK IS A BOTTLENECK - the memory and time any one rank spends preparing, running or writing a distributed simulation are bounded by its own share of the mesh, not by the global mesh or the rank count.
R12 RESTART - a distributed run can be checkpointed and resumed from its written state, and the resumed run continues the interrupted one to solver tolerance.
R13 STORAGE CHANGES PRESERVE RESULTS - a change to how mesh or solver data is stored or indexed leaves residual histories and forces unchanged: bitwise on the CPU at a fixed thread count when only storage changes, and, when the order of a reduction changes, by no more than the same revision differs from itself between two thread counts, on every supported backend.
R14 ELEMENT ACCESS IS STABLE - user code that reads a mesh cell, face or node by index and takes its geometric properties keeps working unchanged across storage changes.

## acceptance

A phase closes when all of the following hold.

1. **Stock-binary run** - the distributed backward-facing-step example completes on stock binaries with no local preferences file and no shell exports, from a clean checkout.
2. **Gate** - the distributed gate passes from the repository's prescribed test command and meets Q2.
3. **Serial regression** - the full serial suite passes with no reduction in test count.
4. **Rank invariance** - the example's residual history at one, two and four ranks meets Q1.
5. **Scaling** - strong-scaling efficiency is recorded over the measured rank range and meets Q3.
6. **Documentation** - user documentation covers distributed setup, launch, supported scope and the CHANGELOG records the feature.

## deliberately NOT requirements

- Distributed LKE and LES turbulence models; distributed Float32 with hypre; multi-GPU scaling validation; AMD GPU parity verification. These are documented gaps, not defects.
- Runtime switching between PETSc builds of different precision inside one Julia session: Julia resolves library preferences per environment at precompilation, so precision selection is an environment choice.
- Validation of the distributed module on Julia 1.10 and 1.11. Excluded by the user for this phase; the compatibility bound still binds anything added to a project file.

## repealed

R4 as first written required that a serial install neither require nor load the distributed-only dependencies. Amended under the same ID to a bounded, measured cost when that cost turned out to be 0.17 s and 27 MB against a six-step refactor of a working module (D18).
