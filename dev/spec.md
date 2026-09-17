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
R4 SERIAL USERS PAY NOTHING - installing and running a serial simulation neither requires nor loads the dependencies used only by distributed runs.
R5 FAMILIAR INTERFACE - a distributed case differs from its serial counterpart only in how the mesh is obtained and how the run is launched; physics, boundary, scheme, solver and runtime setup are written identically.
R6 LAUNCH IS ONE COMMAND - starting a distributed simulation from a shell is a single documented command naming the script, the rank count and the environment.
R7 PARALLEL SPEEDUP - wall-clock time per iteration falls as ranks are added over the supported range, and any departure from ideal is attributed to a measured cause.
R8 RESULTS ARE RANK-INVARIANT - converged fields and residual histories agree across rank counts to solver tolerance.
R9 REGRESSION NET - the distributed feature has an automated gate that runs in ordinary developer and continuous-integration time.
R10 DOCUMENTED SCOPE - the supported and unsupported distributed physics models, floating-point precisions and hardware paths are stated in user documentation.

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

None.
