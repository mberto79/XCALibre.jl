# P1-M7 - preconditioner API, informed by the scaling measurements (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R5, R10. Governing decisions: D28, D29, D31, D32, D33, D34, D35, D36, D37.

## Problem, quantified

Three defects, each measured on the 499,503-cell backward-facing step at a pinned 2200 MHz; numbers in `dev/telemetry/scaling.csv`.

1. `reuse` NAMES THE WRONG THING. `BoomerAMG(reuse=N)` calls `KSPSetReusePreconditioner`, which skips `PCSetUp` entirely, so N-1 solves apply a hierarchy built from an older matrix (D28). A user reads "reuse" as a coefficient update; it is a freeze. `GAMG` carries both `reuse` and `reuse_interpolation`, which mean different things, and nothing in the docstrings separates them.
2. THE DEFAULTS ARE NOT THE MEASURED OPTIMA. `BoomerAMG`'s `reuse=10` gives 0.4216 s/iter at four ranks against 0.3951 at `reuse=25` (D28). `GAMG`'s `reuse=1` gives 0.4116 against 0.3023 at `reuse=25` (D35). The GAMG default costs 36%.
3. THE DOCUMENTATION RECOMMENDS THE WRONG AMG. `BoomerAMG`'s docstring calls it "the recommended pressure preconditioner for distributed runs". Measured, `GAMG(reuse=25)` is faster at every rank count, has the flattest scaling curve of any configuration, and holds rank invariance to 0.50% against BoomerAMG's 240% (D31, D36).

## Approach

Rename for the mechanism, retune the defaults to the measurements, and move the recommendation. No new mechanism is built: PETSc already exposes both the freeze and the coefficient-only update, and the measurements say which to prefer. The one structural finding stays in `architecture.md`: OpenFOAM's agglomeration multigrid sums fine coefficients into agglomerated cells, where PETSc GAMG forms Galerkin coarse operators by sparse triple product, which is why our setup dominated before the freeze (D34).

`Cg()+Jacobi()` stays the documented default for cases of this size and conditioning; it is 11% faster than frozen GAMG at eight ranks (D37). This milestone changes what is recommended WHEN AMG is warranted, not the default itself.

## Configuration space

Preconditioner x reuse x rank count: {Jacobi, BoomerAMG, GAMG} x {1, 5, 10, 25, 50} x {1, 2, 4, 6, 8}, on {499,503, 1,320,368} cells. The covered subset is in `dev/telemetry/scaling.csv`; `dev/scripts/scaling_probe.jl` takes `pc=` and `reuse=` and generates any row. The gate is that table at a PINNED clock - an unpinned row is not evidence (D20).

## Steps

Steps are `P1-M7-S<j>`, allocated in order and never renumbered.

- [ ] **P1-M7-S1** Rename `reuse` to a name that says it freezes, keeping the old keyword accepted with a deprecation for one release - mechanism: the field sets `KSPSetReusePreconditioner`, so the name states the PETSc call's semantics - cost: none at runtime - verdict: the docstring's description of what happens between solves matches what `_maybe_reuse_pc!` does.
- [ ] **P1-M7-S2** Retune the defaults to the measured optima and state the residual-quality cost in the docstring - mechanism: the optimum is where rebuild cost and staleness cross, which the sweep locates - cost: one probe row per candidate - verdict: no configuration in the catalogue is slower than at the old default.
- [ ] **P1-M7-S3** Move the AMG recommendation to `GAMG` and say why in one clause - mechanism: GAMG's coefficient-only update is valid because a non-refining mesh fixes the sparsity pattern - cost: documentation only - verdict: the recommendation cites the telemetry rather than asserting.
- [ ] **P1-M7-S4** Document that AMG breaks rank invariance and Jacobi does not - mechanism: an AMG hierarchy is built from the local partition, which changes with rank count - cost: documentation only - verdict: a user comparing runs at two rank counts finds the caveat before filing it as a bug.

## Exit criterion

A measured recommendation for the distributed pressure preconditioner, the naming settled, and the docstrings matching the semantics. The distributed documentation page names the AMG to use and the rank-invariance caveat.

## Open questions

- Is BoomerAMG worth keeping at all, given GAMG wins on cost, scaling and reproducibility? Settled by a stiffer case where AMG setup amortises - this phase has no such case (D26), so the question carries to whoever adds one.
- Does the freeze interval want to be adaptive rather than a constant? Settled by whether the optimum moves with mesh size; only 499,503 cells has been swept.
