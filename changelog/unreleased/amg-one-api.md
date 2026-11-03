* breaking: `AMG()` requires `coarsening=SmoothAggregation()`, `Geometric()` or `RugeStuben()`, and `coarse_refresh_interval` defaults to 5 instead of 1.
* changed: On a distributed mesh `AMG(...)` runs as PETSc's GAMG with the same coarsening, smoother and sweeps; settings GAMG cannot express stop the run.
* changed: `BoomerAMG()` always runs hypre through PETSc; on a serial mesh it errors and points to `distribute(mesh)` instead of running the native AMG.
* deprecated: `GAMG()` returns `AMG(coarsening=SmoothAggregation())`, or `Geometric()` with `agg_nsmooths=0`; other GAMG settings go through `petsc_options`.
