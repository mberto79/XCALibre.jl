* added: `AMG()` is a preconditioner, `SolverSetup(solver=Cg(), preconditioner=AMG(coarsening=...))`, with `Cg()`, `Bicgstab()` or the new `Richardson()`.
* breaking: `AMG()` requires `coarsening=SmoothAggregation()`, `Geometric()` or `RugeStuben()`, and `coarse_refresh_interval` defaults to 5 instead of 1.
* changed: On a distributed mesh `AMG(...)` runs as PETSc's GAMG with the same coarsening, smoother and sweeps; settings GAMG cannot express stop the run.
* changed: `BoomerAMG()` always runs hypre through PETSc; on a serial mesh it errors and points to `distribute(mesh)`.
* changed: With an AMG preconditioner, equations no longer build an unused second preconditioner; `itmax` defaults to 1000 for every solver.
* deprecated: `SolverSetup(solver=AMG(mode=...))` runs as `solver=<mode>, preconditioner=AMG(...)`; `AMGSolver()` is now `Richardson()`.
* deprecated: `GAMG()` returns `AMG(coarsening=SmoothAggregation())`, or `Geometric()` with `agg_nsmooths=0`; other GAMG settings go through `petsc_options`.
