* breaking: `AMG(coarsening; smoother, sweeps, coarsest, refresh, native=NativeAMG(coarse, scale_correction, fuse))` replaces `pre_sweeps`, `post_sweeps`, `max_coarse_rows`, `coarse_refresh_interval`, `coarse_solve` and `fuse_levels`. Old keywords still work, with a warning.
* breaking: AMG types renamed: `Pairwise`, `SmoothedAggregation`, `Jacobi(weight)`, `Chebyshev`, `GaussSeidel`, `SymGaussSeidel`, `SOR`, `SSOR`, `Direct`, `Smoothed`, `Iterative`. Old names still work, with a warning.
* breaking: `AMG()` defaults to `Pairwise()` coarsening and `refresh=5`.
* breaking: `AMG(coarse_storage=Float32)` is now `SolverSetup(precision=MixedF32())`.
* breaking: `BoomerAMG()` takes typed keywords (`threshold`, `coarsen`, `interpolation`, `max_interp`, `aggressive`, `smoother`, `sweeps`, `refresh`); `freeze` is `refresh`.
* changed: On a distributed mesh `AMG(...)` runs as PETSc's GAMG with its settings translated; settings GAMG cannot express stop the run.
* added: `AMG(...)` with `Bicgstab()` runs on distributed meshes for non-symmetric systems.
* added: `Sweeps(pre, post; grow, max)` for unequal sweeps and more sweeps on coarser levels.
* changed: `BoomerAMG()` always runs hypre through PETSc; on a serial mesh it errors and points to `distribute(mesh)`.
* deprecated: `GAMG(...)` returns the `AMG(...)` of PETSc's GAMG defaults.
* fixed: `AMG()` as the preconditioner of a vector equation (momentum) no longer fails on a serial mesh.
