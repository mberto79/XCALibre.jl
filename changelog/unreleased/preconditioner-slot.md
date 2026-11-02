* added: `AMG()` is a preconditioner, `SolverSetup(solver=Cg(), preconditioner=AMG())`, with `Cg()`, `Bicgstab()` or the new `Richardson()`. `AMG()` maps to `GAMG()` on distributed meshes, and `GAMG()`/`BoomerAMG()` to the native AMG on serial ones. [#244](@ref)
* changed: With an AMG preconditioner, equations no longer build an unused second preconditioner; `itmax` defaults to 1000 for every solver. [#244](@ref)
* deprecated: `SolverSetup(solver=AMG(mode=...))` runs as `solver=<mode>, preconditioner=AMG(...)`; `AMGSolver()` is now `Richardson()`. [#244](@ref)
