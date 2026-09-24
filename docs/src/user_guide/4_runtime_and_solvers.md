# Runtime and solvers
*Final steps before launching a simulation*

## Runtime setup
---

At this stage in the setup workflow, the user defines the runtime information to configure the runtime behaviour of the  flow solver, including the time step to use (only meaningful for transient solutions), as well as information about how often to write results to disk. XCALibre.jl provides a the `Runtime` to perform this operation. 

```@docs; canonical=false
Runtime
```

## Configuration object
---

Once all the simulation configuration information has been defined, from discretisation scheme to runtime information, all settings must be wrapped in a `Configuration` object. The definition, including expected input arguments, for the `Configuration` object are detailed below.

```@docs; canonical=false
Configuration
```

## Initialising fields
---

The last (optional) step before running the simulation is to provide an initial guess for all the fields being solved. Although this step is optional, in most cases the flow solvers will perform better when initialised. To set an initial value for a field, the `initialise!` function is provided, which assigns a starting value to a given field.

```@docs; canonical=false
initialise!
```

A uniform initial guess leaves a velocity field that does not satisfy continuity, which the first few solver iterations must work off before making physical progress. For external and internal flows around bodies, a potential-flow field is a much better starting point: it already satisfies continuity and carries the shape of the geometry.

`potential_flow!` solves a velocity-potential equation on the current mesh and projects the velocity field onto the resulting divergence-free field. Boundary conditions for the potential are inferred from those already assigned to pressure, so no extra setup is needed.

```@docs; canonical=false
potential_flow!
```

Call it after `initialise!` and before `run!`:

```julia
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

potential_flow!(model, config; ncorrectors=5)

residuals = run!(model, config)
```

On meshes with appreciable non-orthogonality, pass `ncorrectors` to run non-orthogonal correctors on the potential equation. Set it to zero for orthogonal meshes.

## AMG solver
---

The `AMG` linear solver can be selected directly in `SolverSetup`. It supports `mode=AMGSolver()` for a standalone multigrid solve and `mode=Cg()` for AMG-preconditioned conjugate gradient on symmetric systems such as pressure equations. `mode` takes an instance, not a symbol: `mode=:cg` throws an `ArgumentError`.

```julia
SolverSetup(
    solver = AMG(
        mode = Cg(),
        coarsening = SmoothAggregation(),
        smoother = AMGJacobi()
    ),
    preconditioner = Jacobi(),
    convergence = 1e-7,
    relax = 1.0,
    rtol = 0.0,
    atol = 1e-5
)
```

For non-symmetric systems use `mode=Bicgstab()`, which gives AMG-preconditioned stabilised biconjugate gradient. A typical case is the pressure equation in compressible flow, where the implicit pressure convection is upwinded and therefore non-symmetric. `Cg()` rejects a non-symmetric matrix. `examples/2D_cylinder_transonic_RANS_AMG_BICGStab.jl` uses it for the pressure equation of a transonic RANS case.

Choice of coarsening for non-symmetric operators: `Bicgstab()` has been tested with the default `SmoothAggregation()`. `RugeStuben()` and `Geometric()` are accepted but have not been validated on non-symmetric operators - `RugeStuben()` builds its strength of connection from each row's entries only, not from the transpose - so prefer the default unless you have checked the alternative on your case.

If `Bicgstab()` breaks down - the shadow residual going orthogonal, or the stabilising step collapsing - the half-step iterate is kept and the solve restarts with a fresh shadow residual, at most twice.

`scale_correction` (on by default) is supported. It makes the V-cycle preconditioner depend on its input, but the solver is right-preconditioned, so the solution and residual it carries stay consistent, and convergence is confirmed against the true residual `b - A*x`.

```julia
SolverSetup(
    solver = AMG(
        mode = Bicgstab(),
        smoother = AMGGaussSeidel(sweep = AMGForwardSweep())
        ),
    preconditioner=DILU(),
    convergence = 1e-7,
    relax = 1.0,
    rtol = 0.0,
    atol = 1e-5     
)
```

### Refreshing the hierarchy

The AMG hierarchy is built once, on the first solve. On later solves the matrix coefficients change but its sparsity does not, so the coarsening and the transfer operators are kept and only their values are refreshed. The finest level is refreshed on every solve. The coarse operators are recomputed every `coarse_refresh_interval` solves; in between, the coarse levels from the last refresh are reused. The default is `1`, which refreshes every solve, as OpenFOAM's GAMG does.

Every solve still converges to the requested tolerance, so a longer interval changes only the cost of each solve, not its result: fewer refreshes, possibly more iterations while the coarse levels are out of date. For steady incompressible simulations, where the pressure matrix changes slowly between iterations, an interval of 5 to 10 is safe. On the 354k-cell motorBike case (steady k-omega, `SmoothAggregation()` with `AMGChebyshev()`, 500 iterations on 8 threads), intervals of 5 and 10 each ran in 57.7 s against 66.9 s at the default. Pressure iterations per solve did not grow between refreshes, and the final residuals matched the default to within 0.5%. Longer intervals have not been checked on transient or compressible cases; keep the default there unless you have measured your case.

```julia
AMG(mode = Cg(), coarsening = SmoothAggregation(), smoother = AMGChebyshev(),
    coarse_refresh_interval = 5)
```

### Solving the coarsest level

The coarsest level of the hierarchy is small but is visited on every V-cycle, so on a GPU the
default host round trip can dominate. `coarse_solve` selects how it is solved:

- `OnDevice(; max_rows = 512)` (default) — device-resident factorisation, no host round trip.
  A coarsest level with more rows than `max_rows` falls back to the host, since the factor is
  rebuilt on every coarse refresh.
- `OnHost()` — always copy the coarsest right-hand side to the host and solve there. Also
  written `coarse_solve = CPU()`.
- `OnDeviceJacobi(; omega, iterations)` and `OnDeviceChebyshev(; degree, eig_ratio, lambda_scale)` — apply a
  fixed number of smoother sweeps on the device instead of solving. Both are constant linear
  operators, so they are valid with `mode = Cg()` as well as `mode = AMGSolver()`.
- `OnDeviceKrylov(; solver, rtol, atol, itmax)` — run an inner Krylov solve on the device. An
  inner Krylov solve is nonlinear in its right-hand side, which breaks the fixed-operator
  assumption of preconditioned CG, so this one requires `mode = AMGSolver()` and is rejected with
  `mode = Cg()`. Passing a GPU backend, as in `coarse_solve = CUDABackend()`, selects it.

```julia
AMG(mode = AMGSolver(), coarsening = SmoothAggregation(), smoother = AMGJacobi(),
    coarse_solve = OnDeviceChebyshev(degree = 8))
```

On a CPU backend every `OnDevice*` option behaves as `OnHost()`.

## Launching flow solvers
---

In XCALibre.jl the `run!` function is used to start a simulation, which will dispatch to the appropriate flow solver for execution. Once the simulation is complete a `NamedTuple` containing residual information is returned for users to explore the convergence history of the simulation. 

```@docs; canonical=false
run!()
```

`run!` shows a progress bar with the current iteration and residuals. Pass `progress=false` to turn it off:

```julia
residuals = run!(model, config; progress=false)
```

We recommend `progress=false` for large-scale runs that are not on a local PC, such as batch jobs on a cluster, where the progress bar only fills the job's log. The residual history returned by `run!` is the same either way.

## Restarting simulations
---

It should be noted that when running a simulation with `run!`, the solution fields in the `Physics` model are mutated. Thus, running the simulation from the previous solution is simply a matter of reissuing the `run!` function. At present, this has the side effect of overwriting any existing solution files (`.vtk` or `.vtu`). Users must be aware of this behaviour.

In some cases, it may be desirable to solve a problem with a steady solver and use the solution to run transient simulations. This is possible using the `change` function.

```@docs; canonical=false
XCALibre.ModelPhysics.change
```
