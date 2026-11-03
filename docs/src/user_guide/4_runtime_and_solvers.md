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

`potential_flow!` solves a velocity-potential equation on the current mesh and sets the velocity to the resulting divergence-free field. By default it starts from rest: the internal velocity is set to zero first, so the flow is driven by the velocity boundary conditions alone (pass `from_rest=false` to project the current field instead). Boundary conditions for the potential are inferred from those already assigned to pressure, so no extra setup is needed.

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

## AMG preconditioner
---

Algebraic multigrid is a preconditioner: `solver` picks the outer iteration and `preconditioner = AMG(...)` the multigrid. Use `solver = Cg()` for symmetric systems such as incompressible pressure, `solver = Bicgstab()` for non-symmetric ones (compressible pressure, momentum, scalars) and `solver = Richardson()` for plain V-cycles with no Krylov acceleration.

```julia
SolverSetup(
    solver = Cg(),
    preconditioner = AMG(),
    convergence = 1e-7,
    relax = 1.0,
    rtol = 0.0,
    atol = 1e-5
)
```

`AMG(coarsening; smoother, sweeps, coarsest, refresh, native)`:

- `coarsening`: how cells are grouped into coarse levels.
  - `Pairwise(merges=1)` (default): each cell is paired with the neighbour it is most strongly coupled to, `merges` times per level, as OpenFOAM's `algebraicPair` agglomeration. Piecewise-constant transfer between levels. Cheap to build, light on memory and robust for non-symmetric matrices.
  - `SmoothedAggregation(; threshold=0.0)`: larger aggregates with a smoothed interpolation; fewer iterations on hard symmetric problems at a higher setup and memory cost.
  - `RugeStuben(; threshold=0.25)`: classical AMG, serial meshes only (on a distributed mesh use `BoomerAMG()`).
- `smoother`: `Jacobi()` (default), `Chebyshev(; degree=3)`, and on the CPU `GaussSeidel()`, `SymGaussSeidel()`, `SOR(ω)`, `SSOR(ω)`. `Jacobi(weight=4/3)` relaxes each level by `weight/λmax`, with λmax the largest eigenvalue of D⁻¹A there; `SOR` and `SSOR` take the relaxation factor `ω` directly.
- `sweeps`: smoother sweeps before and after the coarse correction on each level. `sweeps = 2` means two before and two after, `sweeps = (0, 2)` none before and two after, and `Sweeps(pre, post; grow, max)` adds `grow` sweeps per level down to at most `max`, as OpenFOAM's GAMG. Defaults to 2 for `Jacobi()` and 1 otherwise.
- `coarsest = 4096`: coarsening stops once a level has at most this many rows.
- `refresh = 5`: the coarse levels are recomputed from the current matrix every `refresh` solves. The aggregates and interpolation are kept while the sparsity pattern is unchanged; natively the finest level always uses the current matrix. Use `refresh = 1` to recompute every solve.
- `native = NativeAMG(...)`: settings of XCALibre's own AMG, which a distributed mesh does not use (below).

`SolverSetup(precision = MixedF32())` keeps the outer iteration and finest level in the equation's precision and stores the coarse levels in Float32, about halving the hierarchy's memory.

```julia
SolverSetup(
    solver = Bicgstab(),
    preconditioner = AMG(SmoothedAggregation(); smoother = SymGaussSeidel(), sweeps = (0, 2)),
    convergence = 1e-7,
    relax = 1.0,
    rtol = 0.0,
    atol = 1e-5
)
```

`Cg()` rejects a non-symmetric matrix. A typical non-symmetric case is the pressure equation in compressible flow, where the implicit pressure convection is upwinded; `examples/2D_cylinder_transonic_RANS_AMG_BICGStab.jl` solves it with `Bicgstab()` and AMG. If `Bicgstab()` breaks down (the shadow residual going orthogonal, or the stabilising step collapsing) the half-step iterate is kept and the solve restarts with a fresh shadow residual, at most twice. `Cg()` switches to its flexible form when the V-cycle is not symmetric (`scale_correction`, unequal `pre` and `post`, or a forward-only smoother), which tolerates the non-symmetric preconditioner; a symmetric V-cycle remains the better partner for `Cg()`.

!!! note "Earlier spellings"
    Earlier keywords and types still work for now, each with a one-time deprecation warning naming the current spelling: `AMG(coarsening=..., pre_sweeps=..., coarse_refresh_interval=..., max_coarse_rows=..., coarse_solve=..., scale_correction=..., fuse_levels=...)`, `Geometric()`, `SmoothAggregation()`, `AMGJacobi()`, `AMGChebyshev()`, `AMGGaussSeidel()`, `AMGSOR()` and the `OnDevice*`/`OnHost` coarse solves. `SolverSetup(solver = AMG(mode = Cg()))` becomes `solver = Cg(), preconditioner = AMG(...)`, `AMGSolver()` is `Richardson()`, and `GAMG()` returns the `AMG()` of PETSc's GAMG defaults. `AMG(coarse_storage = Float32)` is now `precision = MixedF32()`.

### Native settings

`NativeAMG(; coarse=Direct(), scale_correction=true, fuse=0)`:

- `coarse`: the coarsest-level solve, visited on every V-cycle.
  - `Direct(; max_rows=512)` (default): exact solve. On a GPU the factor stays on the device while the coarsest level has at most `max_rows` rows, otherwise it is solved on the host; `max_rows = 0` always uses the host.
  - `Smoothed(smoother=Chebyshev(degree=10); sweeps=1)`: fixed sweeps of `Jacobi()` or `Chebyshev()` on the device, a fixed linear operator valid with `Cg()`. Suits a large `coarsest` on a GPU.
  - `Iterative(solver=Bicgstab(); rtol=1e-2, itmax=50)`: a Jacobi-preconditioned Krylov solve on the device. It depends on the right-hand side, so it needs `solver = Richardson()`.
  - On a CPU backend every choice is a `Direct()` solve.
- `scale_correction`: scales each coarse correction by `(r·c)/(c·Ac)`, the factor that minimises the error in the energy norm.
- `fuse`: on a GPU with `Pairwise()`, `Jacobi()` and `Direct()`, `fuse = 1` keeps the transfers between levels as aggregate lists instead of sparse matrices (less memory, no extra work), and `fuse = n > 1` also applies the top `n - 1` coarse levels through the finer matrix instead of storing them, at the cost of extra matrix products per V-cycle; worthwhile when the hierarchy does not fit on the device. A CPU ignores it.

```julia
SolverSetup(solver = Richardson(), preconditioner = AMG(coarsest = 8000,
    native = NativeAMG(coarse = Smoothed(Chebyshev(degree = 8)))), convergence = 1e-7, relax = 1.0)
```

### AMG on a distributed mesh

On a distributed mesh `AMG(...)` runs as PETSc's GAMG with its settings translated:

- `Pairwise()` → `-pc_gamg_agg_nsmooths 0` (unsmoothed aggregation); `SmoothedAggregation(; threshold)` → `-pc_gamg_agg_nsmooths 1 -pc_gamg_threshold <threshold>`. Aggressive coarsening is off (`-pc_gamg_aggressive_coarsening 0`) to mirror the native levels.
- `Jacobi(; weight)` → Richardson with Jacobi and scale `weight/2`; `Chebyshev(; degree, eig_ratio, lambda_scale)` → PETSc's Chebyshev with Jacobi, degree as `-mg_levels_ksp_max_it`, bounds as `-mg_levels_ksp_chebyshev_esteig`; `GaussSeidel()`, `SymGaussSeidel()` → PETSc's SOR within each rank.
- `sweeps` → `-mg_levels_ksp_max_it`, with a separate up smoother (`-pc_mg_distinct_smoothup`) when `pre ≠ post`.
- `coarsest` → `-pc_gamg_coarse_eq_limit`; `refresh = N` → the preconditioner is held for `N` solves and its coarse operators recomputed on the next, keeping the interpolation (`-pc_gamg_reuse_interpolation`).
- With any outer iteration other than `Cg()` the aggregation graph is symmetrised (`-pc_gamg_graph_symmetrize`), so non-symmetric matrices are supported.

What differs from the native AMG:

- GAMG builds its aggregates with its own (MIS) algorithm, not pairwise, so hierarchies and iteration counts are close, not identical.
- `Jacobi()` damps each level by `weight/λmax` natively; PETSc uses one scale, `weight/2`, exact for `λmax ≈ 2`.
- Between refreshes PETSc applies the whole stored preconditioner, while the native AMG updates its finest level every solve.
- `NativeAMG` settings are not used; when they differ from the defaults one info message lists them.

Settings GAMG cannot express stop the run: `RugeStuben()` (use `BoomerAMG()`), `Pairwise(merges > 1)`, `Sweeps(grow > 0)`, `Chebyshev()` with more than one sweep, `SOR`/`SSOR` with `ω ≠ 1`, and `SmoothedAggregation(weight, near_nullspace)`.

The rules for every preconditioner are listed in the preconditioners section of the numerical setup page.

## Launching flow solvers
---

In XCALibre.jl the `run!` function is used to start a simulation, which will dispatch to the appropriate flow solver for execution. Once the simulation is complete a `NamedTuple` containing residual information is returned for users to explore the convergence history of the simulation. 

The residual of each field is the normalised residual `Σᵢ|bᵢ - (Ax)ᵢ| / Σᵢ|Aᵢᵢxᵢ|` of its assembled system, evaluated on the current field `x` just before the linear solve, so it measures how far the previous iteration's fields are from satisfying this iteration's equations. It does not depend on the linear solver, its preconditioner, precision or tolerances, nor on whether the run is serial, distributed or on a GPU. When the field is zero everywhere (a common initial pressure), the denominator is zero and the unnormalised `Σᵢ|bᵢ - (Ax)ᵢ|` is reported for that iteration.

```@docs; canonical=false
run!()
```

`run!` shows a progress bar with the current iteration and residuals. Pass `progress=false` to turn it off:

```julia
residuals = run!(model, config; progress=false)
```

We recommend `progress=false` for large-scale runs that are not on a local PC, such as batch jobs on a cluster, where the progress bar only fills the job's log. The residual history returned by `run!` is the same either way.

The momentum equation of every flow solver carries the full viscous stress, ∇·τ with τ = μ_eff (∇U + (∇U)ᵀ - ⅔ (∇·U) I). The part ∇·(μ_eff ∇U) is discretised implicitly as a Laplacian, and the rest, ∇·(μ_eff dev2((∇U)ᵀ)) with dev2(A) = A - ⅔ tr(A) I, is added as an explicit source evaluated from the current velocity gradient. For a constant viscosity in incompressible flow this source is zero, since ∇·(∇U)ᵀ = ∇(∇·U); it matters wherever the effective viscosity varies, as it does with any turbulence model. At boundary faces the source uses the owner-cell gradient with its normal derivative taken from the velocity condition: (U_b − U_c)/δ for fixed values (`Wall`, `Dirichlet`, `RotatingWall`, and `DirichletFunction` evaluated at the face and the current time), zero for `Zerogradient`, `Extrapolated`, `Outlet` and `NeumannFunction`, and the mirrored value for `Slip` and `Symmetry`; periodic faces are treated as internal faces between the two cells of a pair. Pass `transpose_stress=false` to leave it out, e.g. to compare with results obtained before it was included:

```julia
residuals = run!(model, config; transpose_stress=false)
```

The steady solvers (`simple!` for `Incompressible` fluids and `csimple!` for `Compressible` and `WeaklyCompressible` fluids, used by `run!` for `Steady()` models) offer a consistent (SIMPLEC) pressure–velocity coupling. By default, the pressure equation and the velocity correction use rD = V/a_P, the inverse diagonal coefficient of the under-relaxed momentum matrix. Because a_P grows as the velocity relaxation factor α_U decreases, rD scales roughly with α_U, so the pressure correction is over-damped and has to be compensated by pressure under-relaxation. With `consistent=true` the pressure Laplacian and the velocity and face-flux corrections use rDt = V/(a_P + Σ a_N), the inverse row sum of the momentum matrix, which does not shrink with α_U; the extra terms (rDt - rD)∇p are added to the predicted velocity and face fluxes, so the converged solution is the same. Incompressible runs then need little or no pressure under-relaxation. In compressible flow the weak Laplacian of the default coupling also loses weight against the compressibility term ∇·(ψ ((H/a_P)·S) p) of a `Compressible` fluid; in transonic flow this can amplify round-off from one iteration to the next until the run diverges, even for a uniform free stream on some meshes. The consistent coupling keeps the Laplacian's weight and is worth trying when a steady transonic run diverges. The option works on distributed meshes with `simple!`:

```julia
residuals = run!(model, config; consistent=true)
```

It is off by default, which leaves earlier results unchanged. The transient solvers do not take this option.

## Restarting simulations
---

It should be noted that when running a simulation with `run!`, the solution fields in the `Physics` model are mutated. Thus, running the simulation from the previous solution is simply a matter of reissuing the `run!` function. The continued run also picks up the face mass flux of the last iteration, so a run split into several `run!` calls (for example, to monitor forces every few hundred iterations) retraces the uninterrupted run. Re-initialising the fields with `initialise!` between calls starts afresh. At present, this has the side effect of overwriting any existing solution files (`.vtk` or `.vtu`). Users must be aware of this behaviour.

In some cases, it may be desirable to solve a problem with a steady solver and use the solution to run transient simulations. This is possible using the `change` function.

```@docs; canonical=false
XCALibre.ModelPhysics.change
```
