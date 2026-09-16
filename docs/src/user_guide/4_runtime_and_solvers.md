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

## Launching flow solvers
---

In XCALibre.jl the `run!` function is used to start a simulation, which will dispatch to the appropriate flow solver for execution. Once the simulation is complete a `NamedTuple` containing residual information is returned for users to explore the convergence history of the simulation. 

```@docs; canonical=false
run!()
```

## Restarting simulations
---

It should be noted that when running a simulation with `run!`, the solution fields in the `Physics` model are mutated. Thus, running the simulation from the previous solution is simply a matter of reissuing the `run!` function. At present, this has the side effect of overwriting any existing solution files (`.vtk` or `.vtu`). Users must be aware of this behaviour.

In some cases, it may be desirable to solve a problem with a steady solver and use the solution to run transient simulations. This is possible using the `change` function.

```@docs; canonical=false
XCALibre.ModelPhysics.change
```
