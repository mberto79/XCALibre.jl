# Numerical setup
*Available discretisation schemes, linear solvers and preconditioners*

## Discretisation schemes
---

Part of the methodology to solve the various model equations using the Finite Volume Method (FVM) is to discretise each equation term, essentially, this process linearises the partial differential equation so that it can be represented as a system of linear equations, which can be solved using linear algebra along with iterative solvers. This section presents the discretisation schemes currently available in XCALibre.jl.

Discretisation schemes in XCALibre.jl are organised under the abstract type `AbstractScheme`. As shown previously, a list of available schemes can be found using the  `subtypes` function:

```@repl
using XCALibre # hide
using AbstractTrees # hide
# import Main.subtypes as subtypes # hide
    using InteractiveUtils # hide
AbstractTrees.children(d::DataType) = Main.subtypes(d) # hide
print_tree(AbstractScheme) # hide
```

### Schemes available 
---
#### Time schemes

| **Scheme** | **Description** |
|:-------|:------------|
| SteadyState | sets the time derivative to zero |
| Euler | First order implicit Euler scheme |
| CrankNicolson | Second order central scheme (not implemented yet) |
---
#### Laplacian schemes

| **Scheme** | **Description** |
|:-------|:------------|
| Linear | 2nd order Gauss gradient scheme with linear interpolation |
---
#### Divergence schemes

| **Scheme** | **Description** |
|:-------|:------------|
| Linear | 2nd order central difference |
| Upwind | 1st order upwind scheme |
| BoundedUpwind | Bounded version of the Upwind scheme |
| LUST | 1st/2nd order mixed scheme (fixed at 75% Linear - 25% Upwind) |
---
#### Gradient schemes

| **Scheme** | **Description** |
|:-------|:------------|
| Orthogonal | Green-Gauss uncorrected gradient scheme |
| Midpoint | Green-Gauss skew corrected scheme (2 iterations - hardcoded) |

---
### Specifying schemes

XCALibre.jl flow solvers offer considerable flexibility to users for defining discretisation schemes. However, this means that discretisation schemes must be specified for every term of every equation solved. The schemes must be provided as a `NamedTuple` where each keyword corresponds to the fields being solved, e.g. (U = ..., p = ..., k = ..., <field> = ...). To facilitate this process, the [`set_schemes`](@ref) function is provided. Used without any inputs `set_schemes` uses the default values provided (see details below).

```@docs; canonical=false
set_schemes
```

For example, below we set the schemes for the  `U` and `p` fields. Notice that in the first case the schemes will take their default values (entry for `p`). In the case of `U`, we are only changing the setting for the divergence scheme to `Upwind`.

```jldoctest;  filter = r".*"s => s"", output = false
using XCALibre
schemes = (
    p = set_schemes(), # no input provided (will use defaults)
    U = set_schemes(divergence = Upwind),
)

# output

```

## Linear solvers
---

Linear solvers in XCALibre.jl are provided by [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl). The following solvers types are re-exported in XCALibre.jl

* `BicgstabSolver` is a general purpose linear solver. Works well with non-symmetric matrices e.g. for `U`.
* `CgSolver` is particular strong with symmetric matrices e.g to solve the pressure equation.
* `GmresSolver` is a general solver. We have found it works best on the CPU backend.

For more information on these solvers you can review the excellent documentation provided by the [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) team. 

XCALibre.jl provides the `set_solver` convenience function for setting solvers. See details below. 

```@docs; canonical=false
set_solver
```

### Preconditioners 

XCALibre.jl offers a range of preconditioners which are subtypes of the abstrac type `PreconditionerType`, exploring its subtypes we can find a list of the currently available preconditioners: 

```@repl
using XCALibre # hide
using AbstractTrees # hide
# import Main.subtypes as subtypes # hide
    using InteractiveUtils # hide
AbstractTrees.children(d::DataType) = Main.subtypes(d) # hide
print_tree(PreconditionerType) # hide
```

!!! note

    Only the `Jacobi` and `NormDiagonal` preconditioners have GPU ready implementations. At present these have the most robust implementation and they can be used with both CPU and GPU backends. The other preconditioners can only be used on the CPU. Notice that on our tests the `LDL` preconditioner only works when paired with the `GmresSolver` on the CPU. Also notice that the implementation of the `DILU` preconditioner, although functions, is only experimental. Work on improving the offering of preconditioners is ongoing.

!!! warning

    Internally the storage for sparse matrices was moved to the CSR format. Thus, temporarily, we have removed support of LDL, DILU and ILU0 preconditioners while we work on CSR compatible implementations. 



Below an example is provided in context. Here, we are setting solvers for both the velocity field `U` and the pressure field `p` and packing them into a `NamedTuple` "solvers". The `Jacobi` preconditioner is used in both solvers. Notice that preconditioners are specified with an instance of their type i.e. `Jacobi()`. Internally, the preconditioner instance is used for dispatch. This tupple will then be passed on to create the final `Configuration` object.

```@meta
DocTestSetup = quote
    using XCALibre
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    grid = "backwardFacingStep_10mm.unv"
    mesh_file = joinpath(grids_dir, grid)
    mesh = UNV2D_mesh(mesh_file, scale=0.001)
    mesh_dev = mesh # use this line to run on CPU
    nu = 1e-3
    model = Physics(
        time = Steady(),
        fluid = Fluid{Incompressible}(nu = nu),
        turbulence = RANS{Laminar}(),
        energy = Energy{Isothermal}(),
        domain = mesh_dev
    )
end
```

```jldoctest;  filter = r".*"s => s"", output = false
using XCALibre

# Note: this example assumes a Physics object named `model` already exists

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    )
)

# output

```

```@meta
DocTestSetup = nothing
```