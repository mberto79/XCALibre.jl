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
| Euler | First order implicit Euler scheme (implicit)|
| CrankNicolson | Second order trapezoidal scheme (implicit); `CrankNicolson(ψ)` with an off-centring coefficient `ψ` |

`CrankNicolson` evaluates every term of an equation at the new time level ``t^{n+1}`` and takes the time derivative as

```math
\left(\frac{\partial \rho\phi}{\partial t}\right)^{n+1} = (1 + \psi)\,\frac{(\rho\phi)^{n+1} - (\rho\phi)^n}{\Delta t^n} - \psi \left(\frac{\partial \rho\phi}{\partial t}\right)^n,
```

where the time derivative of the previous step is kept by the equation's `Time` term. With ``\psi = 1`` (`CrankNicolson` or `CrankNicolson(1)`) this is the second-order trapezoidal rule, also with time steps that change from step to step; ``\psi = 0`` is backward Euler, and values in between are first order but damp the oscillations that the trapezoidal rule leaves undamped (stiff modes such as diffusion across thin wall cells, or a non-solenoidal initial field). Boundary values, the pressure and the pressure correctors are all at ``t^{n+1}``.

- The first time step of a run, including a run resumed with `restart` or continued with a second `run!`, is a backward-Euler step: the time derivative of the step before is not stored in the results.
- Terms treated explicitly (the convecting face flux, the eddy viscosity and velocity gradient of the LES and transpose-stress terms, the lagged coefficients of the `KEquation`, `KOmega`, `KOmegaSST` and `KOmegaLKE` equations) are extrapolated to ``t^{n+1}`` from the two previous steps, using the time-step history, and are evaluated from the initial fields on the first step, so the eddy viscosity need not be initialised. This extrapolation is the only difference between `CrankNicolson(0)` and `Euler`.
- The pressure of a `CrankNicolson(1)` run can alternate from step to step (an odd-even mode in time that the trapezoidal rule leaves undamped, started for example by an impulsive start); the velocity is unaffected, but the pressure solves take more iterations. `CrankNicolson(0.9)` damps it. In the face flux, half the step change of the old pressure gradient held in the time derivative takes the compact face gradient, so that this mode cannot grow on non-orthogonal meshes.
- In `cpiso!` the density of the time terms must match the state equation at ``t^{n+1}``: second order needs `outer_loops` passes over momentum, energy and pressure per step (keyword of `run!` and `cpiso!`, default 1). About 8 passes reproduce the order in an acoustic test; each pass costs about one `Euler` time step.
- `relax` below 1 changes the equations solved in each step and lowers the order for either scheme.
- With `AdaptiveTimeStepping`, let the time step change gradually (`maxGrow` and `minShrink` close to 1).
- Use one time scheme for every field: combining `CrankNicolson` and `Euler` raises an `ArgumentError`. `CrankNicolson` also raises an `ArgumentError` in `multiphase!` (explicit first-order volume-fraction transport), `filmModel!` (explicit first-order film coupling), `godunov!` (turbulence transport split from the explicit flow update) and the steady solvers.
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
| LUSTDC | `LUST` in deferred-correction form: upwind matrix coefficients with the 75% (Linear - Upwind) part of the face value added explicitly |
| LUSTLU | `LUST` coefficients in the matrix plus an explicit gradient correction of its upwind quarter (as `LinearUpwind`); for transient and LES runs, needs a gradient of the transported field |
| LinearUpwind | 2nd order upwind scheme: upwind matrix coefficients with an explicit (deferred) correction of the face value from the upwind-cell gradient |
| LimitedLinearUpwind | Bounded version of `LinearUpwind`: the correction is limited per face with the downwind value (TVD bound) |

`Upwind`, `BoundedUpwind`, `LinearUpwind`, `LimitedLinearUpwind` and `LUSTDC` keep the matrix diagonally dominant. `Linear` and `LUST` put central-difference coefficients in the matrix, which loses diagonal dominance where convection dominates and allows oscillations across discontinuities.

`LinearUpwind` reconstructs the face value of the upwind cell `U` as ``\phi_f = \phi_U + (\nabla\phi)_U \cdot (\mathbf{x}_f - \mathbf{x}_U)``; the reconstruction is exact for linear fields on any mesh, and the gradient limiter set for the field (`Schemes(limiter=...)`) is applied to the gradient used. A gradient limiter reduces over- and undershoots but does not remove them from a steady solution; `LimitedLinearUpwind` also bounds the face value between the upwind and downwind cell values and falls back to upwind at local extrema. Both schemes use upwind values on boundary faces.

The solvers supply the gradient for the momentum equation, for `he`, and for `k` and `omega` of `KOmega` and `KOmegaSST` (for `he` the kinetic energy K = |U|²/2 of the explicit flux ∇·(ṁK) is reconstructed at the faces in the same way, and h and K are limited as one total enthalpy: the `he` limiter acts on ∇(h + K) and ∇K, h takes their difference, and the `LimitedLinearUpwind` clip of h also scales the K correction, so that a uniform h + K stays a discrete solution); for other equations both schemes reduce to `Upwind` with a warning.

The gradient of a ghost cell is taken from its owner on distributed meshes. In a user-built equation the gradient field is passed as a third argument, e.g. `Divergence{LinearUpwind}(mdotf, phi, gradphi.result)`.

`LUSTDC` is `LUST` with the same face values but an upwind matrix: the 75% (`Linear` - `Upwind`) part of the face value is added explicitly from the current field, and boundary faces use the `LUST` boundary coefficients (periodic patches keep its implicit coupling). It needs no gradient. A converged steady solution therefore matches `LUST` up to the momentum diagonal used in the face-flux (Rhie-Chow) interpolation, a difference of the same size as changing the velocity relaxation factor. Use it for steady runs (`simple!`, `csimple!`) that diverge with `LUST` where convection dominates. In transient runs the explicit part lags one time step: results match `LUST` as the time step goes to zero, but the scheme becomes unstable at smaller Courant numbers than `LUST`, so `LUST`, which is unchanged, remains the recommended scheme for transient and LES runs. `LUSTDC` is not OpenFOAM's `LUST`: that scheme keeps the 75% linear part implicit and corrects the remaining upwind quarter explicitly with the upwind-cell gradient, as `LinearUpwind` does, whereas both `LUST` and `LUSTDC` use the plain upwind value for that quarter.

`LUSTLU` is the opt-in second-order variant of `LUST` for transient and LES runs. The matrix has exactly the `LUST` coefficients (75% linear, 25% upwind, so the same implicit coupling and time-step limit), and the upwind quarter of the face value is corrected explicitly with the gradient of the upwind cell, `φ_f = ¾ φ_lin + ¼ [φ_U + (∇φ)_U · (x_f − x_U)]`, which removes the first-order dissipation `LUST` retains from that quarter (the gradient is limited with `Schemes(limiter=...)`, as for `LinearUpwind`). On a Taylor-Green vortex at Re 1600 with WALE it raises the kinetic energy at t = 10 by about 37% relative to `LUST`, which was 22% below the OpenFOAM `LUST` reference; the stability limit of `piso!` on a Re 100 cylinder is that of `LUST`. `LUST` itself is unchanged, boundary faces use the `LUST` coefficients, and without a gradient `LUSTLU` reduces to `LUST`. It is available for `U` and the other transported fields of the incompressible and compressible solvers, but not for `he`.

---
#### Gradient schemes

| **Scheme** | **Description** |
|:-------|:------------|
| Gauss | Green-Gauss uncorrected gradient scheme |
| Midpoint | Green-Gauss skew corrected scheme (2 iterations - hardcoded) |
| LeastSquares | Least-squares gradient over the face neighbours, weighted by the inverse squared distance (boundary faces use the boundary value); exact for linear fields on any mesh |

#### Gradient limiters

| **Scheme** | **Description** |
|:-------|:------------|
| CellBased | Barth-Jespersen: one factor per cell keeps the face-extrapolated values within the neighbour minimum and maximum; `CellBased(level)` blends it with the unlimited gradient |
| Venkatakrishnan | Same bounds as `CellBased` with the smooth limiter function of Venkatakrishnan; `Venkatakrishnan(K)` (default `K = 5`) leaves differences below ε² = (K h)³ unlimited, h being the cell length scale |
| FaceBased | Limit gradient calculation based on face extrapolation |
| MFaceBased | Limit gradient calculation based on face extrapolation in multiple dimensions |

`Venkatakrishnan` changes the limiting factor continuously with the field instead of switching it at the bound; with `K = 0` it limits more than `CellBased` and also limits a linear field on distorted cells. Because ε² is compared with squared differences of the field, the effect of `K` depends on the units of the field and the cell size. `CellBased` and `Venkatakrishnan` are constructed without the mesh, e.g. `Schemes(divergence=LinearUpwind, limiter=Venkatakrishnan(5.0))`.

The construction of all gradient limiters currently requires users to pass the mesh object that will be used for the simulations e.g. `MFaceBased(mesh)`. Notice that the mesh object must be moved to the same device where the simulation will be performed. To prevent mistakes and avoid errors, it is recommended to construct the limiters using the simulation `Physics` object directly since it provide a reference to the mesh (domain) e.g. if a `Physics` object named `model` has been created, construct the limiter as `MFaceBased(model.domain)`

---
#### Density-based (`SupersonicFlow`) solver

The explicit density-based solver (`godunov!`, selected with `Fluid{SupersonicFlow}`) advances the conservative variables ``[\rho, \rho\mathbf{U}, \rho E]`` and does not use the divergence, Laplacian or time schemes above. Its inviscid fluxes are set with three entries of the `schemes` tuple, while the `gradient` of `U`, `p` and `T` is used for the viscous fluxes and for the reconstruction.

| **Entry** | **Options** | **Description** |
|:-------|:------------|:------------|
| `flux` | `Rusanov()` | Local Lax-Friedrichs flux, most dissipative and robust (default) |
| | `HLLC()` | Harten-Lax-van Leer flux with contact restoration, sharper contacts and boundary layers |
| `reconstruction` | `Upwind()` | 1st order, cell-centred values |
| | `MUSCL{VanLeer}()` | 2nd order MUSCL, smooth TVD limiter |
| | `MUSCL{MinMod}()` | 2nd order MUSCL, most diffusive TVD limiter |
| | `MUSCL{Superbee}()` | 2nd order MUSCL, least diffusive TVD limiter |
| `time_stepping` | `FEuler()` | Explicit forward Euler, 1st order (default) |
| | `RK2()` | Explicit 2-stage SSP Runge-Kutta, 2nd order |

`reconstruction` must be given. The time step is fixed with `Runtime(time_step=...)` or, with `adaptive=AdaptiveTimeStepping(maxCo=...)`, set every iteration from the CFL condition (convective plus diffusive wave speeds). Only `rho` takes a `convergence` entry in `solvers`, there are no linear solvers. For example (see `examples/2D_cylinder_supersonic.jl`):

For steady problems the explicit solver converges much faster with local (per-cell) pseudo-time stepping, `Runtime(...; adaptive=LocalTimeStepping(maxCo=1.0, startCo=0.2, ramp=200))`: each cell advances with its own time step from the spectral radius of its faces, `dt_i = CFL / (0.5 Σ_f (|U_i·n_f| + a_i) A_f / V_i + ν_eff Σ_f A_f² / V_i²)`, and the Courant number is ramped from `startCo` to `maxCo` over `ramp` iterations. Converged fields do not depend on the time step. The multistage schemes `RK3()`, `RK4()` and `RK5()` (Jameson, low storage) allow a larger Courant number and are meant for this use; implicit turbulence equations use the smallest cell time step. `run!(model, config; forces=(patches=[:wall], interval=50))` also records pressure and viscous force histories, and the returned residuals hold the L2 norms of the density, momentum and energy residuals (`rho`, `rhoU`, `rhoE`) and their values normalised by their value at the end of the Courant ramp (`rho_rel`, ...). See `examples/2D_cylinder_supersonic_LTS.jl`.

A boundary patch on which `U`, `p` and `T` are all `Dirichlet` (or `DirichletFunction`) is treated as a far field: the prescribed values are the free stream, and the boundary state is built from the characteristic (Riemann invariant) relations normal to the face, so outgoing waves leave the domain instead of reflecting. Supersonic faces take the interior state (outflow) or the free-stream state (inflow). Patches where `p` or `T` is not Dirichlet keep the reflecting ghost-state treatment.

```julia
schemes = (
    U              = Schemes(gradient=Gauss),
    p              = Schemes(gradient=Gauss),
    T              = Schemes(gradient=Gauss),
    flux           = HLLC(),
    reconstruction = MUSCL{VanLeer}(),
    time_stepping  = RK2(),
)
```

---
### Specifying schemes

XCALibre.jl flow solvers offer considerable flexibility to users for defining discretisation schemes. However, this means that discretisation schemes must be specified for every term of every equation solved. The schemes must be provided as a `NamedTuple` where each keyword corresponds to the fields being solved, e.g. (U = ..., p = ..., k = ..., <field> = ...). To facilitate this process, the [`Schemes`](@ref) function is provided. Used without any inputs `Schemes` uses the default values provided (see details below).

```@docs; canonical=false
Schemes
```

For example, below we set the schemes for the  `U` and `p` fields. Notice that in the first case the schemes will take their default values (entry for `p`). In the case of `U`, we are only changing the setting for the divergence scheme to `Upwind`.

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

schemes = (
    p = Schemes(), # no input provided (will use defaults)
    U = Schemes(divergence = Upwind, gradient=Midpoint, limiter=MFaceBased(model.domain)),
)

# output

```

## Linear solvers
---

Linear solvers in XCALibre.jl are provided by [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl). The following solvers types are re-exported in XCALibre.jl

- `Bicgstab()` is a general purpose linear solver. Works well with non-symmetric matrices e.g. for `U`.
- `Cg()` is particular strong with symmetric matrices e.g to solve the pressure equation.
- `Gmres()` is a general solver, restarted every 30 iterations. We have found it works best on the CPU backend.

Every linear solve stops when ‖D⁻¹(b − Ax)‖₂ ≤ `atol` + `rtol`·‖D⁻¹(b − Ax₀)‖₂, where D is the absolute diagonal of A and x₀ the field before the solve, whatever the solver, preconditioner, precision (`MixedF32`, `MixedF16`) or backend (`AMG`, PETSc on distributed meshes). D⁻¹(b − Ax) is the correction one Jacobi step would make, so `atol` is in the units of the field (e.g. Pa for pressure) and does not depend on how the equation is scaled; changing the preconditioner does not change what `atol` and `rtol` mean. With `Jacobi()` the solve is the usual Jacobi-preconditioned one. With other preconditioners `Bicgstab()`, `Cgs()` and `Gmres()` scale by D⁻¹ on the left and apply the preconditioner on the right, and `Cg()` checks D⁻¹r each iteration. PETSc solves on distributed meshes follow the same rule: `Jacobi()` monitors D⁻¹r directly; `Bicgstab()` and `Gmres()` solve the row-scaled system, with AMG preconditioners (`GAMG()`, `BoomerAMG()`) still built from A (the scaled matrix is applied on the fly, so no second copy is stored); `Cg()` and `Cgs()` with other preconditioners form D⁻¹r only once PETSc's own norm predicts the target is near, which costs a few extra matrix products per solve.

`Bicgstab()` and `Cgs()` break down when the residual becomes orthogonal to their shadow vector, which can happen when the initial residual is nonzero in a few cells only and the matrix couples them one way (upwind convection). Such a solve is repeated from its initial guess with a fixed, dense shadow vector; if it breaks down again, the field keeps its previous values and a warning is printed. Distributed solves continue from the last iterate instead, which restarts BiCGStab with the current residual as the shadow vector.

A linear solve that stops without meeting its `atol` or `rtol` prints a warning naming the method, the equation (its label on distributed meshes, e.g. `[p]`, or the vector component in serial runs), why it stopped, the number of iterations and the final residual norm, next to the tolerances, e.g.

```
┌ Warning: Cg solve [p] did not converge in 1000 iterations (KSP_DIVERGED_ITS): ‖D⁻¹(b - Ax)‖₂ = 3.1e-7, atol = 1.0e-7, rtol = 0.0 (shown at most 10 times per equation)
```

This covers every solve path: Krylov.jl (the reason is Krylov.jl's status, e.g. `maximum number of iterations exceeded`), `AMG`, mixed precision (`MixedF32`, `MixedF16`) and PETSc, where every negative `KSPConvergedReason` is reported (iteration limit, divergence, breakdown, indefinite preconditioner, NaN). Every path reports ‖D⁻¹(b − Ax)‖₂ of the returned solution. When a mixed-precision solve refines its correction (`rtol` below what the low precision attains), the warning is given when the refinement ends without meeting `rtol`, with ‖D⁻¹(b − Ax)‖₂ in full precision, and not for the individual corrections. Each equation shows at most 10 such warnings; on a distributed mesh rank 0 prints them.

For more information on these solvers you can review the excellent documentation provided by the [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) team. 

XCALibre.jl provides the `SolverSetup` convenience function for setting solvers. See details below. 

```@docs; canonical=false
SolverSetup
```

### Preconditioners 

XCALibre.jl offers a range of preconditioners which are subtypes of the abstract type `PreconditionerType`, exploring its subtypes we can find a list of the currently available preconditioners: 

```@repl
using XCALibre # hide
using AbstractTrees # hide
# import Main.subtypes as subtypes # hide
    using InteractiveUtils # hide
AbstractTrees.children(d::DataType) = Main.subtypes(d) # hide
print_tree(PreconditionerType) # hide
```

!!! note

    Only the `Jacobi` and `NormDiagonal` preconditioners have GPU ready implementations. At present these have the most robust implementation and they can be used with both CPU and GPU backends. The other preconditioners can only be used on the CPU. Notice that on our tests the `LDL` preconditioner only works when paired with the `Gmres()` on the CPU. Also notice that the implementations for `DILU` (experimental), `IC0GPU` and `ILU0GPU` (NVIDIA only), should be considered experimental. Work on improving the offering of preconditioners is ongoing.

!!! warning

    Internally the storage for sparse matrices was moved to the CSR format. Thus, temporarily, we have removed support for `LDL` and `ILU0` preconditioners while we work on CSR compatible implementations. 



Below an example is provided in context. Here, we are setting solvers for both the velocity field `U` and the pressure field `p` and packing them into a `NamedTuple` "solvers". The `Jacobi` preconditioner is used in both solvers. Notice that preconditioners are specified with an instance of their type i.e. `Jacobi()`. Internally, the preconditioner instance is used for dispatch. This tupple will then be passed on to create the final `Configuration` object.

```jldoctest;  filter = r".*"s => s"", output = false
using XCALibre

# Note: this example assumes a Physics object named `model` already exists

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
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
### Mixed-precision linear solves

Most of the time spent in a Krylov solve goes into reading the matrix and vectors from memory, so storing them with fewer bytes per value makes each iteration faster. The `precision` keyword of `SolverSetup` selects how each linear system is solved:

- `FullPrecision()` (default): the system is solved in the mesh's float type.
- `MixedF32()`: the full-precision residual `r = b - Ax` is formed, the correction `A d = r` is solved with the matrix, vectors and preconditioner in `Float32`, and `x + d` is formed in full precision. Fields and residuals keep full accuracy. Works on CPUs and GPUs with any serial preconditioner, and on distributed meshes through PETSc's `Float32` library.
- `MixedF16()`: as `MixedF32()` with the off-diagonal matrix entries stored in `Float16` (the Krylov vectors stay in `Float32`). Serial meshes and the `Jacobi` preconditioner only.

A single low-precision solve can reduce the residual only so far (about `1e-5` with `Float32`, `1e-3` with `Float16`). When `rtol` asks for more, the correction is repeated within the same solve until the full-precision residual meets `rtol`, so tight tolerances, as used in transient runs, are honoured; loose tolerances, as typical in steady runs, take a single correction and no extra work.

When it helps: solves limited by memory bandwidth, i.e. large meshes, and in particular GPUs, whose `Float64` throughput is often a small fraction of their `Float32` throughput. `MixedF16()` benefits only hardware with native `Float16` arithmetic (GPUs); on CPUs it is slower than `MixedF32()`. When it does not: small meshes that fit in cache, solves dominated by the preconditioner setup, and very ill-conditioned systems whose corrections need many refinements. Meshes whose neighbouring cells sit far apart in memory gain less, since reading scattered values costs the same whatever their size; reordering the mesh with [`reorder_mesh!`](@ref) addresses that.

```julia
solvers = (
    U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=0.7,
        rtol=1e-3, precision=MixedF32()),
    p = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-7, relax=0.3,
        rtol=1e-3, precision=MixedF32())
)
```

```@docs; canonical=false
FullPrecision
MixedF32
MixedF16
```
