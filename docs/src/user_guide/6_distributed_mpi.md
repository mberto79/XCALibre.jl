# Distributed (MPI) simulations

XCALibre.jl can run a simulation across several MPI ranks (multi-core, multi-node or multi-GPU),
using PETSc for the distributed linear solves. The physics setup, boundary conditions, schemes and
`run!` call are the same as in a serial script. Only the mesh is partitioned and distributed, so
you do not write a solver, model or boundary condition any differently for parallel execution.

## Requirements

You need `PETSc` and `MPI` in your project environment, and nothing else. The binaries that
`PETSc_jll` and `MPI.jl` install are enough for a Float64 CPU run, with no preferences file and no
shell configuration. Two cases need more:

- `BoomerAMG()` needs a PETSc built with hypre. The stock `Float64` libraries include hypre, so it
  works out of the box at the default precision. `Float32` builds do not include it.
- GPU runs need a CUDA- or ROCm-enabled PETSc, which `PETSc_jll` does not ship. Without one, a
  GPU run stops with an error; it is never moved onto the host.

A custom PETSc or system MPI is selected through `MPIPreferences` and PETSc's own preferences.
Julia resolves preferences per project environment, and PETSc.jl generates its low-level wrappers
at precompilation for the configured library. The scalar precision and the library path are
therefore fixed by the environment you run in and cannot be switched at run time. Use a separate
project environment for each PETSc build.

## Distributing the mesh

Wrap the mesh read in [`distribute`](@ref). Every rank makes the same call. The reader runs only on
rank 0, and the partitioned mesh is scattered so that each rank gets its own `DistributedMesh`.
You do not need to call `MPI.Init()` or write a `rank == 0` guard:

```jldoctest distributed; filter = r".*"s => s"", output = false
using XCALibre, PETSc

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "backwardFacingStep_10mm.unv")

mesh = distribute() do
    UNV2D_mesh(mesh_file, scale=0.001)
end

# output

```

!!! warning
    Do not assign the mesh, or any value that selects a method, inside a `rank == 0` branch of
    your own. A rank holding `nothing` where the others hold a mesh dispatches to a different
    method, and the run then blocks forever waiting for a message that is never sent. Passing the
    reader to `distribute` rules this out.

For a large mesh, also give `dir`. Rank 0 decomposes the mesh into that directory once, and every
rank then loads only its own part, so after the first run no rank ever holds the global mesh:

```julia
mesh = distribute(dir="parts") do
    UNV3D_mesh("path/to/mesh.unv", scale=0.001)
end
```

If `dir` already holds a decomposition for the same number of ranks, it is reused and the reader
is never called. A decomposition for a different number of ranks is replaced.
[`partition_mesh`](@ref) writes the same layout from a standalone process if you want to decompose
ahead of time, and `distribute(dir; comm)` loads it.

## Setting up and running a case

Pass the distributed mesh as the model `domain` and assign boundary conditions against it, exactly
as in serial. The rest of the script is unchanged:

```jldoctest distributed; filter = r".*"s => s"", output = false
velocity = [0.5, 0.0, 0.0]
nu = 1e-3

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh
    )

BCs = assign(
    region = mesh,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Extrapolated(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Symmetry(:top)
        ],
        p = [
            Extrapolated(:inlet),
            Dirichlet(:outlet, 0.0),
            Extrapolated(:wall),
            Symmetry(:top)
        ]
    )
)

solvers = (
    U = SolverSetup(
        solver = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax = 0.7,
        rtol = 1e-1
    ),
    p = SolverSetup(
        solver = Cg(),
        preconditioner = GAMG(),
        convergence = 1e-7,
        relax = 0.3,
        rtol = 1e-2
    )
)

schemes = (
    U = Schemes(divergence = Linear),
    p = Schemes()
)

runtime = Runtime(iterations = 5, write_interval = -1, time_step = 1)
hardware = Hardware(backend = CPU(), workgroup = 1024)

config = Configuration(
    solvers = solvers, schemes = schemes, runtime = runtime, hardware = hardware, boundaries = BCs)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)

is_root() && println("final pressure residual ", residuals.p[end])

# output

```

Two helpers cover what a parallel script still needs. [`is_root`](@ref), used above, is true on
rank 0. It is also true in a serial run where MPI was never initialised, so the same guard works in
both. `bind_device!(backend)` binds the calling rank to its GPU without your script querying the
communicator.

## Launching

Install MPI.jl's launcher once. It resolves the same MPI binary that the package itself uses:

```bash
julia --project=<env> -e 'using MPI; MPI.install_mpiexecjl()'
```

After that, a run is one command:

```bash
mpiexecjl -n 4 julia --project=<env> your_case.jl
```

`mpiexecjl` lives in `~/.julia/bin`, so add that directory to your `PATH`. You can set threads per
rank with Julia's usual `-t` flag, but one thread per rank is the right default. The CPU kernel
backend is already serial, so extra threads add overhead rather than saving time. Call
`activate_multithread(backend)` in the script. Despite its name, it pins BLAS to a single thread,
which stops each rank from taking every core. On a machine with hyperthreading, bind one rank per
physical core, for example `mpiexecjl -n 4 --bind-to core --map-by core julia ...`.

## PETSc solvers in a single process

The same script runs without `mpiexecjl`. Started with plain `julia your_case.jl`, `distribute`
initialises MPI with a single rank and every linear solve goes through PETSc. This is how you use
PETSc's solvers and preconditioners, such as `GAMG()`, in an ordinary serial run, as an alternative
to the Krylov.jl solvers used on a serial mesh. Your script does not need `using MPI`, since
XCALibre depends on it already.

On a GPU backend, the fields stay on the device and the linear solves need a GPU-enabled PETSc.

## Configuring the linear solvers

On a distributed mesh the solves go through PETSc. The `SolverSetup` fields keep their serial
meaning:

- `solver`: `Cg()` → `cg`, `Cgs()` → `cgs`, `Bicgstab()` → `bcgs`, `Gmres()` → `gmres` (PETSc
  `KSP` type).
- `preconditioner`: `Jacobi()` → `jacobi`, `DILU()` → `bjacobi`, `GAMG()` → `gamg`, `BoomerAMG()`
  → `hypre` (PETSc `PC` type). PETSc has no DILU, so `DILU()` maps to its closest relative, block
  Jacobi with an ILU(0) factorisation of each rank's block, and warns once that it has done so.
- `atol`, `rtol`, `itmax`: the stopping tolerances and iteration limit of each linear solve, as in
  serial.
- `convergence`: the residual target that stops the outer iteration, as in serial. It does not
  affect the linear solver.

When each solver is built, rank 0 prints the configuration PETSc received:

```
[ Info: PETSc solve [p]: KSP=cg PC=gamg atol=1.0e-6 rtol=0.01 itmax=1000 reuse_interpolation=true
```

The `petsc_options` keyword of `run!` passes a raw PETSc options string, which reaches every
solve. Use it for anything the list above does not cover. It overrides the mapped types, and it
also names a type for a solver or preconditioner that has no mapping:

```julia
run!(model, config; petsc_options="-pc_type sor -ksp_monitor")
```

PETSc also reads this string at start-up, so options that must be set before PETSc initialises,
such as `-log_view` or `-use_gpu_aware_mpi 0`, go there too and need no environment variable.
Start-up options take effect with the first solver built, since PETSc initialises once per
process.

## Algebraic multigrid for pressure (GAMG and BoomerAMG)

The pressure Poisson solve dominates incompressible runs. Its condition number worsens as the mesh
grows, so Jacobi-preconditioned CG needs more iterations on larger meshes. Algebraic multigrid
(AMG) builds a hierarchy of coarser problems and converges the pressure in a number of Krylov
iterations that is roughly independent of mesh size. Two AMG preconditioners are available. Both
are for symmetric positive-definite systems only (they have no transpose apply), and both run only
through PETSc:

- `GAMG()`: PETSc's native aggregation AMG (`-pc_type gamg`). It needs no extra build flags.
- `BoomerAMG()`: HYPRE BoomerAMG (`-pc_type hypre`). It needs a PETSc built with hypre.

```julia
p = SolverSetup(solver = Cg(), preconditioner = GAMG(), convergence = 1e-7, relax = 0.3, rtol = 0.01)
```

**Prefer `GAMG()` when you want AMG.** It needs no special build, and on the cases measured so far
it was faster than `BoomerAMG()` at every rank count and scaled more evenly (see
[Benchmarks](@ref)). `BoomerAMG()` remains available for operators where classical coarsening
suits better.

!!! note "AMG results depend on the rank count"
    An AMG hierarchy is built from each rank's local partition. It therefore changes with the rank
    count, and runs at different rank counts converge to the same solution along slightly
    different paths. `Jacobi()` does not depend on the partition and gives identical residuals at
    every rank count, so use it when you need to compare runs at different rank counts exactly.

### When AMG helps

AMG adds a cost to every solve that Jacobi does not have: building or refreshing the hierarchy,
then applying a V-cycle. On small partitions this can make AMG *slower* than Jacobi. Jacobi's
iteration count grows with mesh size while AMG's does not, so AMG overtakes Jacobi above some
problem size. That crossover depends on the mesh, the rank count and the memory system, so measure
it on your own case. AMG also reduces the pressure residual much further in each outer iteration,
which can cut the number of outer iterations needed to reach a steady state. As a rule of thumb,
use Jacobi for small and medium cases and AMG for large ones, especially when the pressure solve
dominates.

### Rebuilding vs freezing the hierarchy

In SIMPLE, the pressure matrix keeps a fixed sparsity pattern while its coefficients change
slightly from one outer iteration to the next. Rebuilding the whole AMG hierarchy for every solve
is expensive, so both preconditioners avoid it, in different ways:

- `GAMG()` sets `reuse_interpolation = true` by default. It builds the aggregation and
  interpolation operators once and then, for each solve, recomputes only the coarse operators and
  smoothers, which is cheap. The hierarchy stays numerically current at a fraction of the cost of
  a full setup. This is valid only because the sparsity pattern never changes. `GAMG(freeze = N)`
  goes further and holds the whole preconditioner fixed for `N` solves, skipping even that update
  (default `25`; `freeze = 1` updates for every solve).
- `BoomerAMG(freeze = N)` holds the whole preconditioner fixed for `N` solves and rebuilds it from
  the current matrix on the `N`th (default `10`; `freeze = 1` rebuilds for every solve). HYPRE
  cannot partially reuse a hierarchy, so this all-or-nothing freeze is the only option.

The Krylov solver always uses the current matrix, so a frozen preconditioner changes how fast each
solve converges but not the solution it converges to. A longer freeze saves setup time and can cost
residual reduction per outer iteration. The defaults balance the two.

### Tuning keywords

Each keyword `k = v` is forwarded to PETSc and overrides a default.

**GAMG** → `-pc_gamg_<k> v`. Common options:

- `threshold`: aggregation strength threshold (e.g. `0.01`–`0.05`).
- `agg_nsmooths`: prolongator smoothing steps; `0` gives unsmoothed aggregation, which is cheaper
  and often good for Poisson problems.
- `reuse_interpolation`: reuse aggregation and interpolation across solves (default `true` here).
- `coarse_eq_limit`: the size at which the coarsest level is solved directly.

```julia
GAMG(threshold = 0.02, agg_nsmooths = 0)
```

**BoomerAMG** → `-pc_hypre_boomeramg_<k> v`. The defaults are tuned for 3D (`strong_threshold =
0.7`, `coarsen_type = "HMIS"`, `interp_type = "ext+i"`, `agg_nl = 1`, `agg_num_paths = 2`). HYPRE's
own defaults are 2D-oriented and build an overly complex, memory-heavy hierarchy in 3D. Common
options:

- `strong_threshold`: strength-of-connection threshold; 0.5–0.7 for 3D.
- `coarsen_type`: coarsening algorithm, such as `"HMIS"`, `"PMIS"` or `"Falgout"`.
- `interp_type`: interpolation, such as `"ext+i"` or `"classical"`.
- `agg_nl`: number of aggressive-coarsening levels, which lowers operator complexity and memory.
- `relax_type_all`: smoother, such as `"SOR/Jacobi"`, `"Chebyshev"` or `"l1scaled-Jacobi"`.
- `grid_sweeps_all`: smoother sweeps per level.

```julia
BoomerAMG(strong_threshold = 0.6, coarsen_type = "PMIS", relax_type_all = "Chebyshev", freeze = 20)
```

See PETSc's `-pc_gamg_*` and `-pc_hypre_boomeramg_*` option lists for the full set. Anything not
exposed as a keyword can still be passed through `petsc_options`.

!!! note
    Do not hard-code `BoomerAMG` as a default in shared scripts: a PETSc build without hypre
    raises an error when the solver is constructed. `GAMG` needs no special build and is a safe
    default AMG.

## Terminal output

Informational `@info` messages are printed once, from rank 0, rather than once per rank. Warnings
and errors still come from every rank, so failures on a single rank remain visible. This happens
automatically when the mesh is distributed, and your script needs nothing extra.

## What is and is not supported

Distributed today:

- Steady and transient incompressible flow through the SIMPLE and PISO families.
- `Laminar`, `KOmega` and `KOmegaSST` turbulence, including wall distance.
- CPU and GPU backends, periodic patches, and writing results in OpenFOAM's decomposed layout so
  that the usual tools can reconstruct them.

Not distributed yet. These raise an error rather than silently giving a wrong answer:

- The `KOmegaLKE` transition model and the LES models.
- Float32 with `BoomerAMG`, because the stock PETSc libraries include hypre at Float64 only.
- GPU runs without a CUDA- or ROCm-enabled PETSc build.

## What to expect from parallel performance

- **Measure with a steady clock.** Many CPUs, laptops in particular, lower their clock frequency as
  more cores become busy. This alone can look exactly like poor scaling, so pin the frequency or
  hold package power constant before comparing timings across rank counts.
- **Prefer fewer, larger subdomains.** Every Krylov iteration ends in a global reduction. With many
  small subdomains, the ranks spend a growing share of each iteration waiting at that reduction
  rather than computing.
- **Stop at one rank per physical core.** Beyond that, ranks compete for the same execution
  resources and gain nothing.
