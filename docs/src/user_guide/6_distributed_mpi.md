# Distributed (MPI) simulations

XCALibre.jl can run a simulation across multiple MPI ranks (multi-core, multi-node, or
multi-GPU) using PETSc for the distributed linear solves. The physics setup, boundary
conditions, schemes and `run!` call are **identical** to a serial script — only the mesh is
partitioned and distributed. You do not write a solver, model, or boundary condition any
differently for parallel execution.

## Requirements

`PETSc` and `MPI` in your project environment. Nothing else: the binaries that `PETSc_jll` and
`MPI.jl` install are enough for a Float64 CPU run, with no preferences file and no shell
configuration. Two cases need more:

- `BoomerAMG()` needs a PETSc built with `--download-hypre`. The stock `Float64` libraries do
  carry hypre, so it works out of the box at the default precision; `Float32` builds do not.
- GPU-native solves need a CUDA- or ROCm-enabled PETSc, which `PETSc_jll` does not ship. Without
  one, pass `solve_on=CPU()` to `run!` and the linear solves are staged through the host.

A custom PETSc or system MPI is selected through `MPIPreferences` and PETSc's own preferences.
Julia resolves preferences per project environment and PETSc.jl generates its low-level wrappers
at precompilation for the configured library, so the scalar precision and the library path are a
property of the environment you run in, not something that can be switched at run time. Use a
separate project environment per PETSc build.

## Distributing the mesh

Wrap the mesh read in [`distribute`](@ref). Every rank makes the same call; the reader runs only
on rank 0 and the partitioned mesh is scattered, one rank-local `DistributedMesh` each. There is
no `MPI.Init()` and no `rank == 0` guard to write:

```julia
using XCALibre, PETSc, MPI

mesh_dist = distribute() do
    UNV3D_mesh("path/to/mesh.unv", scale=0.001)
end
```

!!! warning
    Do not assign the mesh, or any value that selects a method, inside a `rank == 0` branch of
    your own. A rank holding `nothing` where the others hold a mesh dispatches to a different
    method, and the run blocks forever waiting for a message that is never sent. Passing the
    reader to `distribute` is what makes that impossible.

For a large mesh, give `dir` as well. Rank 0 decomposes into that directory once and every rank
then loads only its own part, so no rank ever holds the global mesh after the first run:

```julia
mesh_dist = distribute(dir="parts") do
    UNV3D_mesh("path/to/mesh.unv", scale=0.001)
end
```

A decomposition already in `dir` for the same number of ranks is reused and the reader is never
called; one written for a different number of ranks is replaced. [`partition_mesh`](@ref) writes
the same layout from a standalone process if you would rather decompose ahead of time, and
`distribute(dir; comm)` loads it.

Pass `mesh_dist` as the model `domain` and assign boundary conditions against it exactly as in
serial. The rest of the script — `Physics`, `assign`, `SolverSetup`, `Schemes`, `Runtime`,
`run!` — is unchanged.

Two helpers cover what a parallel script still needs. [`is_root`](@ref) is true on rank 0 and
also true in a serial run where MPI was never initialised, so one guard works in both:

```julia
is_root() && println("final residual ", residuals.p[end])
```

and `bind_device!(backend)` binds the calling rank to its GPU without your script querying the
communicator.

## Launching

Install MPI.jl's launcher once. It resolves the same MPI binary the package itself uses:

```bash
julia --project=<env> -e 'using MPI; MPI.install_mpiexecjl()'
```

Then a run is one command:

```bash
mpiexecjl -n 4 julia --project=<env> your_case.jl
```

`mpiexecjl` lives in `~/.julia/bin`; add that to your `PATH`. Threads per rank use Julia's usual
`-t`, but one thread per rank is the right default: the CPU kernel backend is already serial, so
extra threads add overhead rather than removing it. Call `activate_multithread(backend)` in the
script — despite the name it pins BLAS to a single thread, which is what stops each rank taking
every core. On a machine with hyperthreading, bind one rank per physical core, for example
`mpiexecjl -n 4 --bind-to core --map-by core julia ...`.

See `examples/3D_BFS_mpi.jl` for a complete runnable case.

## Configuring the linear solvers

On a distributed mesh the solves go through PETSc. The `SolverSetup` fields map onto PETSc as
follows:

- `solver`: `Cg()` → `cg`, `Bicgstab()` → `bcgs`, `Gmres()` → `gmres` (PETSc `KSP` type).
- `preconditioner`: `Jacobi()` → `jacobi`, `BoomerAMG()` → `hypre`, `GAMG()` → `gamg` (PETSc
  `PC` type).
- `atol`, `rtol`, `itmax`: passed to PETSc `KSPSetTolerances` (absolute/relative residual
  tolerance and maximum iterations). These are the live convergence controls.
- `convergence`: used **only** as the PETSc absolute tolerance when both `atol` and `rtol` are
  set to `0`; otherwise it is ignored (set `atol`/`rtol` instead).

At solver construction each field prints the effective PETSc configuration once (from rank 0),
so you can confirm what PETSc actually received:

```
[ Info: PETSc solve [p]: KSP=cg PC=jacobi atol=1.0e-6 rtol=0.0 itmax=2000
```

Any solver or preconditioner not in the curated list, or any extra PETSc option, can be passed
through as a raw options string via the `petsc_options` keyword of `run!`, e.g.
`run!(model, config; petsc_options="-ksp_monitor -pc_type gamg")`. The same string is given to
PETSc at start-up, so options that must be set before PETSc initialises — `-log_view`,
`-use_gpu_aware_mpi 0` — go there too and need no environment variable. Start-up options take
effect on the first solver built, since PETSc initialises once per process.

## Algebraic multigrid for pressure (BoomerAMG and GAMG)

The pressure Poisson solve dominates incompressible runs, and its condition number worsens as
the mesh grows, so Jacobi-preconditioned CG needs more iterations at larger sizes. Algebraic
multigrid (AMG) builds a hierarchy of coarser problems and converges the pressure in a roughly constant number of
Krylov iterations independent of size. XCALibre exposes two AMG preconditioners, both SPD-only
(no transpose apply) and distributed-only (they error on a serial mesh):

- `BoomerAMG()` — HYPRE BoomerAMG (`-pc_type hypre`). Requires a PETSc build configured with
  `--download-hypre`.
- `GAMG()` — PETSc's native aggregation AMG (`-pc_type gamg`). No extra build flags; always
  available with PETSc.

```julia
p = SolverSetup(solver = Cg(), preconditioner = BoomerAMG(), rtol = 0.01, itmax = 1000, ...)
# or, needing no hypre build:
p = SolverSetup(solver = Cg(), preconditioner = GAMG(), rtol = 0.01, itmax = 1000, ...)
```

### When AMG helps

AMG carries a per-solve overhead (building/refreshing the hierarchy plus applying a V-cycle)
that Jacobi does not, so on small partitions it can be *slower* than Jacobi. The benefit grows
with problem size. On a tetrahedral backward-facing-step benchmark (8 ranks, per-iteration wall
time):

| cells | CG+Jacobi | CG+GAMG | CG+BoomerAMG |
|------:|----------:|--------:|-------------:|
| 0.5M  | 393 ms    | 464 ms  | 465 ms       |
| 2.7M  | 2194 ms   | 2117 ms | 2089 ms      |

Jacobi scales super-linearly while AMG scales sub-linearly, so AMG overtakes Jacobi somewhere
around a million cells on that machine and its lead widens beyond. Where the crossing falls
depends on the mesh, the rank count and the memory system, so measure it on your own case rather
than assuming the figure above: on a laptop at 1.3M tetrahedra we have since seen Jacobi still
ahead. AMG also drives the pressure residual far deeper per outer
iteration (6–8× here), which can cut the number of outer SIMPLE iterations needed to reach a
steady state (a further gain not visible in the per-iteration figure above). Rule of thumb: use
Jacobi for small/medium cases, AMG for large ones (especially when the pressure solve dominates).

### Rebuilding vs reusing the hierarchy

In SIMPLE the pressure matrix keeps a **fixed sparsity pattern** (no mesh refinement) but its
coefficients change slightly each outer iteration. Rebuilding the whole AMG hierarchy on every
solve is expensive, so both preconditioners avoid it — differently:

- `BoomerAMG(reuse = N)` freezes the hierarchy and rebuilds it only every `N` solves (default
  `10`; `reuse = 1` rebuilds every solve). HYPRE cannot partially reuse a hierarchy, so this
  all-or-nothing freeze is the only option; the frozen hierarchy remains a good preconditioner
  in cases where the matrix changes gently between rebuilds.
- `GAMG()` sets `reuse_interpolation = true` by default: it builds the aggregation and
  interpolation operators once and recomputes only the (cheap) coarse operators and smoothers
  each solve, so the hierarchy stays numerically current at a fraction of a full setup. This is
  valid precisely because the sparsity pattern never changes. `GAMG(reuse = N)` can additionally
  freeze the whole preconditioner for `N` solves if wanted (default `1`).

### Tuning keywords

Each keyword `k = v` is forwarded to PETSc and overrides a default.

**BoomerAMG** → `-pc_hypre_boomeramg_<k> v`. Defaults are tuned for 3D (`strong_threshold = 0.7`,
`coarsen_type = "HMIS"`, `interp_type = "ext+i"`, `agg_nl = 1`, `agg_num_paths = 2`) — HYPRE's own
defaults are 2D-oriented and build an over-complex, memory-heavy hierarchy in 3D. Common knobs:

- `strong_threshold` — strength-of-connection threshold; 0.5–0.7 for 3D.
- `coarsen_type` — coarsening algorithm: `"HMIS"`, `"PMIS"`, `"Falgout"`, ...
- `interp_type` — interpolation: `"ext+i"`, `"classical"`, ...
- `agg_nl` — number of aggressive-coarsening levels (lower operator complexity and memory).
- `relax_type_all` — smoother, e.g. `"SOR/Jacobi"`, `"Chebyshev"`, `"l1scaled-Jacobi"`.
- `grid_sweeps_all` — smoother sweeps per level.

```julia
BoomerAMG(strong_threshold = 0.6, coarsen_type = "PMIS", relax_type_all = "Chebyshev", reuse = 20)
```

**GAMG** → `-pc_gamg_<k> v`. Common knobs:

- `threshold` — aggregation strength threshold (e.g. `0.01`–`0.05`).
- `agg_nsmooths` — prolongator smoothing steps; `0` = unsmoothed aggregation (cheaper, often good
  for Poisson).
- `reuse_interpolation` — reuse aggregation/interpolation across solves (default `true` here).
- `coarse_eq_limit` — size at which the coarsest level is solved directly.

```julia
GAMG(threshold = 0.02, agg_nsmooths = 0)
```

See PETSc's `-pc_hypre_boomeramg_*` and `-pc_gamg_*` option lists for the full set; anything not
exposed as a keyword can still be supplied through the `petsc_options` keyword of `run!`.

!!! note
    Do not hard-code `BoomerAMG` as a default in shared scripts: a PETSc build without hypre
    errors at solver construction. `GAMG` needs no special build and is a safe default AMG.

## Terminal output

Informational `@info` messages are printed once (from rank 0) rather than once per rank;
warnings and errors still surface from every rank so rank-local failures remain visible. This
is handled automatically when the mesh is distributed — nothing is required in your script.

## What is and is not supported

Distributed today:

- Steady and transient incompressible flow through the SIMPLE and PISO families.
- `Laminar`, `KOmega` and `KOmegaSST` turbulence, including wall distance.
- CPU and GPU backends, periodic patches, and writing results in OpenFOAM's decomposed layout
  for reconstruction with the usual tools.

Not distributed, and these error or fall back rather than silently giving a wrong answer:

- The `KOmegaLKE` transition model and the LES models. They need the same synchronisation audit
  `KOmegaSST` received and have not had it.
- Float32 with `BoomerAMG`: the stock PETSc libraries carry hypre at Float64 only.
- GPU-native linear solves without a CUDA- or ROCm-enabled PETSc build. Use `solve_on=CPU()`.

Two behaviours to know about:

- `convergence` in a `SolverSetup` is not a distributed control. PETSc converges on `atol`,
  `rtol` and `itmax`; `convergence` is used only as the absolute tolerance when both `atol` and
  `rtol` are zero.
- `wall_distance!` can report that it did not converge while the residual is perfectly
  acceptable. It compares against a fixed threshold that predates the distributed path and the
  message is harmless.

## What to expect from parallel performance

The pressure and momentum solves are bandwidth-bound, so the useful rank count is set by memory
channels rather than cores. On a single-socket laptop with one memory controller, a
communication-free, perfectly balanced vector update already scales at only 63% from two ranks
to eight, which puts a ceiling on everything above it: on a 500k-cell tetrahedral
backward-facing step, parallel efficiency there was 87% at two ranks and 47% at eight. A machine
with more memory channels per core should do considerably better, and the figures above are not
a property of the solver.

Measure your own case before choosing a rank count, and prefer fewer, larger subdomains: each
Krylov iteration ends in a global reduction, so small subdomains spend a growing share of the
iteration synchronising rather than computing.
