# Distributed (MPI) simulations

XCALibre.jl can run a simulation across multiple MPI ranks (multi-core, multi-node, or
multi-GPU) using PETSc for the distributed linear solves. The physics setup, boundary
conditions, schemes and `run!` call are **identical** to a serial script — only the mesh is
partitioned and distributed. You do not write a solver, model, or boundary condition any
differently for parallel execution.

## Requirements

- `PETSc` and `MPI` added to your project environment.
- PETSc built (or configured) with MPI support. For the `BoomerAMG` preconditioner PETSc must
  additionally be configured with `--download-hypre`.
- For multi-GPU runs, a CUDA/ROCm-enabled PETSc build.

## Distributing the mesh

Wrap the mesh read in [`distribute`](@ref). The reader runs **only on rank 0**; the global
mesh is then partitioned (Metis) and scattered as one rank-local `DistributedMesh` per rank.
No manual `rank == 0` guard or `MPI.Init()` is required — `distribute` handles both:

```julia
using XCALibre, PETSc, MPI

comm = MPI.COMM_WORLD

mesh_dist = distribute(comm=comm) do
    UNV2D_mesh("path/to/mesh.unv", scale=0.001)
end
```

Pass `mesh_dist` as the model `domain` and assign boundary conditions against it exactly as in
serial. The rest of the script — `Physics`, `assign`, `SolverSetup`, `Schemes`, `Runtime`,
`run!` — is unchanged.

For very large meshes, partition once offline with [`partition_mesh`](@ref) and load per-rank
with `distribute(dir; comm)` to avoid the rank-0 memory bottleneck.

## Launching

Run the script under `mpiexec` with one process per rank, e.g. for 4 ranks:

```bash
julia --project=<env> -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=<env> your_case.jl`)'
```

Threads per rank are controlled with Julia's usual `-t` / `JULIA_NUM_THREADS`. See
`examples/2D_cylinder_U_mpi.jl` for a complete, runnable case including core-pinning notes.

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
`run!(model, config; petsc_options="-ksp_monitor -pc_type gamg")`.

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

Jacobi scales super-linearly while AMG scales sub-linearly, so AMG overtakes Jacobi around ~1M
cells and its lead widens beyond. AMG also drives the pressure residual far deeper per outer
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
