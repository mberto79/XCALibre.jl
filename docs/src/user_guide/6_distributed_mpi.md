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
- `preconditioner`: `Jacobi()` → `jacobi`, `BoomerAMG()` → `hypre` (PETSc `PC` type).
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

## BoomerAMG for pressure

For distributed runs the pressure Poisson solve typically converges far better with algebraic
multigrid than with Jacobi. Use HYPRE BoomerAMG as the pressure preconditioner (requires the
`--download-hypre` PETSc build):

```julia
p = SolverSetup(solver = Cg(), preconditioner = BoomerAMG(), atol = 1e-6, rtol = 0.0, ...)
```

BoomerAMG is tuned with keyword arguments — each `k = v` becomes the PETSc option
`-pc_hypre_boomeramg_<k> v`:

```julia
BoomerAMG(strong_threshold = 0.7, coarsen_type = "HMIS")
```

See PETSc's `-pc_hypre_boomeramg_*` options for the full list; anything not needed as a keyword
can still be supplied through `petsc_options`. `BoomerAMG` is SPD-only (no transpose apply) and
is distributed-only — it errors if used on a serial mesh.

!!! note
    Do not set BoomerAMG as a default preconditioner in shared scripts: a PETSc build without
    hypre will error at solver construction.

## Terminal output

Informational `@info` messages are printed once (from rank 0) rather than once per rank;
warnings and errors still surface from every rank so rank-local failures remain visible. This
is handled automatically when the mesh is distributed — nothing is required in your script.
