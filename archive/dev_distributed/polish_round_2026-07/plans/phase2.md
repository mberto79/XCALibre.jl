# phase 2: launch ergonomics (#2)
## goal
A normal XCALibre script runs under MPI with ~2 distributed-specific lines. No prun!.
## design
- Add overload `distribute(read::Function; comm=MPI.COMM_WORLD, periodic_patches=())` in
  Distribute_1_partition.jl: on rank 0 mesh=read(), else nothing, then call existing
  distribute(mesh; comm, periodic_patches). distribute already does MPI.Init internally.
  Export it (already exported name; just new method).
- Script becomes:
    using XCALibre, PETSc, MPI
    mesh = distribute(comm=MPI.COMM_WORLD) do
        UNV2D_mesh(path, scale=0.001)
    end
  (PETSc/MPI still `using`-ed since backend ext + comm needed for final print; that is fine.)
- Optional tiny helper for the final rank-0 print if it reads cleanly — else leave the
  `MPI.Comm_rank(comm)==0 && println(...)` as-is (it is one honest line).
## steps
- [ ] add distribute(::Function; …) overload + docstring
- [ ] update examples/2D_cylinder_U_mpi.jl to the reduced form
- [ ] smoke run n=2 (1 iter), then confirm full example runs
## gate
2D_cylinder_U_mpi.jl runs to completion n=4 producing same final residuals order as before
(compare to prior run in dev/baselines.json). Boilerplate reduced (visual/diff check).
## risks/assumptions
- read() must run ONLY on rank 0 (heavy IO); assert inside overload.
- keep the existing distribute(mesh;…) and distribute(dir;…) methods intact.
