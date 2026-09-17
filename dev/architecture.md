# architecture: XCALibre.jl distributed (MPI) module

What is true of the code today. Why a mechanism was chosen belongs in `dev/decisions.md`.

## shape

- `src/Distribute/` is an ordinary submodule of `XCALibre`, re-exported like every other. It owns partitioning, the halo exchange, distributed field wrappers, the distributed linear-algebra seam, distributed equation assembly and the decomposed writer.
- `ext/XCALibrePETScExt.jl` is loaded when `PETSc` is present and supplies the only distributed linear solver, mapping XCALibre solver and preconditioner types onto PETSc's options system.
- `test/distributed/` holds the multi-rank suite, driven by a launcher that spawns `mpiexec` children.
- MPI and Metis are hard dependencies of the package; PETSc is a weak dependency behind an extension.

## flow

- A global mesh is partitioned into per-rank subdomains, either in the same run or offline into a directory that every rank later reads. Each rank receives its owned cells plus a ghost layer and a map from local to global indices.
- Owned rows form the contiguous prefix of each rank's sparse matrix, so the owned submatrix ships to PETSc without gathering, and ghost rows are never sent.
- Field values crossing a partition boundary are refreshed by the halo exchange before any operation that reads a neighbour; gradient, interpolation and turbulence paths each declare where that refresh happens.
- The solver entry point routes to the distributed path on the mesh type, so physics, boundary, scheme and runtime setup are shared with the serial path unchanged.
- Results are written per rank in the decomposed layout that OpenFOAM's reconstruction tools expect.

## interfaces

- The distributed mesh type is what the solver dispatches on; anything that changes it changes the serial-to-distributed routing.
- The PETSc extension imports the distributed solver type and its assemble and solve functions by name from the distributed submodule.
- The solver global-reduction seam is the identity in serial and an all-reduce in distributed; any new convergence quantity must cross it or ranks will disagree about when to stop.
- Periodic pairs are contracted before partitioning so matched cells share a rank and the serial periodic kernels run unchanged.
