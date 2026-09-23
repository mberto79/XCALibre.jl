# architecture: XCALibre.jl distributed (MPI) module

What is true of the code today. Why a mechanism was chosen belongs in `dev/decisions.md`.

## shape

- `src/Distribute/` is an ordinary submodule of `XCALibre`, re-exported like every other. It owns partitioning, the halo exchange, distributed field wrappers, the distributed linear-algebra seam, distributed equation assembly and the decomposed writer.
- `ext/XCALibrePETScExt.jl` is loaded when `PETSc` is present and supplies the only distributed linear solver, mapping XCALibre solver and preconditioner types onto PETSc's options system.
- `test/distributed/` holds the multi-rank suite, driven by a launcher that spawns `mpiexec` children.
- MPI and Metis are hard dependencies of the package; PETSc is a weak dependency behind an extension.

## flow

- A global mesh is partitioned into per-rank subdomains, either in the same run or offline into a directory that every rank later reads. Each rank receives its owned cells plus a ghost layer and a map from local to global indices.
- A rank's part comes from one of three sources with the same local layout: rank 0 extracting from the global mesh, a binary part file per rank written offline (one header for serial and partitioned meshes, format number checked), or a decomposed OpenFOAM case that each rank reads alone, building its ghosts by one exchange across the processor patches.
- Processor-patch rules: both sides list an interface's faces in the same order; ghosts are ordered by owning rank then global id and each side's send and receive lists are sorted by global id, so halos align without negotiation; ghost cells carry only interface faces (and, from the OpenFOAM reader, no nodes); global ids are rank blocks in rank order. The extractor keeps the global face orientation (the owner may be a ghost) while the OpenFOAM reader makes the owned cell the owner; the decomposed writer flips such faces so written normals point out of the owned cell, which makes writer and reader inverse up to geometry recomputed from full-precision points. Original cell ids travel as `cellProcAddressing`; without it they are the block ids, and the reader leaves original face ids unset.
- Mesh readers build Int32 indices by default; a mesh too large for them stops with an error naming `integer_type=Int64` (D194). Stored index arrays downstream (equation matrices, AMG hierarchy, periodic maps) take the mesh's index type (D192, D193).
- Global cell and face ids (`local_to_global`, original ids) are `Int64` whatever the mesh index type, since they outgrow Int32 before any per-rank count does (D191).
- Owned rows form the contiguous prefix of each rank's sparse matrix, so the owned submatrix ships to PETSc without gathering, and ghost rows are never sent.
- On the host PETSc's matrix is created once from the local CSR and each assembly scatters values straight into its diagonal and off-diagonal blocks, which works because every owned row lists owned columns then ghosts, each ascending in global id; device matrices go through a coordinate map built once and read the values in place; on a GPU the matrix, right-hand side and solution stay on the device, and a GPU run against a PETSc without device support stops at setup. Of the PETSc libraries matching the field precision, the one with the narrowest index type that addresses the global system is used. GPU messages between ranks go direct when MPI reports CUDA awareness and are staged through the host otherwise, for the halo exchange and PETSc alike (D71).
- PETSc's x and b own no storage: each solve places the field's owned prefix and the equation's right-hand side in them and resets them afterwards. Each process keeps one live solver per equation label and destroys it collectively when that label is wrapped again, so repeated runs do not leak.
- Field values crossing a partition boundary are refreshed by the halo exchange before any operation that reads a neighbour; gradient, interpolation and turbulence paths each declare where that refresh happens.
- Each mesh holds one halo schedule per width (scalar 1, vector 3, scalar with vector 4), built on first use, shared by every field and equation, tagged by width and driven by persistent requests freed at MPI finalize. A vector solve exchanges all components once after the last one and takes each residual against that component's saved diagonal; residual sums of one equation cross ranks in one all-reduce.
- The solver entry point routes to the distributed path on the mesh type, so physics, boundary, scheme and runtime setup are shared with the serial path unchanged.
- Results are written per rank in OpenFOAM's decomposed binary layout, including the face flux and the loop position, which is also the restart checkpoint: a restart restores cell fields, time and time step before the solver's initial calculations and the face flux after them, so the resumed loop state equals the written one.

## mesh storage

- A mesh's cells, faces and nodes are stored one array per field as top-level `Mesh2`/`Mesh3` fields (`cell_volume`, `face_area`, `node_coords`, ...), same-typed arrays sharing one of nine type parameters (D177, D179); `mesh.faces`/`cells`/`nodes` are rebuilt in `getproperty` as zero-copy `FaceArrays`/`CellArrays`/`NodeArrays` views. Constructors take element vectors, readers build plain vectors, `mesh.faces[i]` rebuilds an element, and `mesh.faces.area[i]` reads the array. Anything leaving the device goes through `adapt`, written out per column so it stays inferable (D178); binary part files pack the columns back into element records. `get_float`/`get_int` are one-element arrays stored like `face_area`/`cell_nodes`, carrying only the mesh float and integer types (D182).
- GPU kernel arguments are copied by value into per-thread local memory, so the discretise kernels receive mesh-free copies of terms, sources and fields (a scheme reaches `term.phi.mesh.face_gDiff` and nothing else of the mesh) plus the cell and face columns they read, never `mesh` or `model` (D184, D187).

## interfaces

- The distributed mesh type is what the solver dispatches on; anything that changes it changes the serial-to-distributed routing.
- The PETSc extension imports the distributed solver type and its assemble and solve functions by name from the distributed submodule.
- The solver global-reduction seam is the identity in serial and an all-reduce in distributed; any new convergence quantity must cross it or ranks will disagree about when to stop.
- Periodic pairs are contracted before partitioning so matched cells share a rank and the serial periodic kernels run unchanged.
