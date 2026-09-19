# P1-M23 - parallel preprocessing (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R11. Governing decisions: D72, D73.

## Problem, quantified

Every mesh passes through rank 0: `distribute(mesh)` and `partition_mesh` read the global mesh (1.6 KB/cell), run Metis and extract each part there (`Distribute_1_partition.jl:284-296`, `:350-355`), and the parts are Julia `serialize` files that must be regenerated after any upgrade (`:348`). No reader builds a `DistributedMesh` from an already-decomposed case, though the writer emits OpenFOAM's `processor<rank>/` layout (`Distribute_7_io.jl`). After M20 the serial path is O(N+P) but still needs one node holding the mesh.

## Approach

Three independent pieces, each usable alone. (1) A binary part format with a header. (2) A decomposed-OpenFOAM reader: each rank reads its own `processor<rank>/constant/polyMesh` with the existing FOAM reader, then ghosts are built by exchanging owner-cell geometry over the processor patches, whose face order OpenFOAM guarantees to match on both sides. (3) A parallel repartition of any `DistributedMesh` through PETSc's `MatPartitioning` (wrapped) so a naive decomposition becomes a balanced one without rank 0. Everything is verified here with `decomposePar` from OpenFOAM 12 or 2512 (both installed) at n=2,4,8.

## Configuration space

case {BFS 10 mm, BFS 5 mm} x decomposition {XCALibre Metis, `decomposePar -method scotch`, `-method simple`} x ranks {2, 4, 8}; verdict per case: fields versus serial to the `test_psimple.jl` bar (1e-5), `reconstructPar` round-trip of written results, and the M19 `check_ghosts` zero.

## Steps

- [x] **P1-M23-S1** DELIVERED (D123): binary part format `rank_<r>.xdm` (D119, D120): one header with the same entries for every file (magic, format version, XCALibre version, `kind=serial|partitioned`, nranks, rank, TI, TF, n_owned, n_ghost, counts of every array; a serial mesh has nranks 1, rank 0, n_ghost 0 and zero partition counts) read and written by one code path, then a mesh block (the serial mesh arrays) and a partition block (empty when serial); a public `mesh_info(path)` returns the header so a user can tell a serial mesh from a part and see its rank count and types, and each loader given the other kind errors naming the right call (`distribute(dir)` under `mpiexec -n <nranks>`); the serial writer and loader may be built here when the code is in place, but ship in a separate PR off main with the serial format; `partition_mesh` writes it, `distribute(dir)` reads it, `.jls` reading is removed - mechanism: a versioned layout independent of Julia's serializer - cost: none - verdict: round-trip equality on all `DistributedMesh` fields; load time at 4 mm P=8 recorded against `.jls`; `test_offline.jl` green.
- [ ] **P1-M23-S2** `distribute(::FOAMCase)` (name to match the reader API): each rank reads `processor<rank>/constant/polyMesh` into a local `Mesh3` with `processor` patches recognised; the `procBoundary` face lists give the interface; neighbours exchange `(centre, volume)` of the owner cells behind each interface face and the ghost cells are appended with only the interface faces attached, exactly as `extract_subdomain` leaves them; `orig_cells` from `cellProcAddressing` when present, else block numbering - mechanism: the same local layout the serial extractor produces, built from local files - cost: one neighbour exchange at start-up - verdict: BFS `decomposePar -method scotch` at n=2,4,8 matches serial to 1e-5 in `test_psimple.jl` form; `check_ghosts` zero; results written and `reconstructPar` completes.
- [ ] **P1-M23-S3** processor-patch ownership rules documented in `dev/architecture.md`: normals point out of the owned cell, face order matches on both sides, ghost cells keep only interface faces; the decomposed writer and reader are inverse to each other (write then read gives a bitwise-equal `DistributedMesh` up to renumbering) - mechanism: documentation plus one test - verdict: `test_io.jl` round-trip green.
- [ ] **P1-M23-S4** `repartition(dm; comm)`: build the distributed dual graph as a PETSc `Mat` (adjacency rows for owned cells, global column ids from `local_to_global`), `MatPartitioningCreate` with `parmetis` or `ptscotch` when `PetscHasExternalPackage` reports one (conda PETSc: verify; stock `PETSc_jll`: verify, else the step errors with the build hint), migrate cells with `Alltoallv` of the per-cell records and rebuild ghosts as in S2 - mechanism: no rank ever sees the global graph - cost: one partition and one migration at start-up - verdict: `decomposePar -method simple` at n=8 repartitioned gives an edge-cut within 20 percent of serial Metis on the same mesh and fields match serial to 1e-5; balance max/min owned cells within 1.05.
- [ ] **P1-M23-S5** the guide's "Distributing the mesh" section gains the decomposed-case route and states which route each mesh size and machine calls for - mechanism: documentation - verdict: docs build green with a doctest at one rank.

## Exit criterion

S1, S2, S3 landed and green at n=2,4,8; S4 landed if a parallel partitioner exists in one local env (else WITHDRAWN here and reopened in P2 with the HPC build); `dev/telemetry/preprocessing.md` holds load times and edge-cuts.

## Open questions

- Whether OpenFOAM 12 and 2512 write `cellProcAddressing` in the same place and format; settled by decomposing the BFS once with each.
- Whether the conda PETSc carries parmetis or ptscotch (`PetscHasExternalPackage`); settled at S4 start.
