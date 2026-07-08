# Phase 1 — Partitioning + local mesh construction (High; critical phase)

Umbrella: `distributed_plan_detailed.md` §2 (local-serial principle, processor faces as
interior faces, §2.3). Everything below is single-process testable — extraction happens on
rank 0 anyway; the MPI scatter is a thin wrapper tested with 2 ranks.

## Files
- `src/Distribute/Distribute.jl` — module: `using MPI, Metis, SparseArrays, StaticArrays, Accessors, Adapt`; `using XCALibre.Mesh`; includes numbered files; exports.
- `src/Distribute/Distribute_0_types.jl` — `Partition`, `ProcessorPatch`, `DistributedMesh`, `AbstractDistributedSolver` (stub for Phase 3).
- `src/Distribute/Distribute_1_partition.jl` — `build_dual_graph`, `partition_cells`, `extract_subdomain`, `decompose`, `distribute`.
- `src/XCALibre.jl` — include after Solvers + `@reexport`.
- `test/distributed/test_partition.jl` — single-process closure tests.

## Types
```julia
struct Partition{VI<:AbstractVector{<:Integer}}
    rank::Int; nranks::Int          # MPI rank convention: 0-based
    n_owned::Int; n_ghost::Int
    local_to_global::VI             # NEW block-contiguous global ids, owned then ghosts
    owner::VI                       # owning MPI rank per local cell
    row_start::Int; row_end::Int    # owned global row block (1-based)
end
struct ProcessorPatch{VI}
    neighbour::Int                  # neighbour MPI rank
    faces::VI                       # local processor-face IDs shared with neighbour
    send_cells::VI                  # owned local cells to send (ordering invariant below)
    recv_ghosts::VI                 # ghost local cells to fill (ordering invariant below)
end
struct DistributedMesh{M<:AbstractMesh,P,PP,VI} <: AbstractMesh
    mesh::M                         # local Mesh3/Mesh2, owned+ghost cells
    partition::P
    procs::Vector{PP}
    orig_cells::VI                  # ORIGINAL global cell id per local cell (I/O, tests)
    orig_faces::VI                  # ORIGINAL global face id per local face
end
```
- `getproperty` forwarding: own fields via `getfield`, everything else forwarded to
  `getfield(dm, :mesh)` — so `ScalarField(dmesh)`, `Physics(domain=dmesh)`, kernels see a
  normal mesh (the §2.2 spike). Also define `propertynames` and `Base.show`
  (the generic `AbstractMesh` show assumes Mesh2/Mesh3 and would error).

## Algorithms & ordering rules (the correctness core)
1. `build_dual_graph(mesh)`: edges from interior faces (`fID > n_bfaces`, detect via
   `ownerCells[1] != ownerCells[2]`); symmetric `sparse(I, J, 1, n, n)`.
2. `partition_cells(mesh, nparts)`: `Metis.partition(G, nparts; alg=:KWAY)` → `Vector{Int}`
   1-based parts. `nparts == 1` short-circuits to `ones`. Log balance + edge-cut.
3. Global block renumbering: new global id = rank-block offset + position of the cell
   within its part in ORIGINAL cell order. `row_start(r) = 1 + sum(counts[1:r-1])`.
4. `extract_subdomain(mesh, parts, part)` (1-based part; MPI rank = part-1):
   - owned cells: original order. Ghosts: neighbours across cut edges (via
     `cell_neighbours`), unique, sorted by `(owning part, original id)` → each neighbour's
     ghosts form a contiguous, ascending block.
   - faces kept: physical boundary faces owned locally (patch-grouped, original order
     within patch, ALL patches kept — empty `IDs_range` allowed so BC assignment by name
     works on every rank), then interior faces with ≥1 owned owner (original order).
     Processor faces (owned–ghost) are interior faces — implicit coupling for free (§2.3).
   - rebuild `Face3D/Face2D` via Accessors `@reset` (only `ownerCells`, `nodes_range`
     change); copy centre/normal/e/area/delta/weight VERBATIM (serial geometry bitwise
     preserved; ghost centres/volumes copied verbatim too, keeping face geometry valid).
   - `cell_faces/cell_neighbours/cell_nsign`: internal-faces-only invariant preserved.
     Owned cells keep their full internal-face list (all present by ghost construction).
     Ghost cells keep ONLY their locally-present processor faces (their other faces don't
     exist locally); ghost-side cell loops write only to ghost rows — garbage, harmless.
   - nodes: union of local cells' `cell_nodes` + local faces' `face_nodes`, sorted by
     original id; `node_cells` inverted from local `cell_nodes`.
   - `boundary_cellsID`: remapped owned ids, boundary faces first — mesh invariant intact.
5. ORDERING INVARIANT (Phase 2 depends on it): for ranks r,q the lists
   `send_cells` on q (owned cells adjacent to r) and `recv_ghosts` on r (ghosts owned by q)
   are BOTH sorted by original global cell id → buffers align index-for-index with no
   further communication. Asserted in tests.
6. `decompose(mesh, nparts)` → `Vector{DistributedMesh}` (single-process; used by all
   Phase 1 tests). `distribute(mesh; comm)` → rank 0: partition + `extract_subdomain` per
   rank + `MPI.send` (object serialization); ranks ≠ 0: `MPI.recv`. `nranks == 1` builds
   locally without communication.

## Tests (`test/distributed/test_partition.jl`, nparts = 1, 2, 4, single process)
Grid: `examples/0_GRIDS/3d_box_1000x1000x1000mm_10.unv` via `UNV3D_mesh` (1000 cells).
- sum of `n_owned` == global ncells; owned `local_to_global` blocks are a partition of
  `1:ncells`; `orig_cells` over owned cells is a bijection onto `1:ncells`.
- global volume sum over owned cells == serial total (rtol 1e-12).
- ghost cell centre/volume == original global cell's, bitwise.
- per-patch boundary face counts sum to serial counts; boundary faces are `1:n_bfaces`
  locally with `ownerCells[1]==ownerCells[2]==boundary_cellsID[fID]`.
- via `orig_faces`: owned–owned interior faces appear on exactly 1 rank; processor faces
  on exactly 2 ranks with identical geometry; each global interior face accounted for.
- ProcessorPatch symmetry: faces(r→q) count == faces(q→r); orig ids of q's `send_cells`
  == orig ids of r's `recv_ghosts` from q, in order (invariant 5).
- local connectivity: every owned cell's `faces_range` length matches global; `nsign`
  consistent with remapped `ownerCells`; `cell_neighbours` ids valid local ids.
- spike: `ScalarField(dm)` works, `length(values) == n_owned + n_ghost`; `show(dm)` works.
- nparts=1 degenerate: same cell/face/node counts as serial mesh.

## Exit criteria
All above green via `run_gate.jl`. MPI scatter smoke test (`-n 2`: ranks report matching
`n_owned` totals) — attempted; if local MPI env blocks it, record in dev/STATE and gate it
in Phase 2 with the harness.
