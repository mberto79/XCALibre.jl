# Plan: memory-traffic scaling for XCALibre.jl (face data layout + single-pool threaded Krylov)

Status: ready to start · written 2026-09-23 · target branch: `HM/distributed-draft`
Code lives in `~/Julia/XCALibre.jl` (branch `HM/distributed-draft`, HEAD `875c16bb`, identical to
the revision benchmarked). Benchmarks live in `~/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS`.

Two changes, in this order:

- **A.** Store every mesh element array (`faces` for both `Face2D` and `Face3D`, plus `cells` and
  `nodes`) as a structure of arrays via StructArrays.jl, so hot kernels read only the fields they
  use. Existing kernel code is unchanged. This was "point 1" in the diagnosis and is worth the
  most.
- **B.** Run the threaded linear solve on one thread pool instead of Julia tasks plus OpenBLAS.
  This was "point 3".

"Point 2" (Int32 indices via `FOAM3D_mesh(...; integer_type=Int32)`) needs no code. It has been
measured: 11 to 16% faster at 6 and 8 cores, with identical residuals (see
[Int32 result](#int32-result)). **Use Int32 meshes as the baseline for everything below.** Change A takes the mesh
integer type as given, so it works with Int32 or Int64 meshes.

---

## 1. Why (the evidence, briefly)

Everything below is in
`3D_motorBike_RANS/data/scaling_diagnosis_2026-09-23/` (`results.txt`, `kernels.jl`, `stream.jl`).

- **The machine:** an i9-14900HX laptop. Memory bandwidth is 24.4 GB/s on 1 core and 39.5 GB/s on 8
  (STREAM triad, pinned P-cores). Anything that streams from DRAM tops out at about **1.6x**.
- **The fitted model:** `T(n) = C/n + B/s(n)`, where `C` is compute that divides across cores, `B`
  is DRAM-bound work, and `s(n)` is the measured bandwidth ratio. It fits every sweep to within
  about 5 to 7%.

  | | C (s) | B (s) |
  |---|---|---|
  | OpenFOAM PCG | 129 | 113 |
  | XCALibre MPI, before the `face_gDiff`/k-omega update | 184 | 113 |
  | XCALibre MPI, after | 64 | 112 |
  | XCALibre threads, after | 34 | 139 |

  The update removed compute, not bytes. **Lowering `B` is the only lever on the 8-core time.**
- **The kernels that set `B`**, timed on the real mesh (353,830 cells, 1,108,598 faces):

  | kernel | 1t ms | 8t ms | speedup |
  |---|---|---|---|
  | face interpolation reading `Face3D` (AoS) | 5.00 | 2.91 | 1.72x |
  | the same from separate Int32/Float64 arrays | 0.81 | 0.11 | 7.48x |
  | cell-based gather over `faces[cell_faces[fi]]` | 8.99 | 4.09 | 2.20x |
  | SpMV, CSR Int64 (the `xmul!` body) | 1.86 | 0.33 | 5.62x |
  | SpMV, CSR Int32 | 1.66 | 0.20 | 8.35x |

  The profiled `discretise!`, `grad!`, `interpolate!` and `turbulence!` all scale 2.1 to 2.2x,
  which matches the AoS rows.
- **Why AoS hurts even though LLVM drops unused fields:**
  - `Face3D{Float64,SVector{2,Int64},...}` is 128 B, so each face spans two cache lines. The
    `faces` array is 142 MB, about 4x the L3.
  - The fields a kernel uses are spread across both lines. `ownerCells` is at bytes 16 to 31 and
    `weight` at 120 to 127. `normal` straddles the line boundary, and `area` is at 104.
  - So almost every access pulls in both lines, all 128 B, from DRAM, and the cell-based loops do
    it twice per internal face.
  - OpenFOAM reads separate arrays (`owner`, `neighbour`, `weights`, `Sf`, `magSf`,
    `deltaCoeffs`) at about 24 B per face, and at 8 ranks those mostly stay in cache.
- **Threaded mode only:**
  - Krylov.jl sends `Vector{Float64}` `kdot`/`kaxpy!`/`kaxpby!`/`kscal!`/`kcopy!`/`knorm` to
    OpenBLAS (see `Krylov/src/krylov_utils.jl` in 0.10.10).
  - OpenBLAS `axpby` is single-threaded. In the profile, `axpby!` goes from 2.4 s at 1 thread to
    8.1 s at 8, and `kfill!` from 1.3 s to 2.0 s. About 13 s of solver vector work doesn't scale.
  - OpenBLAS runs its own spinning pool. Pinning it onto the Julia workers' cores made a run
    **47% slower** (21.6 s to 31.7 s over 100 iterations).
  - The unpinned interactive thread is *not* a factor: `-t 8`, `-t 8,0` and `-t 7,1` all took
    21.5 to 21.7 s.
  - This is the likely cause of the gap between threads (`B` = 139 s) and MPI (112 s). Inferred,
    not isolated.

## 2. Success criteria

Measured on motorBike, 500 iterations, balanced power profile, against the datasets in
`3D_motorBike_RANS/XCALibre/*.txt`:

1. **Correctness:** final residuals (U, p, k, omega) match the current branch to at least 8
   significant figures on 1 and 8 threads, 8 MPI ranks and the GPU. Forces (drag, lift) match to
   5 figures.
2. **No single-core regression** against the Int32 baseline: 1-thread time ≤ 162.9 s, and 1-rank
   ≤ 169.4 s.
3. **Change A:** 8-rank MPI time falls clearly below the Int32 baseline of 69.7 s (8-thread
   baseline: 78.0 s). The projection, if `B` halves again, is about 40 s. The GPU must be faster
   than 22.4 s: the StructArray layout measured 4.5x faster on the GPU interpolation kernel.
   Compile time for the first `run!` rises by no more than 10%.
4. **Change B:** threaded 8t comes within 5% of 8-rank MPI on the same revision. The Krylov
   vector-op share of the 8t profile falls from about 13 s to under 3 s.
5. **Memory footprint:** report `Base.summarysize(mesh)` and the per-array table (`footprint.jl`)
   before and after change A, for the Int32 motorBike mesh. Expected: `cell_nsign` down from
   8.5 MB to 2.1 MB, no duplicated geometry, and no integer array wider than its range needs.
6. `soa_test.jl`-style isolated checks on the real mesh object (not a synthetic copy) show face
   interpolation scaling at least 5x at 8 threads, and the cell gather at least 3.5x.
7. **Existing code keeps working:** `mesh.faces[i].area` and similar idioms in examples and user
   scripts still work unchanged.

## 3. Change A: store every mesh element array as a structure of arrays (StructArrays.jl)

### Decision
**Store all mesh element arrays (`faces`, for both `Face2D` and `Face3D`, plus `cells` and `nodes`)
as structures of arrays, using StructArrays.jl. Do *not* hand-write a parallel set of derived arrays,
and do *not* rewrite every consumer.**

- **What changes:** `StructArray(faces)` stores one array per field (`faces.weight::Vector{TF}`,
  `faces.ownerCells::Vector{SVector{2,TI}}`, and so on). `faces[i]` still returns a `Face3D`.
- **Why existing code gets faster unchanged:** when a kernel does `f = faces[i]; f.weight`, the
  compiler drops the loads for the fields it never uses, so the kernel reads only the columns it
  touches.
- **Consistency:** every element array uses the same storage, which gives the single, consistent
  philosophy for how the mesh is stored.

Measured on the Int32 mesh with **unchanged kernel code** (`data/scaling_diagnosis_2026-09-23/`,
`soa_test.jl` and `soa_gpu.jl`):

| kernel | AoS `Vector{Face3D}` 1t / 8t (ms) | StructArray, same code 1t / 8t (ms) | hand-written SoA 1t / 8t (ms) |
|---|---|---|---|
| face interpolation | 4.08 / 2.22 | 0.75 / **0.136** | 0.72 / 0.109 |
| cell gather (discretisation pattern) | 6.93 / 4.27 | 3.23 / **0.906** | n/a |
| face interpolation, GPU (RTX 4070, KA, wg 32) | 0.270 | **0.060** (identical results) | n/a |

StructArray gets within about 20% of hand-written arrays for a fraction of the edit, and it makes
the GPU 4.5x faster on the same kernel.

### Why not the alternatives
- **Derived arrays next to the structs** (this plan's earlier version):
  - It duplicates the geometry: about 93 MB extra on this mesh, on top of the structs.
  - It creates a staleness risk: two copies of the geometry to keep in sync.
  - It needs every hot kernel rewritten.
  - It gives the same bandwidth win as StructArray, which removes the duplication instead.
- **A hand-written SoA rewrite** (new `FaceArrays`/`CellArrays` types, every `faces[i].x`
  rewritten as `face_x[i]`):
  - Same runtime as StructArray.
  - It touches about 250 `faces[...]` / `cells[...]` sites, including readers, writers, boundary
    conditions, postprocessing, multiphase, LES and the film model.
  - It breaks the public `mesh.faces[i].area` idiom used in user scripts and examples.
  - It can't be checked kernel-by-kernel, because everything changes at once.

  No benefit for a much bigger change.

### Downsides, and how each is handled
- **A new dependency (StructArrays.jl).** It's small, widely used and maintained, and ships Adapt,
  GPUArraysCore and StaticArrays extensions. It's already in the depot (0.7.3).
- **Anything tied to the concrete `Vector{Face3D}` type.**
  - Grep shows only mesh *builders* use concrete `Vector{Cell…}` / `Vector{Face2D…}` / `Vector{Node…}`
    (the UNV2 types, BlockMesher2D, gmsh). They run before conversion, so they keep plain vectors.
  - Kernels already dispatch on `AbstractArray{Cell{…}}`, and `Mesh2`/`Mesh3` constrain `VF` as
    `AbstractArray{<:Face…}`, which StructArray satisfies.
  - Still grep for `pointer(`, `reinterpret(`, `unsafe_wrap` and `sizeof(faces)` on mesh arrays
    before merging.
- **Kernels that read most of a struct's fields** now stream from several arrays instead of one.
  The bytes are the same or fewer, but there are more concurrent memory streams, which hardware
  prefetchers cope with up to about 8 to 16. The only kernels near that are mesh-construction
  geometry routines, which aren't hot. Leave them.
- **Random gathers** (`faces[cell_faces[fi]]`) touch one cache line per field used instead of about
  two per struct. The discretisation and gradient gathers use 2 to 3 fields, so it's a net win, as
  measured.
- **Serialisation.**
  - JLD2 caches (`mesh_*.jld2`) are invalidated: regenerate them.
  - The `.xdm` writer dumps raw `isbits` records with `write(io, v)`, so it must `collect` a
    StructArray first (step 3). The reader is unaffected.
  - Folding `gDiff` into the face struct changes the record size, so bump `_XDM_FORMAT` (step 4b).
  - Delete `parts_*/` in any case, because the mesh type changes.
- **Construction-time element writes** (`faces[i] = Face3D(...)`) become one store per field. That's
  harmless if the conversion happens once, at the end of mesh construction, which is the design
  below.
- **More type parameters** in `Mesh2`/`Mesh3` signatures and error messages, and somewhat longer
  compile times. Measure compile time at the gate: `@time` on the first `run!` of the smoke case.
  Accept up to +10%.

### Scope: all element arrays, both mesh dimensions
- **`faces`:** `Face2D` and `Face3D` have identical layouts (128 B with Int64, 112 B with Int32).
  One code path covers `Mesh2` and `Mesh3`.
- **`cells`:** `Cell` is `centre`, `volume`, `nodes_range`, `faces_range` (64 B with Int64, 48 B with
  Int32). The hot reads are `(; volume, faces_range)` (16 sites), `volume` alone (20),
  `faces_range` alone (9) and `centre` (10).
- **`nodes`:** `Node` is `coords` plus `cells_range`. Not hot in this benchmark, but converted for
  consistency: it's the same one-line change.
- **`boundaries` stays a plain vector.** It holds names and ranges, a handful of entries, never
  indexed per face in a loop.
- **Nested `SVector` fields stay packed per field.** StructArrays keeps `ownerCells` as one
  `Vector{SVector{2,TI}}` (8 B per face with Int32) and `normal` as one `Vector{SVector{3,TF}}`. That
  is the right grain, so don't unwrap them to scalar columns.
- **The per-cell connectivity is already stored as separate arrays** (`cell_faces`,
  `cell_neighbours`, `cell_nsign`). `cell_nsign` only ever holds ±1, so it should be `Int8`: 1 B
  instead of 4 or 8, on a length-2×(internal faces) array read in every gather loop.

### Array width inventory: every stored array at its narrowest safe type
Measured on the Int32 motorBike mesh (serial `mesh_i32.jld2` and part `parts_i32_8/rank_0.xdm`),
using `footprint.jl` in `data/scaling_diagnosis_2026-09-23/`. The Int32 serial mesh object totals
about 230 MB, of which `faces` is 124 MB.

| array | length (serial) | type now (Int32 mesh) | values | hot? | action |
|---|---|---|---|---|---|
| `cell_nsign` | 2,116,940 | Int32, 8.5 MB | −1, 1 | **yes**, every gather | **`Int8`** (1 B): step 4 |
| `cell_faces`, `cell_neighbours` | 2,116,940 | Int32 | face and cell IDs | yes | keep `TI` |
| `diag_nz`, `face_nz` (per equation) | 353,830 / 2,116,940 | Int32 (follows `TI`) | nzval positions | yes | keep `TI` |
| matrix `rowptr`, `colval` | ~2.47M nnz | Int32 (follows `TI`) | | yes | keep `TI` |
| `ownerCells` (in `Face3D`) | 1,108,598 | `SVector{2,Int32}` | cell IDs | yes | keep `TI` |
| `faces_range`, `nodes_range` (in `Cell`/`Face3D`/`Node`) | | `UnitRange{Int32}` | | only `faces_range` | keep `TI`; the unused ranges cost nothing once stored per field |
| `cell_nodes`, `face_nodes`, `node_cells` | 2.9M / 4.5M / 2.9M | Int32 | node or cell IDs | no (setup, output) | keep `TI` |
| `boundary_cellsID`, `boundaries[].IDs_range` | 50,128 / 6 | Int32 | | boundary loops | keep `TI` |
| `partition.local_to_global`, `orig_cells`, `orig_faces` | per rank | Int32 (shares `VI`) | *global* IDs | no | **`Int64`**, for range not size: step 3 |
| `partition.owner` | per rank | Int32 | MPI rank (0 to 5 here) | no (setup) | keep: Int16 would save 0.1 MB per rank and cap the rank count |
| `procs[].faces`, `send_cells`, `recv_ghosts` | ~1–1.5k per patch | Int32 | local IDs | halo exchange | keep `TI` |
| AMG hierarchy index arrays (`Solve/AMG/*.jl`: `I`, `J`, `diag_index`, `marker`, aggregation maps, `AMGMatrixCSR` `rowptr`/`colval`) | ~nnz per level | **`Int` (Int64), hard-coded** | row, column and nnz positions | **yes, when `AMG()` is the solver** | **follow `TI`**: step 4c |
| periodic BC maps (`periodic.jl`: `face_map`, `faceAddress1/2`, `i`/`j`) | boundary-sized | **`Int64`, hard-coded** | face and cell IDs | periodic cases, per iteration | **follow `TI`**: step 4c, low priority |

**Floats are deliberately not narrowed.** Geometry, `gDiff`, fields and matrix values stay in the
mesh float type. Float32 geometry would halve those bytes too, but it changes results, so whether
it's accurate enough is a separate accuracy study, not a layout change. Note that the mesh types
are parameterised on the float type, but `FOAM3D_mesh` currently ignores its `float_type` keyword
and always builds Float64 (`src/FoamMesh/FoamMesh_5_build.jl`).

### Design
- **Convert in one place: the `Mesh2`/`Mesh3` outer constructors** in `src/Mesh/Mesh_0_types.jl`,
  where `face_gDiff` is already derived.
  - Wrap with `_soa(x) = x isa StructArray ? x : StructArray(x)` for `faces`, `cells` and `nodes`.
  - Every reader (FOAM, UNV2, UNV3, BlockMesher2D, gmsh, the distributed part loader) funnels
    through these constructors, so no reader changes.
  - The inner constructor keeps its role of guaranteeing `face_gDiff` is consistent. Build
    `face_gDiff` from the StructArray, or before wrapping; it's the same values either way.
- **Fold `face_gDiff` back into the face structs, but only once the StructArray conversion has
  landed and passed the gate.**
  - Under StructArrays a `gDiff` field *is* its own column, so `faces[fID].gDiff` in the
    `@inline` `scheme!` reads exactly the 8 B that `face_gDiff[fID]` reads today. Same speed, one
    storage philosophy, and one less `Mesh2`/`Mesh3` field and type parameter.
  - Done *before* StructArrays, it would add 8 B to every AoS face read and slow every face kernel.
    Hence the ordering.
  - **Staleness stays impossible:** add `gDiff` as the last field of `Face2D`/`Face3D`, plus an
    outer 8-argument constructor that computes it with `_gDiff` from the other fields. Every
    existing positional call site (about 20: the readers, `Mesh_1_functions.jl`,
    `Distribute_8_foam.jl`, the `Face3D(TI, TF)` dummies) then keeps working and gets a consistent
    `gDiff` for free.
  - Readers that rebuild faces after changing geometry (`Mesh_1_functions.jl` ~111/141/200) go
    through the same constructor, so `update_face_gDiff!` becomes unnecessary. Delete it, and the
    finiteness check moves to a check on `faces.gDiff`.
- Other *derived* per-face coefficients (for example `Sf = area*normal`) follow the same rule as
  `gDiff`: a field on the face struct, computed by the constructor. They are optional follow-ups,
  added only if a profile shows a kernel still bound on those fields.
- **`Adapt`:** StructArrays' Adapt extension moves each column to the device (verified:
  `adapt(CuArray, StructArray(faces))` gives a StructArray of `CuArray`s that KernelAbstractions
  kernels index directly).
  - `Adapt.@adapt_structure Mesh3` then works unchanged.
  - Check `adapt(CPU(), mesh)` (used as `mesh_temp` in `ScalarEquation`) returns a StructArray of
    `Vector`s, not a materialised `Vector{Face3D}`.
- **`update_face_gDiff!` and readers that fill geometry after construction:** `faces[i] = f` on a
  StructArray writes each column, so they keep working. Check they don't `copy`/`collect` the faces
  into a plain vector.
- **`cell_nsign` to `Int8`:** change it in the connectivity step of each reader (`connect_mesh` and
  friends) and in `extract_subdomain`. Its uses (`ns*(w - half)`, `flux*ns`) promote cleanly to the
  float type.

### Index width: signed Int32 locally, Int64 for global IDs (not UInt32)
- **UInt32 doesn't save any memory over Int32.** Both are 4 B, so the bandwidth gain measured in
  [Int32 result](#int32-result) is identical. The only difference is range: 4.29e9 against 2.15e9.
- **The range limit that bites first is per process, and it's about 300M cells.**
  - The widest counters in a process are the matrix `rowptr` (≈ 7 × cells here) and the face count
    (≈ 3 × cells).
  - So Int32 caps one *process* at about 300M cells, and UInt32 would raise that to about 600M.
  - A single process at 300M cells needs hundreds of GB of RAM (the serialised mesh object alone is
    0.3 GB for this 354k-cell case). Bigger meshes are run distributed, where each rank's *local* indices are tiny.
  - This is OpenFOAM's model too: `label` is signed 32-bit by default (`WM_LABEL_SIZE=32`), local to
    each processor directory.
- **UInt32 has real costs in Julia and in the libraries XCALibre calls:**
  - Unsigned subtraction wraps silently: `i - j`, `n - 1` on an empty range, or any `-1`/`0`
    sentinel.
  - Mixed signed/unsigned arithmetic promotes in surprising directions: `Int32 + UInt32 →
    UInt32`, `Int64 + UInt32 → Int64`. That breaks the type stability the `xcalibre-mesh-types`
    skill protects, and can silently widen indices back to 8 B.
  - The external index types are all signed: PETSc `PetscInt` (32- or 64-bit), METIS `idx_t`,
    cuSPARSE/rocSPARSE CSR indices. Every hand-off would need a checked conversion.
- **What large meshes actually need: decouple global IDs from local indices.**
  - `Partition.local_to_global`, `DistributedMesh.orig_cells` and `orig_faces` currently share the
    mesh's index type parameter, `VI` (`src/Distribute/Distribute_0_types.jl`). With an Int32 mesh,
    global IDs would overflow above 2^31 cells *globally*, long before any rank's local counts do.
  - Give the global-ID vectors their own `Int64` type, and keep local mesh indices `TI = Int32`.
  - `_petsclib` already chooses PETSc's `PetscInt` width from the *global* nnz, so it follows
    automatically.
  - This is a small, separate change. It is done in step 3 of change A, with a test that
    builds a partition whose global IDs exceed `typemax(Int32)` (synthetic `local_to_global`
    offsets are enough).
- **Default:** make `integer_type=Int32` the default for mesh readers, with a clear error if a
  per-process count exceeds `typemax(Int32)` (check `nfaces` and the matrix nnz at construction).
  Users can pass `Int64` explicitly for huge single-process meshes.

### Steps
Pass the gate in step 6 after each step.

1. **Add the dependency and convert in the constructors.**
   - Add `StructArrays` to `Project.toml` with a compat entry.
   - Add `_soa` wrapping for `faces`, `cells` and `nodes` in the `Mesh2`/`Mesh3` outer constructors.
     Nothing else changes.
   - **This single step should deliver most of change A.** Run the full gate and the 8-core sweep
     straight after it, before touching anything else.
2. **Fix what step 1 breaks.**
   - Run the test suite, the 2D and 3D examples, one MPI case and one GPU case.
   - Expect small fixes where code assumes a plain `Vector` (for example `similar(faces)` expecting
     a `Vector`, or `collect`).
   - Grep `pointer(`, `reinterpret(`, `unsafe_wrap`, `sizeof(` on mesh arrays.
3. **Distributed path.**
   - **`_xdm_write_array` (`src/Distribute/Distribute_6_format.jl`) needs a change.** It does
     `write(io, v)` with raw `isbits` records, which only works on a plain `Vector`. For a
     StructArray, write `collect(v)` (a transient copy of one rank's part), or write it in chunks.
     The reader returns a `Vector`, which the mesh constructor then wraps, so reading needs no
     change.
   - Confirm `extract_subdomain`, `_write_xdm` and `_read_part_file` round-trip a StructArray mesh
     bit for bit (write, read, compare every column).
   - Also do the global-ID change from
     [Index width](#index-width-signed-int32-locally-int64-for-global-ids-not-uint32) here: give `local_to_global`, `orig_cells` and `orig_faces` their own `Int64` type.
4. **`cell_nsign` to `Int8`**, in every reader's connectivity step and in `extract_subdomain`.
   - The `.xdm` format stores it with the other connectivity arrays and *reads it back as `TI`*
     (`_read_part_file`: `cell_faces, cell_neighbours, cell_nsign = (_xdm_read_array(io, TI, …)
     for _ ∈ 1:3)`). Write and read it as `Int8` and bump `_XDM_FORMAT`; this is one format bump
     together with step 4b.
   - Its type then needs its own mesh type parameter (it no longer shares `VI`). Check every kernel
     signature that constrains `cell_nsign` to `VI`.
   - Uses are `ns*(w - half)` and `flux*ns`, which promote to the float type. Grep for any
     *integer* arithmetic on `ns` (for example `ns*i`) that would now overflow or promote
     differently.
4b. **Fold `face_gDiff` into `Face2D`/`Face3D`** as described under Design.
   - Remove the `face_gDiff` field and its type parameter from `Mesh2`/`Mesh3`.
   - Change `Laplacian{Linear}` `scheme!` to read `face.gDiff`.
   - Bump `_XDM_FORMAT` from 3 to 4, since the face record grows by 8 B, and regenerate
     `parts_*/`.
   - Gate: residuals bit-identical (same `_gDiff` function, same values), and 8-thread time within
     noise of step 4.
4c. **Hard-coded `Int` index arrays follow `TI`.**
   - AMG (`src/Solve/AMG/`): build the hierarchy's index arrays (`I`, `J`, `diag_index`, `marker`,
     aggregation maps, `AMGMatrixCSR` `rowptr`/`colval`) with the finest matrix's index type
     instead of `Int`.
     - Not exercised by the CG benchmark, but hot for any `AMG()` user: its SpMV and smoothers see
       the same Int64 → Int32 gain SpMV showed (5.6x to 8.4x scaling).
     - Gate with the `run_psolver_study.sh` AMG case plus the AMG unit tests.
     - The matrix-free path already uses `Int32` in places (`8_AMG_matrix_free_refresh.jl`). Make
       it consistent rather than mixing.
   - Periodic BC maps (`src/Discretise/boundary_conditions/periodic.jl`): `face_map`,
     `faceAddress1/2` and the `i`/`j` connectivity use `Int64`/`Int`. Switch to `TI`. Low
     priority, since they're boundary-sized.
   - Finally, grep `src/` once more for `zeros(Int`, `Int64[`, `Int[`, `Vector{Int}` in anything
     stored on a mesh, equation or solver struct, and apply the same rule. Scratch arrays local
     to setup code can stay `Int`.
5. **Profile and decide on follow-ups.**
   - Re-profile 100 iterations at 8t, with main-thread wall time as in section 4 step 1.
   - Only if a kernel still scales below about 4x at 8t, consider precomputed per-face coefficients
     (like `face_gDiff`) for it, or a face-based (LDU-style) formulation.
   - The cell gather still reads each internal face twice: 3.57x at 8t, measured.
6. **Gate after each step:**
   - Residuals: 20-iteration smoke run at 1 and 8 threads, compared with stored baseline residuals
     (record them first). Layout changes shouldn't change a single bit, so expect exact equality on
     CPU at a fixed thread count.
   - 2D: one 2D example (for example the 2D BFS or cylinder case in `examples/`), 20 iterations,
     residuals against baseline. This is what exercises `Face2D` and `Mesh2`.
   - GPU: one 20-iteration run (`motorBike_gpu.jl distributed 20`).
   - MPI: one 8-rank 20-iteration run with freshly generated `parts_*`.
   - Compile time: `@time` on the first `run!` of the smoke case, within +10%.
   - Full 500-iteration timings at the end of steps 1 and 4.

### Pitfalls
- **Loads are dropped only when the struct is built and then partly used inside one inlined scope.**
  That's how every current kernel does it (`face = faces[fID]` or destructuring, then a few fields).
  Don't pass a whole `Face3D` into a `@noinline` function, and don't store it in a heap container
  inside a hot loop. That would force all columns to load.
- **Kernel closures capture whole structs.** `Multithread.jl` notes that AcceleratedKernels copies
  captured closures per thread. Pass `faces`/`cells` (now StructArrays) as kernel arguments, as
  `_discretise_scalar_model!` does, rather than capturing `mesh` in an `xcal_foreach` closure.
- **Type stability.** Follow the `xcalibre-mesh-types` skill. An Int32 mesh must not promote to
  Int64 in index arithmetic, which silently doubles the traffic again.
- **GPU block size.** Face loads are now coalesced, so the `workgroup=32` optimum measured on the
  AoS layout may move. Re-sweep once at the end (`./run_gpu.sh 32 64 128 256`).

## 4. Change B: one thread pool for the threaded linear solve

### Design
- **A thin CPU-only vector type**, `XVector{T} <: DenseVector{T}`, wrapping a `Vector{T}` plus the
  static row partition (the same `RangeIterator` split `xmul!` uses). Put it in
  `src/Multithread/` next to `spmvm.jl`.
- **Why a wrapper:** Krylov.jl 0.10.10 dispatches `Vector{<:BlasFloat}` to BLAS and every other
  `AbstractVector` to generic methods. Defining `kdot`, `kdotr`, `knorm`, `kscal!`, `kcopy!`,
  `kaxpy!`, `kaxpby!` and `kfill!` for `XVector` routes all solver vector work onto XCALibre's own
  threaded loops, without touching methods on Base's `Vector`.
- **Workspaces:** they are built in `src/Solve/Solve_1_Krylov_solvers.jl` via
  `_workspace(::Cg, b) = CgWorkspace(KrylovConstructor(b))`. If `b` (and the solution `x`) are
  `XVector`s on the CPU backend, the workspace vectors come out as `XVector` too.
  `KrylovConstructor` uses `similar`, so define `similar(::XVector)` to keep the partition.
- **Methods to add:**
  - `mul!(y::XVector, A::SparseXCSR, x::XVector, α, β)`: the existing `xmul!` body over the same
    partition.
  - Jacobi `ldiv!`/`mul!`: the `_diagonal_mul!` KA kernel in
    `Solve/Preconditioners/preconditioners_2_functions.jl` becomes a partitioned loop.
- **Reductions:** per-thread partial sums in a preallocated `Vector{T}` (one slot per chunk,
  padded to 64 B to avoid false sharing), then a serial sum of 8 numbers. Deterministic given a
  fixed partition, and therefore reproducible.
- **Fuse where Krylov allows it.** Krylov.jl's algorithms call the primitives separately, so true
  fusion (Jacobi apply with `r·z`, the two CG axpys) needs either:
  - (a) a thin CG and BiCGStab of our own on `XVector`, or
  - (b) accepting the unfused version.

  Do (b) first and measure. Only if vector ops are still above ~3 s at 8t, write (a) for CG, the
  pressure solver, since it's the bulk of the iterations.
- **Remove OpenBLAS from the path.**
  - Once no solver vector op goes through BLAS, have `activate_multithread` default back to
    `BLAS.set_num_threads(1)`, so the idle pool can't compete.
  - Keep the keyword for users who call BLAS themselves.
  - Update the docstring, which currently argues the opposite for the reason this change removes.
- **Scope:** CPU backend only. GPU paths keep `CuVector`, and MPI keeps PETSc. The field storage
  (`values`) stays `Vector`. Wrap at the `solve_system!` boundary if making `values` an `XVector`
  ripples too far, which is likely. The wrap must not copy: `XVector(values, partition)` just holds
  a reference.

### Steps
1. **Record baselines.** Profile 100 iterations at 8t and store the Krylov vector-op totals
   (`kaxpby!`, `kaxpy!`, `kdot`, `kfill!`, `mulorldiv!`).
   - **Measure on the main thread only.** A flat per-thread profile charges spawned-task work to
     worker threads and the join to `mul!`, which is how the old README went wrong.
2. **`XVector` basics:** add the type and partition, `similar`, `size`, `getindex`, `setindex!`,
   and `unsafe_convert` so generic fallbacks still work. Unit-test each `k*` primitive against
   `Vector` results at 1 and 8 threads.
3. **Wire it into `solve_system!`** (`src/Solve/Solve_1_api.jl`) for the CPU backend: wrap `b` and
   `values`, and build workspaces from the wrapped `b`. Check that
   `krylov_solve!(..., M=P, ldiv=...)` still resolves the preconditioner method.
4. **Preconditioner:** Jacobi apply on the partition. DILU, if used on the CPU, stays serial for
   now (out of scope).
5. **BLAS default:** set `activate_multithread` back to 1 BLAS thread, and update the docstring
   and `3D_motorBike_RANS/README.md` (the BLAS study section).
6. **Clean up the dead code** in `spmvm.jl` noted in the README:
   - `xmul!(A, x)` references an undefined `y`.
   - `xmul(y, A, x)` throws away its `y`.
   - `Base.:*(A::SparseMatrixCSR, x::SparseXCSR)` calls a 2-argument `xmul` that doesn't exist.

   Delete them or make them correct.
7. **Gate:**
   - Residuals match to at least 8 significant figures. The reduction order changes, so
     bit-identity isn't expected; tighter than 8 figures is a bonus.
   - The 8t vector-op total drops.
   - 1t is not slower.
   - Threads at 8t come within 5% of 8 MPI ranks (success criterion 4).

### Pitfalls
- **`@spawn` per call is fine.** The isolated `xmul!` scaled 5.6x with spawn-and-join included,
  so a persistent thread team isn't needed. Don't build one unless measured.
- **Use the same partition for SpMV and vector ops.** That's the point: the thread that wrote
  `y[r]` in SpMV reads it in the next `axpy`, so the data is still in its L2.
- **`AutoTune` and `static=true`** already give each KA thread a fixed contiguous cell range. Make
  `XVector`'s partition identical, so assembly, solve and field update touch the same chunks per
  thread.

## 5. Benchmark protocol (for both changes)

- **Environment:** the benchmark `env_distributed` pulls the branch from GitHub. For local work,
  `Pkg.develop(path="~/Julia/XCALibre.jl")` in a *copy* of that environment, so the recorded
  environments stay reproducible.
- **Quick check,** about 5 minutes: 100 iterations at 1 and 8 threads, plus 8 MPI ranks.
  - Use scratchpad copies of the drivers that write to a separate results file. The stock drivers
    append to the datasets.
  - Never use `run_openfoam.sh` for a single process count: it deletes
    `openfoam_motorBike.txt` first.
- **Full sweep** at the end of each change: `run_multithreaded.sh`, `run_distributed.sh`,
  `run_gpu.sh`, and archive the previous datasets under `data/`. **Delete `mesh_*.jld2` and
  `parts_*/` first**, because the mesh type changes.
- **Refit `C` and `B`** with the triad ratios `s = [1, 1.172, 1.586, 1.619]` for n = 1, 2, 6, 8.
  A successful change A shows up as a lower `B` with `C` roughly unchanged.
- **OpenFOAM** doesn't need re-running. The 8-rank rerun (80.33 s against 81.66 s on record) showed
  the stored data is sound.

## 6. Out of scope / follow-ups

- Face-based (LDU-style) assembly instead of the cell-based gather: another ~2x on face traffic.
  This is the main candidate if the cell gather is still the slowest kernel after change A (3.57x
  at 8t with StructArray).
- Extra precomputed per-face coefficients (for example `Sf = area*normal`), following the
  `face_gDiff` pattern. Add one only if a profile after change A shows a kernel bound on those
  fields.
- Cell renumbering (RCM) for locality of the thread chunks.
- XCALibre AMG hierarchy reuse (`gamg_gaps.md`).
- Rewriting the README section "What the floor is", which these measurements contradict. Do it
  when change A lands, with the new numbers.

## Int32 result

Measured 2026-09-23 with `FOAM3D_mesh(...; integer_type=Int32)`, 500 iterations, on the same
machine and profile. The Int64 column is the recorded sweep. The matrix is confirmed as
`SparseXCSR{1, Float64, Int32, 2}`, and the MPI part files keep Int32. Residuals match the Int64
runs to 10+ significant figures.

| | cores | Int32 (s) | Int64 (s) | change |
|---|---|---|---|---|
| threads | 1 | 162.90 | 176.61 | -7.8% |
| threads | 2 | 114.32 | 127.44 | -10.3% |
| threads | 6 | 83.77 | 96.93 | -13.6% |
| threads | 8 | 77.95 | 92.86 | -16.1% |
| MPI | 1 | 169.41 | 178.03 | -4.8% |
| MPI | 2 | 112.73 | 120.91 | -6.8% |
| MPI | 6 | 73.80 | 84.52 | -12.7% |
| MPI | 8 | **69.66** | 78.56 | -11.3% |
| OpenFOAM | 8 | 80.33 (rerun) | 81.66 | |

- **Refit over n = 1, 6, 8:**
  - Threads: `B` 144.5 → 119.3 s (-17%).
  - MPI: `B` 116.1 → 98.3 s (-15%). MPI's `B` is now below OpenFOAM's 107.5 s.
- **Scaling from 1 to 8 cores:** threads 1.90x → 2.09x, MPI 2.27x → 2.43x.
- **Caveat:** the 1-core gains (5 to 8%) are near the ±5% repeat noise. The 6- and 8-core gains
  (11 to 16%) are well clear of it.
- **Conclusion:** make Int32 the benchmark default now. Consider making it the default in
  `FOAM3D_mesh` and the other readers for meshes under 2^31 cells and faces. Changes A and B build
  on this baseline, so rerun the Int64 → Int32 control before comparing against them.
