# FOAM3D_mesh optimisation — handover

Tooling + recorded baselines for optimising `FOAM3D_mesh` (OpenFOAM polyMesh → XCALibre Mesh3).
Source: `src/FoamMesh/`. Pipeline: `read_FOAM3D` → `connect_mesh` → `generate_mesh` → `compute_geometry!` → `convert_mesh_float`.

## Status (done)
v0 = pre-optimisation original. Final = committed optimised version.
CLEAN measurement (this box: 14GB RAM, 4GB swap; CRMHL peaks ~12GB, swap stays minimal).

CRMHL 2.58M cells / 5.97M faces:
- v0:    31.8 s / 21.7 GB
- final: 11.3 s /  9.1 GB   → **2.82×, −58% mem**, output bit-identical.

Stage times v0 → final: read 13.2→5.1s | connect 12.9→4.0s | generate 4.5→2.0s | geometry 0.7→0.6s.

What was changed (all bit-identical, see Verification):
- `FoamMesh_1_read.jl`: read_faces/read_points/read_neighbour → single-pass byte scanners (`_next_uint`,`_skip_lines`). Robust to line breaks inside entries.
- `FoamMesh_2_connect.jl`: connect_cell_faces, connect_cell_nodes (raw-CSR + per-cell stamp-dedup), connect_node_cells → CSR (kill vector-of-vectors + push!/append! + intersect. + reduce(vcat)).
- `FoamMesh_3_generate.jl`: `SVector{2}(TI[owner,neighbour])` → `SVector(owner,neighbour)` (removes a per-face heap array; cut generate 3.9→2.0s).

## NEXT-SESSION TASK (the remaining lever)
`read` is now the largest stage (5.1s / 2.6GB). Its main allocator is `read_faces` building
`Vector{Vector{TI}}` (one small `TI[]` per face × 5.97M) and `assign_faces!` copying them into
`Face.nodesID::Vector{TI}` per FoamFace. Collapsing this to flat/CSR storage is **architectural**:
it changes the `Face` struct (`FoamMesh_0_types.jl`), `assign_faces!`, and every reader of
`face.nodesID` (notably `connect_cell_nodes`, `connect_face_nodes`, `connect_cell_faces` use
`face.owner/neighbour/nodesID`). Plan + delegate carefully; full re-validation required.
Target: push read down meaningfully (toward ~3×+ overall).

## Workflow to follow (proven this round)
1. PLAN: identify hotspot via `stages`; design an allocation-free transform that is provably
   order-preserving (CSR count→offset→cursor-fill; dedup via per-cell-contiguous stamp array —
   NEVER single-pass face-order dedup, the c1→c2→c1 stamp overwrite duplicates).
2. CAPTURE v0 (already recorded here — DO NOT rerun): `git stash` the readers → original tree,
   run oracle, `git stash pop`. Baselines in `baselines/v0_*.log`.
3. DELEGATE implementation to a Sonnet agent with an exact spec (algorithm + invariants, e.g.
   "preserve boundary double-write: neighbour==owner on bfaces"). You orchestrate + review the diff.
4. VERIFY bit-identical, cheap-first: small hex meshes → F1 (triangular) → CRMHL (polyhedral).
   A dedup-order bug shows on cavity in seconds; don't burn CRMHL to find it.
5. Report allocs(GB)+GC as the swap-independent metric (time can be swap-distorted on 14GB box).

## Scripts
`foam_validate.jl` — oracle. Bit-exact via raw-bit hashing (reinterpret floats → catches −0.0/NaN)
of all 23 mesh fields incl cell_nodes/node_cells/ranges.
- `julia --project dev/foam_bench/foam_validate.jl hashall <dir1> <dir2> ...`  (scale 1, one process)
- `julia --project dev/foam_bench/foam_validate.jl hashes  <dir> [scale]`
- `julia --project dev/foam_bench/foam_validate.jl bench   <dir> [scale]`   (time/alloc/GC)
- `julia --project dev/foam_bench/foam_validate.jl stages  <dir> [scale]`   (per-stage time/alloc/GC)

`foam_diag.jl` — (a) line-break robustness test (synthetic split-entry faces/points; PASS),
(b) `@code_warntype` on generate_faces (type-stability check). `julia --project dev/foam_bench/foam_diag.jl`

Always run Julia in background (compile+GPU warmup > 2min). Heavy meshes can swap on 14GB.

## Meshes
- small hex: test/grids/OF_cavity_hex/polyMesh, examples/0_GRIDS/{OF_pitzDaily,OF_bump2d,OF_T3A,OF_squareBend}/polyMesh
- F1 (1.68M cells, triangular faces): F1-fetchCFD_Minimal/mesh/polyMesh
- CRMHL (2.58M cells, polyhedral): /home/humberto/casesXCALibre/CRMHL_v1/CRMHL_Wingbody_1v.pw.OF/polyMesh

## Verify a NEW change against v0 (no need to recapture v0)
    julia --project dev/foam_bench/foam_validate.jl hashall <small meshes> > /tmp/new_small.log
    diff <(grep -E 'MESH|FIELD' dev/foam_bench/baselines/v0_small.log) <(grep -E 'MESH|FIELD' /tmp/new_small.log)
    # then F1 + CRMHL hashes vs baselines/v0_f1.log, baselines/v0_crmhl.log
Identical diff = bit-identical to original. `test_mesh_conversion.jl` only checks counts+types (insufficient).
