# Paste this to start the next session

Continue optimising `FOAM3D_mesh` (OpenFOAM polyMesh reader, `src/FoamMesh/`). The reader+connect
were already optimised and committed (CRMHL 2.58M cells: 31.8s/21.7GB -> 11.3s/9.1GB = 2.82x,
−58% mem, bit-identical). Read `dev/foam_bench/HANDOVER.md` first — it has the full status,
workflow, scripts, mesh paths, and recorded v0+final baselines (in `dev/foam_bench/baselines/`,
DO NOT recapture v0).

TASK: attempt the remaining lever — `read` is now the largest stage (5.1s/2.6GB). Its allocator is
`read_faces` building `Vector{Vector{TI}}` (one `TI[]` per face × 5.97M) stored into
`Face.nodesID::Vector{TI}` per FoamFace. Collapse to flat/CSR face-node storage. This is
ARCHITECTURAL: it touches the `Face` struct in `FoamMesh_0_types.jl`, `assign_faces!`, and every
reader of `face.nodesID` (connect_cell_nodes, connect_face_nodes, connect_cell_faces, generate).
Map all `face.nodesID` / `face.owner` / `face.neighbour` uses before changing the struct.

CONSTRAINTS (same as last round):
- Output MUST stay bit-identical to the original. Verify, don't assume.
- Plan first, then delegate implementation to a cheaper Sonnet agent. You orchestrate + review only.
  Keep token use low.
- Parser must stay robust to line breaks inside entries (already passes; keep it).

WORKFLOW (proven):
1. PLAN the allocation-free transform; ensure it is provably order-preserving.
2. DELEGATE to Sonnet with an exact spec (algorithm + invariants). Review the diff yourself.
3. VERIFY bit-identical cheap-first using the oracle:
   - small hex meshes, then F1 (triangular), then CRMHL (polyhedral)
   - `julia --project dev/foam_bench/foam_validate.jl hashall <small dirs>` then `diff` vs
     `dev/foam_bench/baselines/v0_small.log` (grep 'MESH|FIELD'); same for F1 and CRMHL hashes.
   - Identical diff = bit-identical to original. (test_mesh_conversion.jl only checks counts+types.)
4. BENCH with `... stages <CRMHL dir>` and `... bench <CRMHL dir>`; report allocs(GB)+GC as the
   swap-independent metric (this box: 14GB RAM, time can be swap-distorted; allocs cannot).
5. Run all Julia in background (compile > 2min). Update the memory file
   `project_foam_reader_optimisation` and `dev/foam_bench/HANDOVER.md` with results.

Do not commit unless asked.
