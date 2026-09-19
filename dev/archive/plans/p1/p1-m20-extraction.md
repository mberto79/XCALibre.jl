# P1-M20 - O(N+P) extraction (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R11. Governing decisions: D72.

## Problem, quantified

`extract_subdomain` is O(N) per part whatever the part size: the owned scan (`Distribute_1_partition.jl:84`), the interior-face scan, `part_counts`/`pos`/`ctr` rebuilt per part (`:191-198`), three `Dict{Int,TI}` maps (`:95`, `:111`, `:115`), and a processor-patch loop over every interior face per neighbour (`:207-216`). `partition_cells` counts with `count(==(r), parts)` per part. `build_dual_graph` grows Int64 triplets by `push!` and then calls `sparse`. Total O(P·N) time and O(N) churn per part on rank 0; rank 0 holds the global mesh at 1.6 KB/cell (`dev/gotchas.md`).

## Approach

One pass over cells and faces builds every per-part structure at once, then each part is extracted from its own buckets in O(local). Inverse maps are `Vector{TI}` of length N, zero by default, set for the part's local entities and reset by iterating the same local lists (never `fill!`). The dual graph is the mesh's own adjacency: `cell.faces_range` is a row pointer and `cell_neighbours` the column list, so `SparseMatrixCSC(n, n, colptr, rowval, ones)` is built without triplets.

## Configuration space

mesh {BFS 5 mm 499,503; BFS 4 mm 1,320,368; cascade periodic} x P {8, 64} x periodic {no, yes}; the verdict compares serialized parts byte for byte, and `test_partition.jl`/`test_periodic.jl` stay green.

## Steps

- [x] **P1-M20-S1** LANDED (D86): identical parts on all three meshes (same Metis result); `partition_cells` 0.302 to 0.192 s at 5 mm P=8 - `build_dual_graph` from `faces_range`/`cell_neighbours` directly; `partition_cells` counts in one pass; the periodic contraction keeps its union-find but builds the super-graph the same way - mechanism: the adjacency already exists in CSR form - cost: none - verdict: identical `parts` vector from Metis (same graph, same seed) on all three meshes; time recorded.
- [x] **P1-M20-S2** LANDED (D86): `_PartIndex` shared by `decompose` and `distribute`; parts bitwise identical; `dev/telemetry/extraction_cost.md` (4 mm mesh no longer on this machine, cascade periodic used instead) - a `_PartIndex` built once per decomposition: cells bucketed per part (CSR by part), `part_counts`, `offs`, global `pos`, and per part the ghost set found by scanning only that part's owned cells' neighbours; interior faces bucketed per part in one face pass (a face touching two parts lands in both) - mechanism: every per-part structure is a bucket of a single global pass - cost: O(N + F) once plus O(local) per part - verdict: `extract_subdomain(mesh, parts, r)` output bitwise identical to today's for every r (compare `serialize` bytes); `partition_mesh` wall time and peak RSS (`/usr/bin/time -f "%e %M"`) recorded before and after at 5 mm P=8,64 and 4 mm P=8 in `dev/telemetry/extraction_cost.md`.
- [x] **P1-M20-S3** LANDED (D86): vector maps and per-neighbour lists from one pass over the part's faces; `decompose` at P=64 fell 1.8x, not 5x, because the P·N term was 1 s of 1.77 s at 500k cells and the O(local) `push!` construction remains - the three `Dict`s become reset-on-exit `Vector{TI}` maps owned by `_PartIndex`; the processor-patch loop iterates the part's own interior-face bucket - mechanism: as S2 - cost: 3N words of TI once - verdict: as S2; the P=64 time must fall by at least 5x on 5 mm (the P factor) or the step records why.

## Exit criterion

Parts bitwise identical; `dev/telemetry/extraction_cost.md` holds before/after time and RSS for the three configurations; suite green.

## Open questions

- SETTLED: with rows sorted and deduplicated the CSR graph gives the same Metis partition as the triplet build (digests identical, D86).
