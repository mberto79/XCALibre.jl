# Extraction cost (P1-M20)

Machine: this laptop, clock unpinned, `dev/petscenv_stock`, Julia 1.13. Script: session scratchpad `extraction_baseline.jl` (reads the mesh, times `partition_mesh` cold, then `partition_cells`, `decompose` and `partition_mesh` warm, and prints a SHA-256 over every part's serialised bytes with the header line excluded). Peak RSS is `/usr/bin/time %M` for the whole process, mesh read included. The 4 mm BFS mesh named in the plan no longer exists on this machine, so the periodic cascade stands in as the third configuration.

| config | code | partition_cells warm s | decompose warm s | partition_mesh warm s | peak RSS MB | parts digest |
|---|---|---|---|---|---|---|
| BFS 5 mm tet, 499,503 cells, P=8 | before (3818c35b) | 0.302 | 0.82 | 1.00 | 1442 | 69d41fc8 |
| BFS 5 mm tet, P=8 | after | 0.192 | 0.58 | 0.72 | 1460 | 69d41fc8 |
| BFS 5 mm tet, P=64 | before | 0.383 | 1.77 | 1.89 | 1402 | f1d79168 |
| BFS 5 mm tet, P=64 | after | 0.328 | 0.97 | 1.08 | 1585 | f1d79168 |
| cascade periodic 2.5 mm, 126,520 cells, P=8, top/bottom | before | 0.073 | 0.34 | 0.91 | 938 | 9417736c |
| cascade periodic, P=8 | after | 0.071 | 0.23 | 0.30 | 938 | 9417736c |

- Parts are bitwise identical in all three configurations (full digests in the scratchpad `extraction_before2.txt` / `extraction_after.txt`).
- The P factor at this size was about 1 s of the 1.77 s: `decompose` at P=64 falls 1.8x, not the 5x the plan asked for, because what remains is O(local) output construction (cell, face and node copies through `push!`), which the rewrite does not change. The rewrite matters where P·N is large; at 5M cells and 256 parts the removed term would have been roughly 20 s.
- Peak RSS is within run-to-run noise except P=64, where the per-part face buckets add about 180 MB.
