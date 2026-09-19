# Preprocessing: part load time and edge-cut (P1-M23)

Machine: this laptop, `dev/petscenv_stock`, Julia 1.13, CPU. Load: `dev/scripts/part_load.jl` (all parts in one process, second pass, page cache warm). Raw logs: `~/.cache/xcal_m23/` (not tracked).

## S1 load time, `.jls` (format 2, HEAD 09cb77a4) against `.xdm` (format 3)

- BFS 10 mm n=8 `.jls`: total 0.017 s, 39.6 MB.
- BFS 4 mm n=8 `.jls`: total 0.345 s (max 0.050 s per part), 738.3 MB; partitioning 1.32M cells took 3.5 s at 2350 MB peak.
- BFS 10 mm n=8 `.xdm`: total 0.013 s, 39.6 MB.
- BFS 4 mm n=8 `.xdm`: total 0.324 s (max 0.049 s per part), 738.3 MB; partitioning 3.4 s at 2354 MB peak.
- Verdict: load time and size unchanged within noise (both formats are raw array reads); the gain is survival across upgrades and a checkable header, not speed. Round trip exact on every mesh, partition and patch array (`test_offline.jl` n=1,2,3, 2D BFS and 3D box, serial and partitioned kinds).
