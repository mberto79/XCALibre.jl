# Preprocessing: part load time and edge-cut (P1-M23)

Machine: this laptop, `dev/petscenv_stock`, Julia 1.13, CPU. Load: `dev/scripts/part_load.jl` (all parts in one process, second pass, page cache warm). Raw logs: `~/.cache/xcal_m23/` (not tracked).

## S1 load time, `.jls` (format 2, HEAD 09cb77a4) against `.xdm` (format 3)

- BFS 10 mm n=8 `.jls`: total 0.017 s, 39.6 MB.
- BFS 4 mm n=8 `.jls`: total 0.345 s (max 0.050 s per part), 738.3 MB; partitioning 1.32M cells took 3.5 s at 2350 MB peak.
- BFS 10 mm n=8 `.xdm`: total 0.013 s, 39.6 MB.
- BFS 4 mm n=8 `.xdm`: total 0.324 s (max 0.049 s per part), 738.3 MB; partitioning 3.4 s at 2354 MB peak.
- Verdict: load time and size unchanged within noise (both formats are raw array reads); the gain is survival across upgrades and a checkable header, not speed. Round trip exact on every mesh, partition and patch array (`test_offline.jl` n=1,2,3, 2D BFS and 3D box, serial and partitioned kinds).

## S2 `distribute(FOAMCase)` on `decomposePar` output

Case: BFS 10 mm tet (68243 cells) converted with OpenFOAM 12 `ideasUnvToFoam`, laminar SIMPLE 20 iterations, Jacobi, inner rtol 1e-8; serial `FOAM3D_mesh` against the gathered distributed fields (`dev/scripts/foam_decomposed.jl`, cases in `~/.cache/xcal_m23/of_bfs10*`). Relative max difference, all four components pass 1e-5; `check_ghosts` 0 at every n. `load_s` includes compilation.

- OF12 scotch n=2: Ux 1.06e-9, Uy 1.73e-9, Uz 2.21e-9, p 6.90e-10; load 2.38 s.
- OF12 scotch n=4: same to 3 digits; load 3.04 s.
- OF12 scotch n=8: same to 3 digits; load 4.29 s.
- v2512 scotch n=4: same to 3 digits; load 3.27 s. Both versions write `cellProcAddressing` in `processor<r>/constant/polyMesh`.
- Results written at n=2 into the decomposed case: OF12 `reconstructPar -latestTime` completes once fields carry `dimensions` (added to both writers); reconstructed U matches serial to 1.08e-9 (Ux), 1.82e-9 (Uy).
- The 2e-9 floor is identical at every n, so it is the serial path (Krylov.jl, one geometry pass) against the distributed one (PETSc, per-processor geometry), not the decomposition.
