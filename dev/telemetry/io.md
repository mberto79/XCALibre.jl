# Decomposed output and restart (P1-M24)

Machine: this laptop, `dev/petscenv_stock`, Julia 1.13, CPU. Raw logs: `~/.cache/xcal_m24/` (not tracked).

## S1 binary writer

- Write time, BFS 4 mm tet (1.32M cells) n=2, mesh plus U and p, slowest rank, second call (`dev/scripts/io_bench.jl`, ASCII writer loaded from 590b30e3): binary 0.22 s and 66.3 MB on rank 0 including `phi`; ASCII 2.0 s and 107.9 MB without `phi`.
- OpenFOAM 12 reads the binary case: `checkMesh -case processor0` Mesh OK (34456 cells), `foamToVTK -case processor1` completes, `reconstructPar -latestTime` reconstructs U, p and phi; reconstructed U and p match the serial run to 1.1e-9 and 6.9e-10 relative (10 mm BFS decomposePar scotch n=2, 20 iterations). ParaView is not installed here; its OpenFOAM reader was not run.
- The FOAM mesh reader now parses binary points, faceCompactList faces, owner, neighbour and label lists, so `FOAMCase` reads what the writer writes (`test_io.jl` n=1,2).

## S2 restart

- `test/distributed/test_restart.jl`, 3D box (125 cells), 100 iterations with a write at 50, then a fresh model resumed from 50: SIMPLE laminar, PISO laminar (dt 0.1, restart by time 5.0) and SIMPLE k-omega SST under Jacobi; p residuals of iterations 51-100 and final U, p (and nut) identical, max relative difference 0.0 at n=2 (printed), within 1e-10 at n=1,2,4 (asserted); the resumed run's first 50 residual slots keep their initial fill, so it did resume.
- Serial SIMPLE, PISO and k-omega cases unchanged with the hooks in place (4 serial test cases, 20/20); gate 10/10.
