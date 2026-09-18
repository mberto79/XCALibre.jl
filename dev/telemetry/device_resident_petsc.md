# device-resident PETSc solve (P1-M10)

Case: `dev/scripts/scaling_probe.jl worker`, BFS tet 5 mm (499,503 cells), laminar SIMPLE, U bcgs+jacobi, p cg+jacobi, 30 iterations, per_iter = (t30 - t3)/27. Clock not pinned. Before = commit 1093742b (host staging of A, b, x every solve); after = COO assembly from the nzval pointer plus device-to-device vector copies.

GPU (RTX 4070 Laptop, CUDA PETSc 3.24, n=1), s/iter:
- before: 0.0912, 0.1026
- after: 0.0746, 0.0720
- final residuals identical to 13 significant figures (p 1.10363280327421e-3).

CPU (n=2, --bind-to core), s/iter, interleaved before/after:
- before: 0.4504, 0.4565, 0.4491, 0.4679, 0.4682 (median 0.4565)
- after: 0.4918, 0.4447, 0.4401, 0.4385, 0.5151 (median 0.4447)
- final residuals bitwise identical before and after.
