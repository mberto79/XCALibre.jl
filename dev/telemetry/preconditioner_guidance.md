# preconditioner guidance measurements (P1-M13)

Case: BFS tet 5 mm (499,503 cells), laminar SIMPLE, U bcgs+jacobi, p cg+<pc> rtol 0.01 (relative to initial residual, D53), 30 outer iterations, `dev/scripts/equal_thermal.sh`, stock PETSc_jll Float64/Int32.

S1, n=2, interleaved, s/iter and final p residual. BASE = 58a95d13; NEW = MAT_SPD on pure-Laplacian matrices + BoomerAMG P_max=4:
- gamg: BASE 0.4627, 0.4738 (p 7.8523e-5); NEW 0.4788, 0.4809 (p 7.8523e-5, bitwise identical)
- boomeramg: BASE 0.6469, 0.6576 (p 6.1934e-5); NEW 0.6503, 0.6272 (p 6.2482e-5)

S4 smoke, n=2, unpinned, 30 iterations, s/iter and final p residual:
- IC0GPU (bjacobi+icc): 0.4935, p 9.9443e-5
- ILU0GPU (bjacobi+ilu): 0.5178, p 9.9443e-5
- Jacobi: 0.4692, p 1.2410e-4
