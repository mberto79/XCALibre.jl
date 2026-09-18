# preconditioner guidance measurements (P1-M13)

Case: BFS tet 5 mm (499,503 cells), laminar SIMPLE, U bcgs+jacobi, p cg+<pc> rtol 0.01 (relative to initial residual, D53), 30 outer iterations, `dev/scripts/equal_thermal.sh`, stock PETSc_jll Float64/Int32.

S1, n=2, interleaved, s/iter and final p residual. BASE = 58a95d13; NEW = MAT_SPD on pure-Laplacian matrices + BoomerAMG P_max=4:
- gamg: BASE 0.4627, 0.4738 (p 7.8523e-5); NEW 0.4788, 0.4809 (p 7.8523e-5, bitwise identical)
- boomeramg: BASE 0.6469, 0.6576 (p 6.1934e-5); NEW 0.6503, 0.6272 (p 6.2482e-5)

S4 smoke, n=2, unpinned, 30 iterations, s/iter and final p residual:
- IC0GPU (bjacobi+icc): 0.4935, p 9.9443e-5
- ILU0GPU (bjacobi+ilu): 0.5178, p 9.9443e-5
- Jacobi: 0.4692, p 1.2410e-4

S2, CPU, equal thermal, one run each, s/iter and final p residual (5 mm = 499,503 cells; 4 mm = 1,320,368 cells):
- 5 mm n=2: jacobi 0.6373 (p 1.2410e-4); gamg 0.4651 (p 7.8523e-5); boomeramg 0.6347 (p 6.2482e-5)
- 5 mm n=8: jacobi 0.2114 (p 1.2410e-4); gamg 0.2079 (p 6.8064e-5); boomeramg 0.2386 (p 7.7266e-5)
- 4 mm n=2: jacobi 2.3785 (p 9.6255e-5); gamg 1.4560 (p 1.2094e-4); boomeramg 2.0136 (p 3.9898e-5)
- 4 mm n=8: jacobi 0.9066 (p 9.6255e-5); gamg and boomeramg stopped by memguard at MemAvailable ~1.45 GB (an unguarded attempt OOM-killed the desktop)
- 5 mm to 4 mm at n=2 (cells x2.64): jacobi time x3.73, gamg x3.13, boomeramg x3.17

S5, 5 mm n=1, local CUDA PETSc 3.24 with CUDA hypre, RTX 4070 Laptop vs same env on CPU, s/iter and final p residual:
- jacobi: GPU 0.0748 (p 1.2409749477951e-4), CPU 1.1601 (p 1.2409749477949e-4)
- gamg: GPU 0.0588 (p 2.537667222860e-4), CPU 0.6953 (p 2.537667222849e-4)
- boomeramg: GPU 0.1185 (p 9.8853e-5), CPU 0.9936 (p 6.2468e-5); PETSc switches BoomerAMG to PMIS/ext+i/l1-Jacobi on device (ihypre.c)
- 4 mm n=4 (equal thermal): jacobi 1.3176 (p 9.6255e-5); boomeramg 1.1084 (p 3.8668e-5); gamg stopped by memguard at MemAvailable 1479 MB
- peak RSS per rank, 4 mm n=2, 5 iterations (`/usr/bin/time -f %M`): jacobi 2.79/2.67 GB, gamg 3.71/3.67 GB, boomeramg 3.48/3.49 GB
