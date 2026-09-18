# PETSc index width (P1-M14)

Case: BFS tet 5 mm (499,503 cells), laminar SIMPLE, U bcgs+jacobi, p cg+jacobi, 30 outer iterations, `dev/scripts/equal_thermal.sh` NS="2 8", stock PETSc_jll, interleaved. INT64 = f1edb22e (first Float64 library, 64-bit indices); INT32 = narrowest index type that fits the global nonzero count. s/iter:
- n=2: INT64 0.6952, 0.6883; INT32 0.6781, 0.6403
- n=8: INT64 0.2380, 0.2472; INT32 0.2150, 0.2128
- final residuals bitwise identical (p 1.2409749477950918e-4 at n=2).
