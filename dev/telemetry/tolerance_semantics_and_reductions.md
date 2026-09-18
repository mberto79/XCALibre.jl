# PETSc tolerance semantics and reduction count (P1-M11)

Case: `dev/scripts/scaling_probe.jl`, BFS tet 5 mm (499,503 cells), laminar SIMPLE, U bcgs+jacobi rtol 0.1, p cg+jacobi rtol 0.01, 30 outer iterations. Distributed runs under `dev/scripts/equal_thermal.sh` (NS="2 8"), stock env. Before = 107ca8f6 (PETSc default: rtol relative to the right-hand-side norm, CG preconditioned norm); after = rtol relative to the warm-started initial residual (UIRNorm) and CG natural norm.

Final residuals at outer iteration 30:
- serial Krylov.jl: p 1.0866e-4, Ux 1.8469e-5, Uy 5.2357e-4, Uz 3.6540e-4
- before, n=2 and n=8: p 1.1036e-3, Ux 9.2842e-4, Uy 1.5843e-2, Uz 1.4543e-2
- after, n=2 and n=8: p 1.2410e-4, Ux 1.8280e-5, Uy 5.2284e-4, Uz 7.8904e-4
- n=2 and n=8 agree to 13 significant figures before and after.

Per outer iteration, s (equal thermal, one run each):
- before: n=2 0.5801, n=8 0.2017 (speedup 2.88)
- after: n=2 0.6962, n=8 0.2456 (speedup 2.83)

PETSc events, n=2, 14 outer iterations (56 KSP solves), `-log_view`:
- before: VecTDot 2384, VecNorm 1325, VecDot 70, VecDotNorm2 35, MatMult 1304; reductions per CG iteration ~3.1
- after: VecTDot 2762, VecNorm 120, VecDot 156, VecDotNorm2 78, MatMult 1572; reductions per CG iteration ~2.1
