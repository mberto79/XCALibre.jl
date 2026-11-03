* changed: `AMG(Pairwise())` keeps transfers as aggregate lists instead of sparse matrices on every backend: 1.5-1.65× faster solves with half the hierarchy memory on 3D Poisson matrices.
* changed: `NativeAMG(fuse=n)` counts the coarse levels applied through the finer matrix (was `n - 1`), on CPU and GPU, with `Jacobi()` or `Chebyshev()`.
