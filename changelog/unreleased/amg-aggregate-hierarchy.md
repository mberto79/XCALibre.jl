* changed: `AMG(Pairwise())` keeps transfers as aggregate lists instead of sparse matrices on every backend: 1.5-1.65× faster solves with half the hierarchy memory on 3D Poisson matrices.
* changed: `NativeAMG(fuse=n)` counts the coarse levels applied through the finer matrix (was `n - 1`), on CPU and GPU, with `Jacobi()` or `Chebyshev()`.
* changed: `AMG()` defaults to `Pairwise(2)`, and equal couplings pair across faces facing z, then y, then x, whatever the cell numbering: 6.5 s instead of 12.8 s for 100 SIMPLE iterations on a 192k-cell hex mesh, 8 threads.
