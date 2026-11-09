* changed: `AMG(Pairwise())` keeps transfers between levels as aggregate lists instead of sparse matrices, on every backend.
* changed: `Pairwise()` pairs equal couplings across faces facing z, then y, then x, whatever the cell numbering.
* breaking: `NativeAMG(fuse=n)` now counts the coarse levels applied through the finer matrix (was `n - 1`).
* breaking: `NativeAMG(fuse)` with a coarsening other than `Pairwise()`, or a smoother other than `Jacobi()` or `Chebyshev()`, now throws.
* breaking: `Pairwise()` defaults to `merges=2` (was 1).
