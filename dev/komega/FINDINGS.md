# motorBike KOmega: why XCALibre is slower than OpenFOAM, and what was changed

Branch `HM/KOmega-profiling`. Mesh read from the benchmark directory; nothing there is modified.
Case: motorBike, 353,830 cells, SIMPLE, k-omega, matched linear solvers, p-rtol 0.1.

## Baseline (what the benchmark reports)
                1 core      8 cores
  OpenFOAM      238.87 s    81.66 s
  XCALibre      266.92 s   121.79 s
  ratio         1.12x       1.49x
XCALibre scales 2.19x on 8 threads, OpenFOAM 2.93x.

## Convergence is NOT the problem
Krylov iterations per SIMPLE iteration, measured:
  equation   OpenFOAM   XCALibre
  k          1.00       1.10
  omega      1.01       1.00
  U (3 cmp)  6.13       8.50
  p          63.76      25.78   (XCALibre runs the looser rtol)
k and omega match OpenFOAM exactly. The turbulence linear solves are already as cheap as
OpenFOAM's. The gap is entirely cost per iteration.

## Where the time goes (ms per SIMPLE iteration, phase timers)
                    1 thread   8 threads   1t->8t
  discretise (x4)     207.7       53.3       3.9x
  Krylov (x6)         173.8      114.9       1.5x
  gradients/flux       99.0       59.6       1.7x
  residual (x6)        25.4        9.6       2.6x

At 1 thread ASSEMBLY is the largest single cost, 36% of the iteration. Assembling the k
equation once (57.4 ms) costs as much as ~18 sparse mat-vecs, and more than the entire
pressure solve's 25.8 CG iterations.

KOmega's share of the whole iteration: 37% at 1 thread, 31% at 8. Two of the four assemblies
are k and omega, and they are the two most expensive ones.

## Why XCALibre's assembly is expensive (vs OpenFOAM)
OpenFOAM's `fvm::laplacian` is two flat array statements over faces:
    upper[f] = deltaCoeffs[f]*gammaMagSf[f];   negSumDiag();
~40 bytes per face, addressing precomputed in lduAddressing, each face visited once.

XCALibre assembles cell-by-cell (a deliberate no-atomics GPU design) and per (cell,face) pair:
  - loads the whole 128-byte `Face3D` struct - and every internal face is visited twice
  - loaded `cells[nID]`, a ~64-byte random access, that NO scheme uses (dead since the only
    reader is a commented-out line in the Laplacian)
  - recomputed the full Laplacian face geometry (2 dot products, a norm, a divide)
  - searched the CSR row linearly for each coefficient's position (`spindex`)
Estimated ~500 MB of traffic per assembly against OpenFOAM's ~40 MB.

## Changes made, with measured effect

Assembly cost is independent of how many SIMPLE iterations a run does, so these are
comparable across runs; Krylov cost is not (early iterations need more CG sweeps), so only
same-iteration-count runs are compared for it.

  discretise, ms/iter     1 thread          8 threads
                        before  after     before  after
  omega                  58.14  43.66      13.60  14.47*
  k                      57.44  43.46      13.45  14.36*
  U                      51.90  50.91      13.95  14.77*
  (*) the 8-thread "after" column is from the step3 run, which came out ~10% slower overall
  than the run before it on every phase including untouched ones - machine noise, not a
  regression. The clean 8-thread measurement of changes 1+2 alone is:
  omega 11.54, k 11.49, U 12.03, p 9.02 - assembly total 53.3 -> 44.1 ms, -17%.

1. `cells[nID]` removed from both assembly kernels - a dead random load per (cell,face).
   `scheme!` now takes `nID` instead of the cell struct.
2. Laplacian face geometry reduced to `area/(|normal.e|*delta)`. Verified equal to the
   expanded form to 1.1e-15 over all 1,058,470 internal faces (`check_laplacian_algebra.jl`).
   1+2 together: k and omega assembly -24% at 1 thread, -15% at 8.
3. `spindex` replaced by precomputed nzval index maps (`diag_nz`, `face_nz` on the equation),
   also used by the relaxation, `inverse_diagonal!` and `H!`.
   MEASURED NEUTRAL ON CPU (53.32 -> 53.48 ms) - the CSR row stays in cache, so the linear
   scan was never the cost. Kept because it should matter more on GPU, but it bought nothing
   here and is the change to drop first if the diff needs to shrink.
4. BLAS thread count - INVESTIGATED, NOT ADOPTED. `activate_multithread` sets BLAS to 1
   thread, and Krylov.jl sends every dot and axpy on a `Vector{Float64}` straight to BLAS, so
   the vector half of every linear solve is serial at any Julia thread count. In isolation at
   n=353830 on 8 threads this is worth a lot: dot 59.0->12.3 us (4.8x), axpy! 79.3->7.5 us
   (10.6x); nrm2 does not thread either way. Predicted end-to-end gain ~5% at 8 threads.
   A paired in-process A/B (`ab_blas.jl`, 3 reps x 15 iterations, arms alternating so drift
   hits both) could NOT resolve it:
     BLAS 1 : 471.9 509.5 468.8 ms/iter   (min 468.8)
     BLAS 8 : 467.1 525.0 493.9 ms/iter   (min 467.1)
   Run-to-run noise is +/-6%, larger than the effect. The default is therefore left at 1 and
   the finding recorded; `activate_multithread(backend, nthreads=N)` already exposes it.
   To settle this, run `ab_blas.jl` with more reps on an otherwise idle machine.
5. `turbulence!` source/flux update fused from seven passes into two (one cell, one face),
   with the strain-rate magnitude no longer written to Pk and read back twice.

Correctness:
- Residuals on motorBike match to 13 significant figures (reassociated arithmetic in the
  Laplacian); k, omega and nut field sums match to 15. The comparison is against the code
  with the nz index maps already in, since those are provably index-equivalent to the
  `spindex` calls they replace - not against unmodified `main`.
- Test cases, all passing after round 2 as well (`dev/komega/komega_tests.jl`, on
  `--project=test`), 52/52 in 1m48s, zero failures, same counts as before the changes:
    2d_incompressible_flatplate_KOmega_lowRe       7/7
    2d_incompressible_flatplate_KOmega_HighRe     19/19
    2d_incompressible_transient_KOmega_BFS_lowRe   7/7
    2d_compressible_KOmega_flatplate_fixedT       10/10
    2d_incompressible_laminar_BFS                  4/4
    3d_incompressible_laminar_BFS                  5/5
                                            total 52/52, 0 failures, 1m58s
  The full `Pkg.test()` suite has NOT been run to completion on this branch: this machine has
  14 GB of RAM and the 3D cases plus the profiling runs exhaust it. Run it on a larger box
  before merging - in particular the LES, multiphase, MRF, periodic and supersonic cases,
  which also go through the assembly kernel and `scheme!`.

## Round 2: cached face coefficient + selectable face/cell assembly

Three changes, measured at 8 threads on the same case (phase timers, ms/iter):

  discretise        start   round2-cell   round2-face
    omega           13.60      5.31          6.75
    k               13.45      5.07          6.77
    U               13.95     10.49          7.83
    p               12.32      3.93          6.00
    TOTAL           53.32     24.79         27.35     (-54% / -49%)

6. PER-FACE LAPLACIAN COEFFICIENT, cached on the equation. `gDiff[f] = area/(|normal.e|*delta)`
   is constant for a fixed mesh, so the Laplacian reads one number instead of eight and skips
   two dot products, a norm and a divide. For k and omega, whose Upwind divergence reads
   nothing from the face, the 128-byte `Face3D` load disappears from the assembly entirely -
   which is why they drop hardest (13.5 -> 5.1 ms, -62%).
7. FACE-BASED ASSEMBLY with atomics, selectable via `Hardware(assembly=FaceAssembly())`;
   `CellAssembly()` is the default and the pre-existing path. Both are kept and both are
   exercised by `ab_assembly.jl`.
   RESULT ON CPU: the two are within ~1% of total runtime, which matches what the user found
   independently. Per equation it splits: cell wins on the scalar equations (k, omega, p),
   face wins on the vector one. That is consistent with what each still reads - after change
   6 the scalar schemes touch no face struct at all, so the cell loop's duplicated work is
   nearly free, while U's LUST divergence still reads `face.weight` and so benefits from
   loading the face once. Face assembly is the better choice on GPU (user's own experiments:
   hardware atomics, and conflicts are rare enough not to hurt the CPU either).
8. WALL-FUNCTION BUFFERS preallocated on `KOmegaModel` instead of allocating and zeroing two
   cell-sized arrays per outer iteration. turb_wallfun 2.72 -> 1.82 ms.
   The fused source loops from round 1 also show clean now: turb_sources 3.65 -> 2.20 ms.

Equivalence of the two assemblies after 20 SIMPLE iterations on motorBike: residuals agree to
1.8e-5 relative, field sums to 1e-7..1e-9. That is round-off divergence amplified through 20
nonlinear iterations, not a discrepancy in the matrix - but note FaceAssembly is NOT bitwise
reproducible run to run, because the order of the atomic diagonal accumulations varies.
CellAssembly remains bitwise reproducible.

Combined effect of rounds 1 and 2 on assembly at 8 threads: 53.3 -> 24.8 ms/iter, -54%.

## Verdict: this does not beat OpenFOAM yet

Measured against the instrumented steady-state cost (not wall - the 20-iteration walls carry
~57 ms/iter of one-off `run!` setup that is ~4 ms at the benchmark's 500):
  1 thread: saved ~30 ms of ~573  (~5%);  the gap to OpenFOAM was 12%
  8 threads: saved ~9 ms of ~264  (~3%);  the gap to OpenFOAM was 49%
So roughly half the 1-core gap is closed and little of the 8-core gap. The two levers that
would close the rest are both identified and both unimplemented - see below.

## Still on the table, not done
- BIGGEST REMAINING 1-THREAD LEVER. `faces[fID]` is still a 128-byte load per (cell,face),
  and every internal face is visited twice, so the motorBike mesh moves ~271 MB of face
  structs per assembly. Precomputing one per-face scalar (`area/(|normal.e|*delta)`, exactly
  OpenFOAM's `deltaCoeffs*magSf`) plus the interpolation weight removes the struct load
  entirely for k and omega - their Upwind divergence reads nothing from the face at all.
  Estimated ~4x less assembly traffic. Needs somewhere to cache the array; an extra field on
  the Laplacian `Operator` is the least invasive spot.
- BIGGEST REMAINING 8-THREAD LEVER. Krylov scales at only 1.5x (p 1.70x, U 1.44x) while
  assembly scales 3.9x. Part of that is the serial BLAS-1 above. One confound was not
  eliminated: `pinthreads(:cores)` pins the Julia threads, but OpenBLAS spawns its own
  unpinned threads, which on this P/E hybrid chip may land on efficiency cores. Re-run
  `ab_blas.jl` with OpenBLAS pinned (or more reps on an idle machine) before concluding the
  effect is not there.
- Every internal face is still assembled twice on CPU. A face-based CPU path (keeping the
  cell-based one for GPU) would halve the scheme work.
- `wall_cell_accumulators` allocates and zeroes two N-cell arrays every outer iteration
  inside `correct_production!`; they belong in `KOmegaModel`.
- `residual()` costs a full extra SpMV plus two reductions after every solve (25.4 ms/iter at
  1 thread). OpenFOAM gets its residual free from the solver.
- `turb_gradU` (14.7 ms at 8 threads) and `simple_flux`/`simple_gradp` scale at only ~1.6x.
- Int32 indices: OpenFOAM runs `WM_LABEL_SIZE=32` by default, XCALibre's `FOAM3D_mesh`
  defaults to Int64 - ~17% more SpMV traffic. `integer_type=Int32` is already supported;
  `dev/komega/build_mesh_i32.jl` builds the mesh. NOT YET MEASURED. This is a benchmark
  setting, not a code change.

## Next levers, now that assembly is no longer the top cost

At 8 threads the steady-state split is now roughly: Krylov ~115 ms, assembly ~25 ms,
gradients/flux/Hv ~48 ms, residual ~10 ms. The ranking has changed:
- KRYLOV (~44%) is the target and is bandwidth bound. The one untried traffic reduction is
  Int32 indices: OpenFOAM runs `WM_LABEL_SIZE=32`, XCALibre's `FOAM3D_mesh` defaults to
  Int64, so colval alone is ~10 MB/SpMV of avoidable traffic, and every index array in the
  assembly halves too. `integer_type=Int32` already exists; `build_mesh_i32.jl` builds it.
  STILL NOT MEASURED.
- `turb_gradU` (~15 ms) is Green-Gauss grad(U), and its cell pass visits each internal face
  twice exactly like the old assembly did. The same face-loop-with-atomics treatment now
  available for the matrix applies to it.
- `residual()` still costs a full extra SpMV plus two reductions after every solve.

## Diff notes for a future PR
- `scheme!` is exported and its signature changed twice: `cellN` -> `nID`, and a `gDiff_f`
  argument was added before `nID`. Any user-defined scheme method breaks and must be updated.
  Nothing in-tree read `cellN`.
- `Hardware` gained a third field (`assembly`) and a third type parameter. It is `@kwdef`, so
  keyword construction is unaffected, but positional `Hardware(backend, workgroup)` now needs
  a third argument.
- `ScalarEquation`/`VectorEquation` gained `owner_nz`, `neig_nz` and `gDiff` alongside the
  index maps. `nz_index_maps` builds all of them on the host in one serial pass per equation;
  on a 354k-cell mesh that is a visible one-off cost at setup and would be worth a kernel.
- `KOmegaModel` gained a `wall_buffers` field. `correct_production!` and
  `correct_eddy_viscosity!` take an optional trailing `buffers` argument, defaulting to the
  old allocating behaviour, so other turbulence models are unaffected.
- The `xcprof` phase timers live in `src/` (Multithread/profiling.jl plus call sites in
  Solvers_1_SIMPLE, Solve_1_api and RANS_kOmega). They are opt-in and cost one Ref load when
  off, but they would be stripped or moved behind a compile-time flag for a real PR.
- Change 3 (nz index maps) is measured neutral on CPU. If the diff needs to shrink, drop it
  first; keep it only if a GPU measurement justifies it.
- Residual comparisons in this file are against the code with the nz index maps already in,
  since those are provably index-equivalent to the `spindex` they replace. They are not a
  comparison against unmodified `main`.

## Reproducing
  julia --project=dev/komega -t N dev/komega/profile_motorbike.jl <iters> <out> [profile]
  julia --project=dev/komega dev/komega/check_laplacian_algebra.jl
  julia --project=dev/komega -t 8 dev/komega/ab_blas.jl <reps> <iters> <out>
The mesh is read from the benchmark directory and nothing there is written.
