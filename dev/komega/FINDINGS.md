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

### Equivalence of the two assemblies - and a trap in how to check it

Comparing SOLUTIONS is useless on this case. After a single SIMPLE iteration the two
assemblies gave field sums differing by up to 20% (sumP), which looked like a bug. It is not.
At the benchmark's `rtol = 0.1`, a perturbation of 1e-16 in the matrix can flip whether a
Krylov solve stops after n or n+1 iterations, and one extra iteration changes the answer by
O(rtol), i.e. ~10%. The case is chaotic at the tolerance it is run at, so no solution-level
comparison on it can resolve anything finer than the solver tolerance. The same effect makes
the CELL path's answers depend on thread count, because the wall-function accumulators
(`Atomix.@atomic sums[cID] += Pf`) sum boundary faces in thread order.

The right gate is the MATRIX, compared before any solve (`matrix_check.jl`,
`matrix_check_mb.jl`):
  BFS 2D, 1800 cells, random fluxes of both signs, Upwind / Linear / LUST:
      max|dA| = 0, 0 of 8760 entries differ, b bitwise identical - all three schemes.
  motorBike, 353,830 cells, 2,470,770 nonzeros, real Wall/Slip/wall-function BCs,
  full k-equation term list (Time + Upwind + Laplacian + Si), 8 threads:
      max|dA| = 3.55e-15 (1.9e-16 relative), 0 entries differ above 1e-12, b bitwise identical.
So the face path builds the same matrix; the residual 1e-16 is the atomic accumulation order.

Note FaceAssembly is NOT bitwise reproducible run to run, for that reason. CellAssembly is
bitwise reproducible for the matrix, but the solver as a whole is not, because of the
wall-function atomics above - that is pre-existing and independent of this work.

Combined effect of rounds 1 and 2 on assembly at 8 threads: 53.3 -> 24.8 ms/iter, -54%.

## Verdict: XCALibre now WINS at 1 core, and still loses at 8

500 iterations, the benchmark's own case and settings, measured the same way:

                    before      after     OpenFOAM    result
  1 core            266.92 s    188 s     238.87 s    21% FASTER
  8 threads         121.79 s    100 s      81.66 s    22% slower
  scaling 1->8        2.19x      1.88x      2.93x

The 1-core win comes almost entirely from assembly: 207.7 -> 43.2 ms/iter (-79%), which was
36% of the iteration. At 8 threads assembly was already only 20% of the iteration and scaled
well (3.9x), so shrinking it helps less; what is left is dominated by Krylov, which scales at
1.5x. Scaling got WORSE (2.19 -> 1.88) precisely because the part that scaled well is now
small. Closing the 8-thread gap means fixing Krylov scaling, not assembly.

Caveats on these two numbers, stated rather than buried:
- They are ProgressMeter's loop totals, read from the run logs, because `bench500.jl` had a
  closure-scoping bug (`t = ...` inside a `redirect_stdout do` block creates a local, so the
  value never escaped). The benchmark's own figure additionally includes
  `setup_incompressible_solvers`, ~2.3 s on this mesh, so the comparable numbers are ~190 s
  and ~102 s. The script is fixed; a clean rerun should confirm.
- One sample each, on a machine with +/-6% run-to-run noise. 21% is well outside that; the
  8-thread 22% is too.

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

## Round 3: the Krylov gap, Int32 indices, and the GPU

### The measurement that the end-to-end A/B could not make

Round 1 left the BLAS-1 question unresolved because the whole-run A/B has +/-6% noise and the
effect is smaller than that. The fix is to stop timing whole runs: `krylov_bench.jl` builds the
real motorBike matrices (unsymmetric transport for Bicgstab, pure Laplacian for Cg, real BC
set, 2,470,770 nonzeros), then solves with `atol = rtol = 0, itmax = 50` so every variant does
exactly the same 50 iterations, and takes min of 14 reps. Iteration counts are printed to prove
the work is equal.

ms per Krylov iteration, motorBike, n = 353,830:

                          8 Julia threads        1 Julia thread
    variant             bicgstab      cg      bicgstab      cg
    blas1 opDiag i64      4.006     1.634      6.695     3.077   <- activate_multithread's default
    blas8 opDiag i64      3.238     1.331      5.861     2.902   <- what the benchmark actually ran
    blas1 thrDiag i64     3.749     1.578      6.722     3.066
    blas8 thrDiag i64     2.847     1.146      5.897     2.904
    blas1 opDiag i32      3.578     1.343      6.477     2.879
    blas8 thrDiag i32     2.340     1.005      5.644     2.731

Primitives at the same size (min of 200) explain it completely:

                        8 threads              1 thread
    spmv (threaded)      0.32 ms                2.04 ms
    diag precon, opDiagonal    0.104            0.130
    diag precon, threaded      0.011            0.125
    BLAS dot   1 / 8 threads   0.052 / 0.014    0.073 / 0.007
    BLAS axpy  1 / 8 threads   0.076 / 0.012    0.095 / 0.008

At 1 thread a Bicgstab iteration is 6.70 ms of which 4.1 ms is the two mat-vecs: it is a
bandwidth problem and only Int32 addresses it. At 8 threads the mat-vecs scale 6.4x and fall to
0.64 ms of 4.01 ms, so five sixths of the iteration is serial BLAS-1 vector work and a serial
diagonal preconditioner. That, and not the sparse kernel, is why Krylov scaled at 1.5x.

### 9. THREADED DIAGONAL PRECONDITIONER

`Preconditioner{Jacobi}` and `{NormDiagonal}` applied the diagonal through LinearOperators'
`opDiagonal`, a serial broadcast that runs once (Cg) or twice (Bicgstab) per Krylov iteration
over a full-length vector. `diagonal_operator` replaces it with a KernelAbstractions kernel, so
the same code path covers CPU threads and GPU. Worth 6% of every Krylov iteration at 8 threads
on its own and 14% in combination with threaded BLAS-1; neutral at 1 thread (0.125 vs 0.130 ms).

### 10. BLAS THREAD DEFAULT - and a hole in the earlier 1-core numbers

`activate_multithread(backend)` defaulted to `nthreads=1`. The default is now
`Threads.nthreads()`, so the vector half of each solve gets the same thread budget as the rest
of the solver.

While checking this I found that Julia starts OpenBLAS with **16** threads on this machine
whatever `-t` says, and `bench500.jl` never called `activate_multithread` at all. Two
consequences, both of which change earlier claims in this file:

- Every "1 core" number reported here, the 188 s result and the 266.92 s baseline alike, was
  run with 16 OpenBLAS threads doing the dots and axpys. They were not single-core runs.
  `bench500.jl` now calls `activate_multithread`, so `-t 1` means one core, and the 1-core
  column below is the first honest measurement of it.
- At 8 threads, threaded BLAS-1 is therefore not a new gain: it was already there. The live
  baseline is `blas8 opDiag`, not `blas1 opDiag`, so of the isolated 29% only the threaded
  preconditioner's 12-14% is new, which is ~4% of the iteration. The 500-iteration run agrees:
  100 s -> 97.01 s.

The isolated bench remains the right instrument; the error was in choosing which of its rows
was the status quo.

### Invariant the face path depends on

`_discretise_faces!` passes `nothing` where the cell loop passes `mesh.cells[cID]`. That is
safe because none of the nine `scheme!` methods reads the cell: every use of `cell.volume` in
`Discretise_1_schemes.jl` is inside `scheme_source!`, which only ever runs in the cell pass
(checked across Time/Euler/CrankNicolson, Divergence, Laplacian and Si). A new `scheme!` that
reads the cell would break `FaceAssembly` silently, since the cell-based default would keep
working. If that becomes a risk, pass the owner cell rather than `nothing`.

### 11. INT32 INDICES - the largest remaining lever, and it is free

`FOAM3D_mesh(...; integer_type=Int32)` (already a documented keyword on all three readers)
shrinks every connectivity array and the CSR addressing from 8 bytes to 4. Nothing else
changes: geometry stays Float64, and the 500-iteration residuals agree with the Int64 run to
11 significant figures in all four fields.

500 iterations, benchmark case and settings, `@elapsed run!`, BLAS matched to the Julia thread
count:

                    Int64      Int32     OpenFOAM
      1 core       178.39 s   164.13 s   238.87 s
      8 threads     97.01 s    76.72 s    81.66 s
      scaling 1->8    1.84x      2.14x      2.93x

Int32 is worth 8% at 1 thread and 21% at 8. It helps more than the Krylov bench alone predicts
(which saw 11% on the solve) because it shrinks assembly, gradient and flux traffic too, not
just SpMV. It also improves scaling, since less traffic per core is exactly what a
bandwidth-bound code at 8 threads needs.

## Verdict: XCALibre is now faster than OpenFOAM at 1 core AND at 8 threads

                       start      now      OpenFOAM    result
      1 core          266.92 s   164.13 s   238.87 s   1.46x FASTER
      8 threads       121.79 s    76.72 s    81.66 s   1.06x FASTER

Both "now" figures use the Int32 mesh; with Int64 it is 178.39 s and 97.01 s, i.e. still ahead
at 1 core and 19% behind at 8. The 1-core column is the first one measured with BLAS actually
restricted to one thread - see change 10, which invalidated every earlier 1-core number,
including the 188 s previously reported here.

Caveats, stated rather than buried: one sample per cell on a machine with +/-6% run-to-run
noise. The two Int32 results are 8% and 21%, so the 8-thread one is outside the noise and the
1-thread one is marginally so. The scaling row is the honest weak spot - OpenFOAM still gets
2.93x where we get 2.14x, so a machine with more cores would likely favour it again.
