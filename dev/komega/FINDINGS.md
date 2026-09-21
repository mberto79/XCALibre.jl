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
  UPDATE, after round 3: the full `Pkg.test()` suite DOES run on this machine and passes,
  1544/1544 in 7m38s, exit 0 (only skip is a CUDA AMG test the suite skips when CUDA is absent
  from the test env). The earlier claim that 14 GB was not enough was wrong - that OOM came
  from running motorBike profiling jobs concurrently with the suite, not from the suite. Run
  the suite on an idle machine and it is fine. It covers Incompressible, Compressible,
  Godunov, Multiphase, Thin Film, AMG, DILU and Smoothers, so it exercises both the assembly
  kernels and the new `diagonal_operator` on every solver.

## Round 2: cached face coefficient + selectable face/cell assembly

Three changes, measured at 8 threads on the same case (phase timers, ms/iter):

  discretise        start   round2-cell   round2-face
    omega           13.60      5.31          6.75
    k               13.45      5.07          6.77
    U               13.95     10.49          7.83
    p               12.32      3.93          6.00
    TOTAL           53.32     24.79         27.35     (-54% / -49%)

6. PER-FACE LAPLACIAN COEFFICIENT, cached per mesh. `gDiff[f] = area/(|normal.e|*delta)`
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
- `scheme!` is exported and its signature changed once: `cellN` -> `nID`. Any user-defined
  scheme method breaks and must be updated. (The `gDiff_f` argument this list used to mention
  was removed again when the coefficient moved onto the Laplacian operator.)
- `Mesh2`/`Mesh3` gained a `face_gDiff` field and a type parameter, after `face_nodes`. A
  13-argument outer constructor derives the array, so all four construction sites (UNV2,
  UNV3, FoamMesh, `_rebuild_mesh_float`) are untouched and no call site can supply a stale
  one; the 3D readers, which fill the face geometry after the mesh exists, are covered by
  `update_face_gDiff!` at the end of `compute_3d_geometry!`. Anything constructing a mesh
  with all 14 fields positionally, or destructuring one positionally, breaks. `Operator` is
  back to four fields and carries no coefficient.
- Boundary faces store `area/delta` and internal faces `area/(|normal.e|*delta)`, branching
  on `is_boundary` (new, exported): boundary faces store their owner cell twice, which UNV2,
  UNV3 and FoamMesh each set deliberately. The MPI path must honour that for processor
  faces. The `@define_boundary Laplacian` blocks still read `(; area, delta) = face`, so
  results are bitwise identical to the previous placement.
- `ScalarEquation`/`VectorEquation` gained `diag_nz` and `face_nz`, so anything constructing
  them positionally breaks.
- Every turbulence model struct gained a `wall_scratch` field, built by `wall_scratch(mesh,
  boundaries, config)` in its `initialise`.
- `src/precompile.jl` is stale: its type-literal statements naming `ScalarEquation`,
  `Operator` or `scheme!` no longer match, so they silently stop precompiling. Regenerate it
  before merging - startup latency on this repo has been a problem before.
- The opt-in phase timers (`XCPROF`, `@xcprof`) were removed; recover from `211474fe` if a
  phase split is ever needed again.
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

A related inconsistency between the two instruments used in this file: `profile_motorbike.jl`
calls `activate_multithread`, so every phase-timer table above ran with BLAS on 1 thread, while
`bench500.jl` did not, so every 500-iteration headline ran with BLAS on 16. The "Krylov scales
only 1.5x" diagnosis is therefore a BLAS-1 measurement. It changed no decision - the serial
diagonal preconditioner and the Int32 lever are both real under either setting - but the two
tables were never taken under the same conditions. Both scripts now call it.

### Invariant the face path depends on

`_discretise_faces!` passes `nothing` where the cell loop passes `mesh.cells[cID]`. That is
safe because none of the nine `scheme!` methods reads the cell: every use of `cell.volume` in
`Discretise_1_schemes.jl` is inside `scheme_source!`, which only ever runs in the cell pass
(checked across Time/Euler/CrankNicolson, Divergence, Laplacian and Si). A new `scheme!` that
reads the cell would break `FaceAssembly` silently, since the cell-based default would keep
working. If that becomes a risk, pass the owner cell rather than `nothing`.

### 11. INT32 INDICES - the largest remaining lever, and it is free

`FOAM3D_mesh(...; integer_type=Int32)` (already a documented keyword on all three readers)
shrinks every connectivity array and the CSR addressing from 8 bytes to 4. Verified on the 2D
BFS mesh: `colval`, `rowptr`, `cell_faces` and all four nz index maps come out `Int32`, so it
propagates through the matrix builder and not just the mesh. Nothing else
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

## VOID - the table below was measured in the wrong power mode

The runs in this section were taken with the laptop accidentally in "performance" platform
profile, while the OpenFOAM benchmark was run in "balanced". They are kept for the record but
are NOT the comparison; see "Verdict, balanced power mode" at the end of this file. The
difference turned out to be small (3.4% at 8 threads, 1.5% at 1 core) and changes no
conclusion, but it changes the margins.

## Superseded verdict (performance mode)

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

## Round 3b: the GPU (RTX 4070 Laptop, 8 GB)

Isolated assembly, min of 20 reps after 3 warm-up launches, arms interleaved, every timed
region closed with `KernelAbstractions.synchronize`. The solver's phase timers wrap
asynchronous launches and are meaningless on GPU, so they are not used here. One process runs
every variant because each new process pays the full GPU kernel compilation (~15 min).

  discretise!, ms          k        p        U      2k+p+U
    pre-gDiff, cell      6.377    2.644    6.260    21.657
    current,   cell      5.384    0.555    6.286    17.609
    current,   face     14.980    0.713   13.239    43.911
    current,   cell i32  5.160    0.442    5.954    16.716
    current,   face i32 14.623    0.664   12.818    42.728

### The gDiff change helps on GPU too, but for a different reason

Assembly per SIMPLE iteration 21.657 -> 17.609 ms, -19%. The split is nothing like the CPU's:
p falls 79% (2.644 -> 0.555), k falls 16%, U not at all - where on CPU k fell 62%. That fits.
On CPU the win was mostly the 128-byte `Face3D` load disappearing; on GPU that load is
coalesced and bandwidth is not the binding constraint, so what is left is the removed
arithmetic - two dot products, a norm and a divide. The p equation is pure Laplacian, so
removing that arithmetic removes nearly all of its work; U's LUST divergence still dominates
and is untouched.

### Face assembly is 2.5x SLOWER than cell assembly on this GPU

REMOVED FROM THE BRANCH. The numbers in this section and the four-way table below are the
reason: face assembly loses on GPU and ties on CPU, so it earns nothing and costs a second
assembly path to maintain. `AbstractAssembly`/`FaceAssembly`/`CellAssembly`, the `assembly`
field of `Hardware`, the two face kernels, and the `owner_nz`/`neig_nz` index maps are gone;
the cell path is unchanged, so no result above needs re-measuring. The scripts that existed
only to compare the two (`ab_assembly.jl`, `matrix_check*.jl`, `eq_check*.jl`, `gpu_run.jl`)
went with it.

Not the expected result, and opposite to the CPU, where the two are within 1%. Aggregate 2k+p+U is 2.5x
(43.911 vs 17.609); k alone is 2.8x. End-to-end over 20 SIMPLE iterations: cell 227.79 ms/iter,
face 254.78 ms/iter, face 12% slower.

It is not atomic contention. The cell path uses no atomics on internal faces at all; the face
path issues exactly two Float64 atomics per face whatever the equation is, so the whole atomic
cost is bounded by p's excess over cell, 0.16 ms. k's excess is 9.6 ms. What the ratio tracks
instead is the number of terms:

    p (1 term)      1.28x
    U (3 terms)     2.11x
    k (4 terms)     2.78x

What does scale with term count is per-thread work: `_discretise_faces!` calls `_scheme!`
twice, once from each side of the face, so a 4-term equation inlines eight scheme bodies into
one thread with both results live at once. The cell loop emits the body once inside a loop and
reuses the registers. Register spill to local memory is the explanation consistent with all
three ratios; it is an inference from the term-count trend, not a measured register count.

If that is right, the fix is to stop emitting `_scheme!` twice - process the two sides in a
loop over a 2-tuple of (nID, sign, diagonal, offdiagonal), or split the face kernel per term.
Worth testing before concluding that face assembly is wrong for GPUs in general: this measures
one kernel structure, not the idea.

Int32 is worth 5% of assembly on GPU (17.609 -> 16.716). That is not comparable with the 21%
it is worth on the 8-thread CPU, which is an end-to-end figure; CPU assembly was never timed
in isolation against an Int32 mesh.

The 20-iteration full solve also confirms `diagonal_operator` (change 9) compiles and runs on
CUDA, with residuals in the expected range. GPU vs CPU wall time is NOT compared here: the GPU
run is 20 iterations and the CPU benchmark 500, and early SIMPLE iterations carry far more
Krylov work, so the per-iteration figures are not comparable.

### GPU end-to-end, all four combinations (100 iterations each, one process)

  RTX 4070 Laptop            wall      ms/iter
    i64  cell               9.06 s      90.65
    i64  face              11.61 s     116.09
    i32  cell               8.87 s      88.66
    i32  face              11.68 s     116.80

Residuals identical across all four to 5 significant figures.

- Cell beats face by 28% end-to-end, so the 2.5x assembly gap is not an artefact of the
  isolated measurement. `CellAssembly` is the right default on GPU as well as CPU.
- Int32 is worth 2.2% on GPU, inside run-to-run noise, against 21% on the 8-thread CPU. The
  assembly-level figure said 5%, so the two agree: this GPU is not addressing-bandwidth bound
  on this case. Int32 is a CPU lever here, not a GPU one - the opposite of the usual
  assumption, and the reason it is worth measuring rather than assuming.

Use 100 iterations, not 20, for any per-iteration GPU figure: the 20-iteration run earlier in
this file reported 227.79 ms/iter for the same i64 cell configuration that averages 90.65 over
100, because the early SIMPLE iterations carry far more Krylov work.

Cross-device comparison, stated carefully: the GPU's 100-iteration average (88.66 ms/iter) is
already below the 8-thread CPU's 500-iteration average (153.4), and lengthening a run only
lowers its per-iteration average, so the GPU is ahead of the 8-thread CPU by at least 1.7x on
this case. A matched-iteration comparison was not run.

## VERDICT, balanced power mode - the like-for-like comparison

Platform profile `balanced`, power-profiles-daemon `balanced`, EPP `balance_performance`,
turbo on; confirmed by the user to be the same power setting used for the OpenFOAM benchmark.
Int32 mesh, `CellAssembly`, 500 iterations, `@elapsed run!`, BLAS matched to the Julia thread
count, machine otherwise idle.

                          XCALibre     OpenFOAM    ratio
      1 core              166.53 s     238.87 s    1.43x faster
      8 threads            79.34 s      81.66 s    1.03x faster
      GPU, RTX 4070 Lap    29.55 s        -        2.76x faster than OF on 8 cores

Power mode cost 3.4% at 8 threads (76.72 -> 79.34) and 1.5% at 1 core (164.13 -> 166.53), so
the earlier performance-mode numbers were not far off, but these are the ones to quote.

Read the 8-thread row honestly: 2.8% is INSIDE the +/-6% run-to-run noise measured on this
machine. At 8 threads XCALibre is level with OpenFOAM, not demonstrably ahead; claiming a win
there needs repeats. The 1-core result and the GPU are well outside noise.

GPU per-iteration falls from 88.66 ms over 100 iterations to 59.10 ms over 500, the same
early-iteration effect noted above - another reason to quote only full-length runs.

## Round 4: the 8-thread "scaling deficiency", measured rather than inferred

Everything below is from this branch's current code (gDiff, Int32, threaded diagonal, cell
assembly), balanced power mode, machine idle.

### 1. The machine's roof: threads cannot buy 2.9x here

STREAM triad (2 reads + 1 write, 320 MB arrays, far outside the 36 MB L3),
`dev/komega` env, `pinthreads(:cores)`, min of 6:

      threads    GB/s    scaling vs 1 thread
         1       25.0      1.00x
         2       28.5      1.08x
         4       36.8      1.48x
         8       38.8      1.55x
        16       32.4      1.23x
        24       31.0      1.24x

One core already pulls 25 GB/s, 64% of everything the memory system will ever give. The
maximum possible 1->8 speedup of a purely DRAM-bound phase on this laptop is 1.55x, and past
8 threads the E-cores make it worse. So 2.93x is not a number any bandwidth-bound solver
reaches on this machine with threads, and XCALibre's 2.10x is already ABOVE the streaming
roof (it is not purely bandwidth-bound; assembly and the gradients have cache reuse).

### 2. Fresh phase split - the old narrative is dead

`profile_motorbike.jl 50 ... i32`, ms per SIMPLE iteration:

      phase              1 thread   8 threads   1t->8t
      p_krylov              76.97      34.95      2.20x
      U_krylov              65.95      37.41      1.76x
      k+omega_krylov        21.49      13.70      1.57x
      turb_gradU            24.81      13.88      1.79x
      simple_flux           17.13      10.71      1.60x
      simple_gradp          16.23      10.74      1.51x
      simple_Hv             12.39       7.25      1.71x
      simple_massflux        9.81       4.04      2.43x
      discretise (x4)       38.17      23.77      1.61x
      residual (x4)         23.52       9.15      2.57x
      WALL                 436.38     242.72      1.80x

Two things changed since the Round 1 table. Assembly no longer scales at 3.9x - it scales at
1.61x, because gDiff and the cached nz indices turned it from compute-bound into
bandwidth-bound. And Krylov no longer scales at 1.5x - it is 1.76-2.20x, because the BLAS
thread count and the diagonal preconditioner were fixed. **Every phase now sits in the
1.5-1.8x band, i.e. on the triad roof.** There is no lagging phase left to fix: the code is
uniformly memory-bound at 8 threads, and adding threads is finished as a lever.

(Phase ratios come from a 50-iteration window, which over-weights the early solver-heavy
iterations; the headline 2.10x is from the 500-iteration runs. The ratios are the point.)

### 3. Linear-solver work is already matched, or better

Mean Krylov iterations per solve. XCALibre's profile covers the first 50 SIMPLE iterations,
so OpenFOAM is counted over the same window; its 500-iteration means are given too, because
early solves are much heavier (its first three p solves are 82, 143, 62 iterations).

                     XCALibre      OpenFOAM        OpenFOAM
                     first 50      first 50        all 500
      U (each)          2.8       2.52-3.02      1.93-2.12
      p                25.1          86.9           63.9
      k                 1.08          1.02           1.01
      omega             1.00          1.02           1.00

`xcprof_report` divides units by SIMPLE iterations, not by calls, so the U row's raw 8.4 is
three component solves; 2.8 each is the comparable figure.

Momentum and turbulence are matched solve for solve. Pressure is not: XCALibre runs 3.5x
fewer iterations there, but at rtol 0.1 against OpenFOAM's relTol 0.01, so that is a
tolerance difference, not a win - and the README puts the cost of closing it at ~20%, which
is larger than the whole disputed 8-thread margin. XCALibre still only ties on wall time
while doing less pressure work. The gap is therefore cost per unit of work, not amount of work. That
also removes "fewer Krylov iterations" (multigrid) from the top of the lever list: on this
machine OpenFOAM's own GAMG is worth only 4% at 8 cores (76.18 vs 79.66 s, psolver_study.txt).

### 4. The OpenFOAM reference numbers are PCG, not GAMG - check this

Every reference log (`OpenFOAM/log.simpleFoam_{1,2,6,8}`, written 20 Sep 19:35-19:41) reports
`diagonalPCG:  Solving for p`, and their `ExecutionTime` values are exactly the four quoted
figures 238.87 / 170.52 / 90.89 / 81.66. `system/pSolver` is currently the PCG copy.
`fvSolution` says the main benchmark should use `pSolver.GAMG`, so either the logs were made
while `run_psolver_study.sh` had PCG in place, or the intent changed. Consequences:

- The comparison is currently solver-MATCHED (both Cg/PCG + Jacobi/diagonal), not unmatched.
- OpenFOAM's best configuration on this machine is 206.89 s at 1 core and 76.18 s at 8
  (psolver_study.txt), so against its best the honest read is 1 core 1.24x faster, 8 threads
  4% slower - not 1.43x / level.
- The matched comparison still favours XCALibre on tolerance: OpenFOAM's p runs relTol 0.01
  (63.9 its), XCALibre rtol 0.1 (25.1 its). The README justifies this (different residual
  definitions; 0.01 costs ~20% for a 0.003% drag change), but ~20% is the size of the whole
  disputed margin, so it belongs next to any claim of a tie.

### 5. What actually produces OpenFOAM's 2.93x: decomposition, not threading

OpenFOAM at "8 cores" is 8 MPI ranks of ~44k cells each. A 44k-cell subdomain's matrix and
vectors are a few MB and largely stay in cache; a 354k-cell shared-memory problem does not.
XCALibre's own numbers on this same case show the identical effect, on identical code:

                    1 core    8 cores   scaling
      MPI ranks     299.13     93.32     3.21x
      threads       276.42    124.56     2.22x

MPI starts 8% slower on one core and finishes 25% faster on eight. That is the whole story of
the scaling ratio, and it is a property of the decomposition, not of OpenFOAM.

### 6. Ranked levers, by measured ms at 8 threads

1. **Re-measure the MPI path with this branch's changes.** It is the only lever already
   demonstrated to be worth >20% at 8 cores on this case. It needs no new solver code, but it
   is not on this branch: `src/` has no Distribute module, the distributed benchmark ran on
   `HM/distributed-draft`, so this means merging that branch (or cherry-picking gDiff + the
   Int32 mesh onto it) first. Only some of this branch transfers: gDiff and Int32 do, the
   threaded diagonal and the BLAS default do not, because PETSc owns the solve there.
   PROJECTION, not a measurement: if MPI keeps its 25% edge over threads, 8 ranks lands near
   60-65 s against OpenFOAM's 76-82 - outside the +/-6% noise, which the threaded number is
   not.
2. **Fuse the gradient/flux group** - 46.6 ms/iter at 8 threads (19%), spread over
   `turb_gradU`, `simple_gradp`, `simple_flux`, `simple_Hv`, `simple_massflux`. These are
   separate full passes over the same face and cell arrays. Fusing passes removes DRAM
   traffic, which is the only currency left. NOT face-based loops - that pattern was just
   rejected.
3. **Cell renumbering** (RCM or a space-filling curve) to shrink the matrix bandwidth. Same
   idea as 1, applied inside one process: it improves `x[colval[nz]]` locality in the SpMV
   and the neighbour loads in assembly. The benchmark never runs `renumberMesh`, so both
   codes read snappyHexMesh order and this is an absolute win, not a gap-closer.
4. **`residual()`** - 9.15 ms/iter (3.8%), a full extra SpMV plus two reductions after each
   of six solves. OpenFOAM gets its residual from the solver for free.
5. **Kernels that anti-scale**: `turb_sources` 3.05 -> 3.87 ms and `simple_copies`
   0.47 -> 0.87 ms are SLOWER on 8 threads than on 1. Short loops paying thread-launch
   overhead; merge them into adjacent kernels or run them serially.

What is NOT worth doing, with the reason: more threads (past 8 the bandwidth drops), a
better pressure preconditioner (OpenFOAM's own is worth 4% at 8 cores and XCALibre already
does 2.5x fewer pressure iterations), and further assembly work (23.8 ms and bandwidth-bound).
