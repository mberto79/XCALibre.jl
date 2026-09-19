# Memory breakdown (P1-M16)

Machine: this laptop, `dev/petscenv_stock` (PETSc_jll 3.22, Float64/Int32 library), Julia 1.13, CPU backend, n=2, laminar BFS with Jacobi on U and p, 5 SIMPLE iterations, parts written offline (`part` mode) so no rank holds the global mesh. Script: `dev/scripts/mem_probe.jl`. Raw per-rank tables and PETSc allocation sites: `dev/telemetry/memory_breakdown/`. Rank 0 shown; rank 1 is within 1 percent everywhere.

Two runs per mesh: `gc=0` is the natural run (its HWM is the peak a user sees); `gc=1 malloc=1` forces `GC.gc(true)` before every reading, so live bytes attribute cleanly, and starts PETSc with `-malloc_debug`, so `PetscMallocGetCurrentUsage` and the exit leak dump give PETSc's own heap by allocation site.

## Peak RSS per rank (natural run)

| mesh | global cells | local cells rank 0 | peak RSS MB | per local cell |
|---|---|---|---|---|
| 10 mm | 68,243 | 34,428 | 852 | 25.3 KB |
| 5 mm | 499,503 | 250,558 | 1202 | 4.9 KB |
| 4 mm | 1,320,368 | 662,391 | 1950 | 3.0 KB |

Fit: peak ≈ 790 MB fixed + 1.85 KB per local cell (slope 1.83 KB 10→4 mm, 1.90 KB 5→4 mm). The earlier 2.79 GB at 4 mm n=2 (`preconditioner_guidance.md`) came from `scaling_probe.jl`, which calls `run!` three times in one process, each building new equations and PETSc objects. Re-run on the same parts it peaks at 2838/2830 MB per rank (`memory_breakdown/scaling_probe_4mm_n2_three_runs.tab`), 890 MB above one run, so it is not one run's footprint; whether the extra is uncollected garbage or PETSc objects never destroyed between runs is open (P1-M21-S5).

## Attribution, 4 mm n=2 rank 0 (660,176 owned cells)

| component | MB | B/cell | share of 1950 peak | measured as |
|---|---|---|---|---|
| Julia runtime + loaded packages | 546 | fixed | 28% | RSS before the mesh |
| compiled code + allocator slack | ~284 | ~fixed | 15% | final RSS minus runtime, Julia live and PETSc; 260 MB of it at 10 mm |
| local mesh + partition maps | 366 | 553 | 19% | live delta; `summarysize` 340 mesh + 26 partition/procs/orig |
| fields U, p, Uf, pf | 62 | 98 | 3% | live delta |
| aux fields ∇p, mdotf, rDf, nueff, divHv | 51 | 81 | 3% | live delta |
| U equation: A0 55, A 55, 5 vectors 26 | 135 | 214 | 7% | live delta and `summarysize` |
| p equation: A 55, 3 vectors 15 | 70 | 111 | 4% | live delta and `summarysize` |
| PETSc matrix storage, U and p | ~110 | 174 | 6% | exit dump: values 49.6, column indices 24.8, row/diag/aux ~35 |
| PETSc COO maps, U and p | 99.2 | 158 | 5% | exit dump: two 8-byte-per-nonzero arrays per matrix (`MatSplitEntries_Internal`, `MatSetPreallocationCOO_MPIAIJ`); classified by bytes per nonzero (PetscCount is 8 bytes, values are the third such array), not by source line, since the dump is 3.22 and the local source 3.24 |
| PETSc vectors | 75.5 | 120 | 4% | exit dump: KSP work vectors 45.3, x/b and others 30.2 |
| SIMPLE work arrays: Hv 16, rD/prev/p_boundary_reference 16, gradU 48 | ~80 | 126 | 4% | struct arithmetic from `SIMPLE`'s allocations; the measured forced-GC HWM rise over the run is 121 MB, of which PETSc KSP vectors are 55, leaving 66; they are garbage once `SIMPLE` returns, so live bytes after the run do not show them |
| setup transient (not resident after) | +205 | 325 | HWM | HWM during U equation construction above its settled RSS |
| GC slack | ~80 | 121 | 4% | natural peak 1950 minus forced-GC peak 1870 |

PetscMalloc total 234.6 MB after setup, 290.1 MB after 5 iterations (KSP work vectors appear at the first solve). Julia live after the run 716 MB against 707 MB summed by stage. The raw `size fields.momentum` (427 MB) and `fields.momentum+aux` in the logs are `summarysize` reaching the mesh through a path `exclude` misses; they are not field costs, and the table uses live deltas.

Operator copies per equation, counting one CSR (55 MB) as one: U 3.9 (Julia A0 and A, PETSc matrix, COO map), p 2.9 (Julia A, PETSc matrix, COO map). Ghost rows are 0.3 percent of local rows at n=2, so owned-row storage alone saves almost nothing at this rank count; it grows with rank count as the surface-to-volume ratio rises.

## Rank-0 global mesh (serial, before partitioning)

| mesh | RSS after load MB | per cell above runtime | peak during `partition_mesh` MB |
|---|---|---|---|
| 5 mm | 958 | 0.86 KB | 1239 |
| 4 mm | 1683 | 0.90 KB | 2389 |

## What each cure is worth at 4 mm n=2 (from the table)

- PETSc COO maps: 99 MB per rank, 34 percent of PETSc's heap, above the 15 percent bar in P1-M21-S2, so P1-M21-S3 goes ahead.
- Julia index arrays shared across A0, A and p (one `rowptr`/`colval` per mesh, D67): about 60 MB.
- Preallocated connectivity instead of triplets (D67): the 205 MB setup transient, which sets the peak only when setup is the high-water mark.
- Zero-copy PETSc vectors (P1-M21-S4): the x and b copies, about 20 MB, plus 12 copies per iteration of time.
- `GC.gc(true)` after setup (P1-M21-S5): up to 80 MB at n=2.
- The fixed ~800 MB per rank (runtime plus compiled code) is untouched by every planned cure and exceeds the whole per-cell cost below about 440k cells per rank.

## After P1-M21-S3 (host matrices without the COO map), 4 mm n=2 rank 0

PetscMalloc after the run 290.1 → 205.8 MB; natural peak RSS 1950 → 1793 MB; RSS after PETSc setup 1739 → 1548 MB; residual histories unchanged (`memory_breakdown/s3_4mm_gc*.tab`). The saving is 84 MB of PETSc heap rather than the full 99 MB because `MatCreateMPIAIJWithArrays` keeps its own row bookkeeping; the rest of the peak drop is the COO setup transient.
