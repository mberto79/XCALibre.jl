# Fixed per-rank memory (P1-M25)

Machine: this laptop, `dev/petscenv_stock` (PETSc_jll 3.22), Julia 1.13, CPU backend, laminar BFS at 10 mm with Jacobi on U and p, 2 SIMPLE iterations, offline parts, `gc=0`. Script: `dev/scripts/mem_probe.jl` (smaps_rollup per stage; top 25 `/proc/self/smaps` paths by private bytes at `runtime` and `iterations`, anonymous regions keyed by permissions). Raw: `dev/telemetry/memory_breakdown/m25s1_10mm_n{1,4}.txt`. Rank 0 shown; ranks agree within 3 MB. Wall 16 s (n=1) and 20 s (n=4).

## S1: shared vs private, MB per rank (smaps_rollup)

- n=1 runtime: RSS 576, PSS 465, private 407 (clean 80, dirty 327), shared 170.
- n=4 runtime: RSS 572, PSS 373, private 324 (dirty), shared 248 (clean). The node holds the 248 once, so four ranks cost 4x324 + 248 = 1544 MB, not 4x572 = 2288.
- n=4 iterations: RSS 817, PSS 608, private 556, shared 261. Over setup and two iterations private grows by 232 MB, shared by 13.
- Private clean 80 at n=1 turns shared at n=4: file pages only this process mapped (libpetsc variants, MPI, OpenBLAS).

## S1: private bytes by mapping, rank 0 at n=4 (runtime → iterations)

- `sys.so` (Julia system image; relocated data pages): 90.5 → 90.8 private, 76 shared.
- GC heap (`[anon rw-p]`): 78.3 → 223.3. gc_live 65 → 127, so about 100 MB at iterations is GC pages not returned.
- PETSc.jl package image (87 MB file; wrappers for every configured libpetsc): 46.8 private + 31.4 shared; 78.2 private at n=1.
- Other package images (XCALibre 10.6, LLVM.jl 11.5, StaticArrays 7.6, Krylov 6.9, SparseArrays 5.8, Graphs 4.7, GPUArrays, DataStructures, ...): 48.9 → 49.3.
- malloc heap (`[heap]`): 25.0 → 104.7 (PETSc objects plus LLVM/codegen working memory).
- Pkg package image (loaded, never used by a run): 17.3 private + 12.4 shared.
- JIT machine code (`[anon r-xp]`): 2.2 at iterations. Runtime compilation costs heap, not code pages.
- libpetsc: five library variants mapped (Int32, Int64, Int64 debug, complex Int32, ...); 7.7-10.8 private at n=1, shared at n=4.

## S1 verdict

- The 790 MB fixed cost of D99 is about 250 MB shared per node plus about 540 MB private per rank at 10 mm; PSS lowers the apparent n=4 cost by 35 percent at `runtime`.
- Private after `using` + `MPI.Init` (324): Julia runtime (`sys.so` 91, GC heap 78, malloc 25) about 195, which XCALibre cannot remove; package images about 115, of which PETSc.jl 47 and Pkg 17 are the largest removable rows.
- The private growth over the first run (232) is GC heap (+145) and malloc (+80), not code: compiler working memory and unreturned GC pages. A precompile workload that removes inference and codegen from the run is the largest single candidate; S2 must separate compiler heap from solver data before S3 picks it.
