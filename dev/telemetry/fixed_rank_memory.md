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

## S2: private MB added by loading a package alone in a fresh process

Script `dev/scripts/pkg_mem.jl one <Name>...` (smaps_rollup delta around `Base.require`; each number includes the package's own dependencies, so rows overlap). Bare Julia after start-up: 124 private. Raw: `dev/telemetry/memory_breakdown/m25s2_pkgs.txt`; the whole batch ran in 14 s.

- XCALibre (loads MPI, GPUArrays, LLVM, Graphs, Pkg already; 135 modules): 158 private, 35 shared.
- PETSc on top of XCALibre+MPI: +102 private (260 vs 158). Loads PETSc_jll, SCALAPACK32_jll and dlopens all ten libpetsc variants (single/double x real/complex x Int32/Int64, plus Int64 debug) although one is used; its package image is 87 MB. Largest removable row.
- PETSc alone 177, MPI 66, GPUArrays 95 (LLVM 70 of it), Graphs 60 (via KrylovPreconditioners), Pkg 57, KernelAbstractions 41, Krylov 28, Metis 20, StaticArrays 20, SparseArrays 16.
- Pkg comes in through MPI (PkgVersion), LLVMExtra_jll (LazyArtifacts) and PETSc directly, so XCALibre cannot drop it alone.

## S2: solver data vs compiler and allocator memory, n=4 rank 0 (`gc=1`, then `gc=1 trim=1`, `repeat=1`)

- GC live bytes at `runtime` → `iterations`: 23.5 → 50.7. Solver data added in Julia's heap is 27 MB; private memory grows 229 MB (313 → 542).
- A second `run!` adds 21 MB private: the first-run growth is one-off (compilation and its retention), not per-run.
- `malloc_trim(0)` after each GC returns 28 MB (`[heap]` 75 → 48; private at `iterations` 542 → 514). `[anon rw-p]` stays at 238 with 51 MB live, so about 187 MB of anonymous pages is neither live Julia data nor free malloc memory. Settled below: GC page retention.
- Transient: HWM 842 vs 803 RSS at `iterations`.
- `gclog=1`: Julia's GC reports bytes_resident 64 → 254 MB (`runtime` → `iterations`) against heap_size 211 and 51 MB live after a full collection. Pool pages grow in 64 MB blocks and are not returned.
- `--heap-size-hint=250M` on each rank (`gc=0`): `[anon rw-p]` 238 → 94, private at `iterations` 556 → 390 (−30 percent), RSS 817 → 652, HWM 701, hash unchanged. `t_iter_s` 11.0 → 12.7, which includes compilation; the time cost is not yet measured.

## S2 verdict

- Candidate cures, by attributed private MB per rank at n=4: (1) GC page retention, about 165 (the heap-size target lets pool pages dirtied by compilation and setup churn stay resident); (2) PETSc.jl loading every libpetsc variant and its 87 MB wrapper image, 47 private per rank at n=4 (the rest of the single-process +102 is file pages that are shared across ranks), and the mechanism lives in PETSc.jl, not XCALibre; (3) GPUArrays/LLVM on CPU runs, about 70-95 but a direct XCALibre dependency, so outside this milestone. Pkg and `sys.so` (91) are not removable by XCALibre.

## S3: GC memory target set at runtime (refused, D105)

`mem_probe.jl gcmax=<MB>` calls `jl_gc_set_max_memory` before setup; 10 mm n=4, 20 iterations, `repeat=2`, rank 0, private MB at `iterations` / after a second `run!`, and `run!` seconds. Raw: `dev/telemetry/memory_breakdown/m25s3_*_i20.txt`.

- base: 530 / 472, 1.11 s; base repeated: 564 / 500, 1.17 s (run-to-run noise about 30 MB).
- gcmax 330: 528 / 528, 1.25 s. gcmax 250: 479 / 459, 1.30 s; HWM 826 → 760.
- At 2 iterations gcmax 250 gave 425 against 564, but the startup flag gave 390: a target set after `using` cannot reclaim the pool pages already dirtied by loading.
- The retention is a first-run transient that the stock GC drains; at steady state the runtime target buys nothing measurable and costs 11-17 percent per `run!`. Julia 1.13 also reads `JULIA_HEAP_SIZE_HINT` from the environment, the launch-time equivalent of the flag.

## S4: precompile upper bound (case-specific statements)

`--trace-compile` on each rank of a 10 mm n=4 worker: 577 signatures and 18.2 s of compilation per rank (KernelAbstractions 6.7 s over 113, XCALibre 6.0 s over 70 with `SIMPLE` alone 4.4 s, Base 2.2, Core 2.1, MPI 0.3). Deduplicated over ranks and without `Main.`: 937 statements. 532 carry a baked kernel size: 256 are a rank's local cell or boundary-face count, which differs per rank and mesh, and 276 are `StaticSize{(1,)}`; 52 are typed on this case's physics or boundary conditions. They were evaluated at precompile time in a throwaway package loaded before setup (`mem_probe.jl pre=1`, env `dev/petscenv_m25pre`, not committed): 937 compiled, 31 MB image, package precompile 14 s. Raw: `dev/telemetry/memory_breakdown/m25s4_*.txt`.

- 2 iterations: private at `iterations` 559 → 359 (−36 percent), RSS 820 → 622, GC live 213 → 61, `t_iter_s` 11.46 → 0.04, worker wall 24 → 6 s, hash `dbc3c69ab48b3394` unchanged.
- 20 iterations: private 566 → 358, `t_iter_s` 10.58 → 0.38, hash `550bb695b7dbab9c` unchanged.
- Generic subset only (353 statements: no rank-sized kernels, nothing typed on this case's physics or boundary conditions): private 572, `t_iter_s` 11.15, i.e. no gain. The saving lives entirely in the case-typed `SIMPLE` call tree and the sized kernels.
- Runtime compilation is the cause of the first-run retention. The bound holds only for the exact mesh, rank count and boundary conditions traced; a shippable version needs kernel types that do not carry the launch size, plus a workload covering the supported cases.

## S4: the documented per-case recipe, run literally (D108)

The three steps of "Precompiling a production case" in the distributed guide, run as written in a fresh copy of `dev/petscenv_stock` (`dev/petscenv_recipe`, not committed) with `mem_probe.jl` as the case script; XCALibre developed from the checkout since the branch is unregistered. Trace run 19 s, 937 statements (936 compile, 1 returns false), package precompile 13 s. Raw: `dev/telemetry/memory_breakdown/m25s4_recipe{,_off}.txt`.

- Without / with the package: private at `iterations` 556 → 359 (−35 percent, bar 389), RSS 817 → 621, `t_iter_s` 12.09 → 0.04, worker wall 22 → 7 s, hash `dbc3c69ab48b3394` both. `using XCALibre` 0.98 s stock vs 0.91 s in the recipe env.
- Traps found and fixed in the recipe: `Pkg.develop` without `preserve=Pkg.PRESERVE_ALL` upgraded PETSc.jl (whose newer version ships only Int64 libraries) and the traced `SIMPLE` signature (Int32 PETSc) no longer matched, leaving 3.5 s of compilation; `Pkg.add` inside the new package wrote `[compat]` pins at the newest versions, which conflict with the case env, so its dependencies are copied from the case env's `Project.toml` instead.
