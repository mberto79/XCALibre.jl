# Audit: XCALibre.jl distributed (MPI) implementation

Branch `HM/distributed-draft` at `814d5b36` (82 commits over `main`), read 2026-09-18. Scope: `src/Distribute/`, `ext/XCALibrePETScExt.jl`, the GPU extensions, the seams added to `src/Solve`, `src/Solvers`, `src/Calculate` and the turbulence models, the distributed tests, and the telemetry in `dev/`. Method: code read plus the recorded measurements; nothing was re-run. Every claim below cites the line it comes from or the telemetry file that measured it. Where a cure names a PETSc or MPI call I have not exercised on this stack it says "verify".

## Verdict

The module is correct, well-measured and cleanly layered for the range it was built on: one node, up to eight CPU ranks, two ranks on one GPU, meshes to about 1.3M cells. Inside that range it ties OpenFOAM's GAMG per iteration with Jacobi and scales better (`SCALING_SUMMARY.md` §6). It is not yet a platform for massive parallelism, and the reasons are structural rather than bugs:

1. Every mesh passes through one rank, serially, at O(P·N) time and O(N) memory. This is a hard ceiling, not a slow path.
2. The linear system exists in three to four copies per rank and every cell kernel also runs over ghost rows. Peak memory is 4.1 KB per cell per rank before any AMG, against roughly 1.5 to 1.8 KB of live data by struct layout.
3. Communication is synchronous and unfused: eight blocking exchange rounds, eight residual all-reduces and twelve device-wide synchronisations per SIMPLE iteration, on top of PETSc's own.

Fixing the first makes scale possible; the second and third make it fast. Separate from those, six items block a production release (segfaults, untested multi-node and multi-GPU, device binding, memory bounding) and are listed on their own so they are not lost inside the structural work.

## Top 3 changes

### 1. Replace the rank-0 serial preprocessing with a parallel partition and distribution pipeline

**What changes.** Rank 0 no longer reads the global mesh, runs Metis, or builds every subdomain. Each rank reads a block of the mesh, a parallel partitioner assigns cells, cells migrate once, and ghosts and processor patches are built from an exchange of face-owner lists. Parts persist in a versioned binary format that any Julia version can read.

**Evidence from this branch.**
- `distribute(mesh)` runs `partition_cells` then `extract_subdomain` once per rank, sequentially, and `MPI.send`s each result ([Distribute_1_partition.jl:284-296](src/Distribute/Distribute_1_partition.jl#L284-L296)). `partition_mesh` is the same loop writing files ([Distribute_1_partition.jl:350-355](src/Distribute/Distribute_1_partition.jl#L350-L355)).
- Each `extract_subdomain` is O(N) whatever the part size: the owned scan ([:84](src/Distribute/Distribute_1_partition.jl#L84)), the full-face scan for interior faces, `part_counts` and `pos` rebuilt for every part ([:191-198](src/Distribute/Distribute_1_partition.jl#L191-L198)), and three `Dict{Int,TI}` maps ([:95](src/Distribute/Distribute_1_partition.jl#L95), [:111](src/Distribute/Distribute_1_partition.jl#L111)). The processor-patch loop is O(neighbours × interior faces) per part ([:207-216](src/Distribute/Distribute_1_partition.jl#L207-L216)). Total O(P·N) time on one rank.
- `build_dual_graph` grows Int64 `I, J` by `push!` ([:8](src/Distribute/Distribute_1_partition.jl#L8)), so the graph alone is 32 bytes per interior face plus growth slack, before Metis makes its own copy.
- Rank 0 holds the global mesh at about 1.6 KB per cell (`dev/gotchas.md`); the 14 GB laptop was OOM-killed at 1.3M cells and eight ranks with AMG (`dev/telemetry/preconditioner_guidance.md`).
- Parts are Julia `serialize` files that the docstring itself says must be regenerated after a Julia or XCALibre upgrade ([:348](src/Distribute/Distribute_1_partition.jl#L348)).
- The only reader that can build a mesh in pieces is the OpenFOAM one, and there is no reader for an already-decomposed `processor*/` case, although the writer produces exactly that layout ([Distribute_7_io.jl](src/Distribute/Distribute_7_io.jl)).

**Cost.** This is the largest item: a parallel reader (decomposed OpenFOAM first, since it needs no partitioner and the writer already emits it), a parallel partitioner (PETSc's `MatPartitioning` with ParMETIS or PT-Scotch if the build has one, else a two-level scheme: coarse Metis on a sampled graph plus local refinement; verify what PETSc.jl wraps), an `Alltoallv` cell migration, a ghost-construction exchange, and a binary part format with a header. Three to four milestones. A cheap intermediate step is worth landing first: make `extract_subdomain` O(N + P) total by bucketing cells per part once, computing `part_counts` and `pos` once, and replacing the three `Dict`s with `Vector{TI}` inverse maps sized N. That alone removes the P factor and roughly halves rank-0 transient memory, and it changes no output.

**Measurement.** Wall time and peak RSS of `partition_mesh` at 5M and 20M cells for P in 8, 64, 256, before and after; bitwise-identical parts for the intermediate step; the same residual history at n=2 for the parallel path.

**When it does not help.** Below about 5M cells and 64 ranks the offline `partition_mesh` path already works, its serial cost is paid once, and this change buys nothing for those users. It is the change for the users who cannot run at all today.

### 2. One copy of the linear system, owned rows only, and measured memory hygiene

**What changes.** The rank-local CSR holds owned rows only, with columns indexed over owned then ghost cells (the ordering the partition already produces). PETSc reads that matrix and the owned prefix of every vector in place. Sparsity is built once per mesh and shared by every equation. Transient allocation at setup is bounded, and per-rank heap size is under control.

**Evidence from this branch.**
- Matrix multiplicity. Momentum holds `A0` and `A` as two full CSRs including ghost rows ([ModelFramework_0_types.jl:193-194](src/ModelFramework/ModelFramework_0_types.jl#L193-L194)); PETSc holds its own `mpiaij` copy plus the COO permutation arrays that `MatSetPreallocationCOO` keeps for the life of the matrix ([XCALibrePETScExt.jl:145](ext/XCALibrePETScExt.jl#L145)). That is three and a half to four copies of the momentum matrix and two and a half to three of the pressure matrix per rank. On a GPU every one of them is in HBM.
- Vector copies. `psolve!` copies the owned prefix into PETSc's `x` and back out ([:210-216](ext/XCALibrePETScExt.jl#L210-L216)), and `passemble!` copies `b` ([:201-207](ext/XCALibrePETScExt.jl#L201-L207)): three vector copies per component solve, twelve per SIMPLE iteration. The owned entries are the contiguous prefix `1:n_owned` of every XCALibre array, which is exactly what `VecCreateMPIWithArray` and its CUDA variant need for a zero-copy wrap (verify the LibPETSc surface in PETSc.jl 0.4).
- Ghost rows. `discretise!` ([Discretise_2_generated_distretisation.jl:31](src/Discretise/Discretise_2_generated_distretisation.jl#L31), [:116](src/Discretise/Discretise_2_generated_distretisation.jl#L116)), `green_gauss!` ([Calculate_1_green_gauss.jl:12](src/Calculate/Calculate_1_green_gauss.jl#L12)), `div!` ([Calculate_0_divergence.jl:34](src/Calculate/Calculate_0_divergence.jl#L34)), `inverse_diagonal!` ([Solvers_0_functions.jl:102](src/Solvers/Solvers_0_functions.jl#L102)) and `H!` ([:194](src/Solvers/Solvers_0_functions.jl#L194)) all run over every local cell, producing rows the comments call "garbage by design" and that `sync!` then overwrites. At 1 to 4 percent ghosts (the n=8 BFS numbers in `dev/telemetry/tolerance_semantics_and_reductions.md`) this is noise; at 100k tets per rank it is 10 to 20 percent of every cell kernel and of the CSR.
- Transient peaks. `sparse_matrix_connectivity` grows `i, j` by `push!` ([:223-241](src/ModelFramework/ModelFramework_0_types.jl#L223-L241)), `_build_A` then runs `sparsecsr` twice from the same triplets for a `VectorEquation`, and every equation (U, p, k, ω, y) repeats the whole construction including `adapt(CPU(), mesh)` ([:139](src/ModelFramework/ModelFramework_0_types.jl#L139)), which on a GPU copies the entire mesh back to the host once per equation.
- Measured against estimate. Peak RSS is 2.79 and 2.67 GB on the two ranks at 660k cells each, about 4.1 KB per cell, with Jacobi (`dev/telemetry/preconditioner_guidance.md`). By the struct layouts in [Mesh_0_types.jl](src/Mesh/Mesh_0_types.jl) a tet mesh with Int64 indices is roughly 0.55 KB per cell, and fields, gradients, work arrays and the two equation systems add roughly 1.0 to 1.2 KB, so live data is about 1.5 to 1.8 KB per cell. The rest is transient construction garbage and GC slack. This is an estimate, not a measurement; P1-M16's breakdown is the right first step and is already planned.
- Heap control. Each Julia rank sizes its collector against the whole node with no knowledge of its siblings, and `dev/gotchas.md` records that `--heap-size-hint` cannot be used because it changes the precompile cache flags. Re-verify that claim: if the precompile run and the ranks use the same flag the cache should match, and the cure is simply to precompile under the launch flags. Until a per-rank heap bound works, a node full of ranks will over-commit exactly as the 14 GB box did.

**Cost.** Owned-row CSR touches shared serial code: the equation constructors, the `ndrange` of six kernels and the `spindex` lookups, so it belongs on the single shared-code memory PR that D67 already reserves. The PETSc side is local to the extension: split-array or `MatUpdateMPIAIJWithArray`-style value updates in place of COO (verify which are wrapped), zero-copy vectors, and one shared sparsity per mesh. The GC work is measurement plus a documented launch recipe.

**Measurement.** Per-rank peak RSS and `Base.gc_live_bytes()` after setup and after iteration 10, PETSc `-memory_view`, at two mesh sizes and n=2 and n=8; per-iteration time unchanged or better; residuals bitwise identical for the CSR change with Jacobi.

**When it does not help.** On CPU nodes with 256 GB and 100k cells per rank memory is not the binding constraint and the copies cost only bandwidth already shown to be in cache. This change decides the maximum cells per GPU and whether a laptop or a small node can run a 5M-cell case at all.

### 3. Overlapped, fused, stream-aware communication, with node-local device binding

**What changes.** One halo schedule per mesh, shared by every equation, with distinct tags and persistent requests. Fewer exchange rounds per iteration by fusing fields whose dependencies allow it. Interior cells computed while boundary-adjacent messages are in flight. One fused all-reduce for all residuals. On GPUs, stream-ordered handoff to PETSc instead of device-wide synchronisation, and device binding by node-local rank.

**Evidence from this branch.**
- Every `sync!` is a full blocking round: Irecv, pack, device sync, Isend, Waitall, unpack, device sync, Waitall ([Distribute_2_halo.jl:117-141](src/Distribute/Distribute_2_halo.jl#L117-L141)). No interior/boundary split, so nothing overlaps.
- Rounds per laminar SIMPLE iteration, from the call sites: three width-1 exchanges for U, one per component solve ([Distribute_5_solvers.jl:51-63](src/Distribute/Distribute_5_solvers.jl#L51-L63), [:74](src/Distribute/Distribute_5_solvers.jl#L74)); `rD` ([Solvers_0_functions.jl:106](src/Solvers/Solvers_0_functions.jl#L106)); `Hv` ([:199](src/Solvers/Solvers_0_functions.jl#L199)); `p` after its solve; `p` again after `explicit_relaxation!` ([Solve_1_api.jl:336](src/Solve/Solve_1_api.jl#L336)); `∇p` after `grad!` ([Calculate_0_gradient.jl:109](src/Calculate/Calculate_0_gradient.jl#L109)). Eight rounds, plus one per limiter call and per non-orthogonal corrector, plus the turbulence equations.
- Three of those rounds can be one. The momentum matrix is assembled once from the U synced at the start of the iteration, and the y and z solves read only owned rows and boundary faces, so a single width-3 exchange after the third solve is equivalent, provided the three residuals are computed after it rather than inside each `solve_system!`. `rD` and `Hv` both read only the assembled momentum matrix and the U ghosts that exchange just filled, so they can share one four-wide exchange. Eight rounds become five with no change in results.
- `residual` issues two `Allreduce` per component ([Distribute_5_solvers.jl:88-89](src/Distribute/Distribute_5_solvers.jl#L88-L89)): eight per iteration that can be one eight-element reduction, on top of PETSc's roughly two per Krylov iteration after D54.
- `wrap_eqn` builds a fresh `HaloExchange` for every equation ([:27](src/Distribute/Distribute_5_solvers.jl#L27)) while the mesh already caches width-1 and width-3 schedules ([Distribute_3_fields.jl:50](src/Distribute/Distribute_3_fields.jl#L50), [:57](src/Distribute/Distribute_3_fields.jl#L57)): duplicated device index arrays and buffers, and two schedules for the same neighbours. All messages use `tag=0` ([Distribute_2_halo.jl:119](src/Distribute/Distribute_2_halo.jl#L119)); safe while one exchange is in flight at a time, and a constraint any overlap must lift first.
- GPU synchronisation. Each component solve pays `CUDA.device_synchronize` three times: before `MatSetValuesCOO`, and before and after `KSPSolve` ([XCALibrePETScExt.jl:195-233](ext/XCALibrePETScExt.jl#L195-L233), [XCALibre_CUDAExt.jl:36](ext/XCALibre_CUDAExt.jl#L36)). Twelve device-wide drains per iteration because PETSc and CUDA.jl run on different streams. Each halo adds two `KernelAbstractions.synchronize` ([Distribute_2_halo.jl:126](src/Distribute/Distribute_2_halo.jl#L126), [:138](src/Distribute/Distribute_2_halo.jl#L138)); the second is unnecessary, since the next kernel is stream-ordered behind the unpack. The GPU iteration is 0.075 s at 500k cells (`dev/telemetry/device_resident_petsc.md`), so these drains are a measurable fraction. The cure is to record an event on XCALibre's stream and have PETSc wait on it, or run both on one stream (verify `PetscDeviceContext` access through PETSc.jl), not to reduce solves.
- Device binding uses the global rank modulo the device count ([XCALibre_CUDAExt.jl:33](ext/XCALibre_CUDAExt.jl#L33)). No `Comm_split_type` exists anywhere in `src`, `ext` or `examples`. On more than one node this is right only when ranks fill nodes contiguously with exactly as many ranks per node as GPUs; otherwise two ranks share a device and others sit idle. The node-local rank from `MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)` is the fix, with an error when ranks per node exceed devices.
- Load balance. Metis balances owned cells to 1.001 while ghost counts span 654 to 2373 at n=8 (D59); no vertex weights for wall-function or boundary cells. `-log_sync` shows the solve itself balanced (D59), so what is left is arrival spread at reductions, which fusing and overlap attack directly.

**Cost.** Two to three milestones inside `src/Distribute/` plus small `ndrange` changes so kernels can run over an index subset. The fused U exchange and fused residuals are a day each. Overlap needs a boundary-adjacent cell list computed at partition time.

**Measurement.** Rounds and all-reduces counted per iteration before and after; per-iteration time at n=8 with the clock pinned and at n=2 on the GPU; PETSc `-log_view` `VecNorm` share and ratio; residuals bitwise identical with Jacobi.

**When it does not help.** At two to four ranks with 500k cells each the rounds are about one percent of the iteration and the change is invisible. It pays at 64 ranks and beyond, and on GPUs at any rank count, where a device drain costs more than a whole halo exchange.

## Release blockers

These are not structural and should not wait for the three changes above.

1. **Unexplained segfault on the CUDA-aware direct path.** One run in four at n=2 on Open MPI with `OMPI_MCA_opal_cuda_support=true` (`dev/telemetry/conda_cuda_petsc.md`). Root-cause before release: run under `compute-sanitizer` and `CUDA_LAUNCH_BLOCKING=1`, then isolate with `cuda_aware=false` on the halo only, so the fault is attributed to the halo, to PETSc's scatter, or to two ranks sharing one device through CUDA IPC.
2. **`BoomerAMG()` on GPU fields with a host-only hypre segfaults instead of erroring** (D70). Detect at setup or refuse `BoomerAMG` with device fields unless the user opts in; the current check only tests that hypre exists ([XCALibrePETScExt.jl:167](ext/XCALibrePETScExt.jl#L167)).
3. **Nothing beyond two ranks is gated, and nothing beyond one node has ever run.** The gate is five files at n=2 ([gate.jl:7](test/distributed/gate.jl#L7)); the GPU test binds two ranks to one device ([test_gpu.jl:21](test/distributed/test_gpu.jl#L21)); AMD is mirrored and unverified. The documentation's first sentence promises multi-node and multi-GPU ([6_distributed_mpi.md:3](docs/src/user_guide/6_distributed_mpi.md#L3)). At minimum: n=3 and n=5 in the gate (odd counts expose asymmetric neighbour lists), an oversubscribed n=8 nightly, and one real multi-node, multi-GPU run recorded in telemetry before the claim stays in the docs.
4. **Device binding by global rank** (change 3 above) is wrong on the configuration the docs advertise.
5. **Per-rank heap cannot be bounded** while `--heap-size-hint` is off the table (`dev/gotchas.md`). Re-verify the cache-flag claim; if it holds, document the precompile-under-launch-flags recipe; if not, a node full of ranks will over-commit.
6. **`.jls` parts are version-fragile** and fail after any upgrade, sometimes with a type error deep in the solver rather than at load ([Distribute_1_partition.jl:348](src/Distribute/Distribute_1_partition.jl#L348)). A header with Julia, XCALibre and format versions checked at load is the minimum; a binary format is the fix.
7. **The branch carries 82,933 lines of committed simulation output.** `docs/1/U`, `k`, `nut`, `omega`, `p` and `y` are OpenFOAM field files a run dropped into `docs/`, and `PLAN_REVIEW.md`, `SCALING_SUMMARY.md` and `distributed_plan_detailed.md` sit at the repository root. They must leave the branch, or move under `dev/` or `archive/`, before a pull request is reviewable.
8. **`initialise_writer(::VTK, ::DistributedMesh)` returns `nothing`** ([Distribute_7_io.jl:217](src/Distribute/Distribute_7_io.jl#L217)), so a distributed run asked for VTK output silently writes nothing. It should warn or error.

## What is solid

Keep these as they are; they are the reason the module is correct.

- The seam design: `sync!`, `wrap_eqn`, `unwrap_eqn`, `is_distributed_mesh`, `global_max` are inlined identities in serial and the solver bodies are shared line for line ([Solve_1_api.jl:511-530](src/Solve/Solve_1_api.jl#L511-L530)).
- Owned-row discipline: canonical min/max owner row in `make_symmetric!` ([Solve_1_api.jl:555](src/Solve/Solve_1_api.jl#L555)) and `correct_mass_flux!` ([Solvers_1_SIMPLE.jl:411](src/Solvers/Solvers_1_SIMPLE.jl#L411)); residual over owned rows only; reference cell by original global id.
- Flux consistency across processor faces is bitwise by construction: `extract_subdomain` preserves original owner order through `g2l`, so both ranks compute the same interpolation and the same `aN·(p2 − p1)`. This is why Jacobi runs agree to 15 significant figures across rank counts (`SCALING_SUMMARY.md` §8).
- Rank-uniform API: `distribute(reader)` removes the rank-guard class of hang that broke the BFS example (archived findings).
- In-place COO value assembly from the `nzval` pointer, on host or device (D51), and the narrowest-index PETSc library chosen from an all-reduced count (D61).
- Tolerance semantics matched to Krylov.jl: `rtol` relative to the initial residual and CG on the natural norm ([XCALibrePETScExt.jl:156](ext/XCALibrePETScExt.jl#L156), [:180](ext/XCALibrePETScExt.jl#L180)), which cut 30 SIMPLE iterations' final residual by 10 to 30x (D53).
- The measurement discipline: pinned clocks, withdrawn attributions recorded as such, `-log_sync` used to separate wait from work.

## Findings by area

### Correctness

- Ghost consistency is enforced by convention, not by construction. Each primitive that must sync carries a comment; nothing asserts it. A debug-mode `sync!` that exchanges into a scratch buffer and asserts ghosts equal owner values, run once per primitive in the gate, would catch a missing sync when LKE, LES, energy or the compressible solvers are wired. The SST `nut` sync ([RANS_kOmegaSST.jl:323](src/ModelPhysics/Turbulence/RANS_kOmegaSST.jl#L323)) was found by reasoning, not by a test.
- Tensor ghosts are declared unconsumed ([Distribute_3_fields.jl:62](src/Distribute/Distribute_3_fields.jl#L62)). True today: `gradU` ghost values are garbage, and only pointwise cell quantities are derived from them before anything is interpolated to a face. Any future model that interpolates a quantity derived from a tensor in ghost cells (an LES eddy viscosity from the strain rate is the obvious one) will be silently wrong at processor faces. The debug check above is the guard.
- `comm` is not propagated. `distribute(mesh; comm)` accepts a communicator, but the cached schedules ([Distribute_3_fields.jl:50](src/Distribute/Distribute_3_fields.jl#L50)), `wrap_eqn` ([Distribute_5_solvers.jl:27](src/Distribute/Distribute_5_solvers.jl#L27)) and `global_max` ([:110](src/Distribute/Distribute_5_solvers.jl#L110)) all use `COMM_WORLD`. Not a bug today; it rules out sub-communicator runs (ensembles, coupled solvers, one job running several cases). Store the communicator on the `DistributedMesh`.
- `_gpu_comm!` writes PETSc's exported global `use_gpu_aware_mpi` through `dlsym` ([XCALibrePETScExt.jl:102](ext/XCALibrePETScExt.jl#L102)). It works on PETSc 3.24 and 3.25; it is a private symbol and will break silently or crash on a build that renames or removes it. Guard with a version check and a clear error.
- `setReference!` scans `orig_cells` with `findfirst` on every pressure solve ([Distribute_5_solvers.jl:99](src/Distribute/Distribute_5_solvers.jl#L99)); cache the local id at wrap time.
- `quiet_nonroot!` replaces the user's global logger ([Distribute_1_partition.jl:265](src/Distribute/Distribute_1_partition.jl#L265)). Acceptable, but a user who installed their own logger loses it without notice.
- `HaloCache` fields are `Any` ([Distribute_0_types.jl:50-53](src/Distribute/Distribute_0_types.jl#L50-L53)); each `sync!` pays a dynamic dispatch. Deliberate function barrier; fine at a few calls per iteration, worth typing when the schedule becomes a single shared object.
- Periodic handling is sound: contraction colocates pairs; the empty-patch path ([periodic.jl:112-121](src/Discretise/boundary_conditions/periodic.jl#L111-L121)) covers a rank with no periodic faces. Merged super-vertices are unweighted (the code's own "ponytail"), so imbalance grows with the periodic fraction of the mesh.

### Scalability ceilings

- Serial preprocessing (change 1). Nothing else matters until this is gone.
- Cell-count-only partitioning with no weights and a one-layer ghost. Adequate for second-order schemes with synced gradients; a scheme needing second neighbours would need a two-layer halo, which `extract_subdomain` cannot build.
- Pure MPI, one thread per rank, BLAS pinned to one ([`activate_multithread`], `dev/gotchas.md`). On 128-core nodes that is 128 ranks per node and a ghost fraction and reduction cost to match. Hybrid MPI plus KernelAbstractions threads is possible (`Multithread.jl:22` already sizes on `Threads.nthreads()`) but is recorded as slower today; it becomes worth revisiting once communication is overlapped.
- `gather` reconstructs whole fields on rank 0 ([Distribute_7_io.jl:4-22](src/Distribute/Distribute_7_io.jl#L4-L22)); any post-processing built on it inherits the rank-0 ceiling.

### Memory

Four distinct sources, in decreasing order of what they cost at scale:

1. Rank-0 global mesh and O(P·N) extraction (change 1).
2. Linear-system multiplicity and ghost rows (change 2).
3. Transient setup allocation: triplet growth, double `sparsecsr`, one construction per equation, `adapt(CPU(), mesh)` per equation on GPU (change 2).
4. Collector behaviour under MPI: one heap heuristic per rank sized to the node (blocker 5).

Two numbers exist: 4.1 KB per cell peak per rank and 1.6 KB per cell for the rank-0 mesh. Everything else above is an estimate from struct layouts and should be replaced by P1-M16's breakdown before any cure is designed.

### Communication and reductions

Covered by change 3. Two smaller items:

- Halo host mirrors are allocated only when MPI is not CUDA-aware ([Distribute_2_halo.jl:33](src/Distribute/Distribute_2_halo.jl#L33)); correct, and the branch is decided once at construction from `MPI.has_cuda()`, which is only valid after `MPI.Init` (`dev/telemetry/conda_cuda_petsc.md`). `distribute` initialises MPI first, so the order holds, but a user constructing a `HaloExchange` before `distribute` would get the staged path silently.
- `PipeCG` lost 17 percent at eight ranks (`SCALING_SUMMARY.md` §7). Pipelined and communication-avoiding Krylov are for 64 ranks and beyond; do not revisit before change 3 lands.

### GPU

Evidence is thin: one RTX 4070, two ranks sharing it, one mesh. Nothing below is a measured multi-GPU claim.

- Twelve `device_synchronize` per iteration and one redundant stream sync per halo (change 3).
- Global-rank device binding (blocker 4).
- The direct CUDA-aware path is the one with the unexplained segfault (blocker 1); the staged path is the one that has always passed.
- A GPU run needs a custom or conda PETSc; `PETSc_jll` has no CUDA. The documented conda route is good. A build-free GPU path would need a native distributed solver (below).
- AMD: `petsc_device_info(::ROCArray)` errors by design (D52); the halo path is mirrored and untested.

### Linear solver strategy

All distributed solves go through PETSc, and the module's own AMG is serial only; `SCALING_SUMMARY.md` §9 calls the native AMG the weak component. This is a pragmatic dependency, not a defect, and PETSc's GAMG on the GPU (0.059 s per iteration at 500k cells) is the right default today. For the long range, two paths are open and both are outside P1: distribute the native KernelAbstractions AMG over the halo machinery, which removes the matrix copies and the custom-PETSc requirement on GPUs; or AmgX through PETSc, which needs yet another build. A decision is not needed now, but change 2 should be designed so the XCALibre CSR can be the single operator either path consumes.

### Testing and validation

- Gate: n=2 only (blocker 3). Add n=3, n=5, an oversubscribed n=8, and one mesh where a rank owns zero faces of some patch (already covered for periodic, not for ordinary patches).
- Add the ghost-consistency assertion as a test, and a per-primitive sync count so a new redundant exchange fails the perf gate the way a new allocation does ([test_perf.jl](test/distributed/test_perf.jl)).
- Fields are compared against a serial reference at n=2 in the gate ([test_psimple.jl:12-21](test/distributed/test_psimple.jl#L12-L21)); agreement across rank counts (15 significant figures with Jacobi) is only checked by hand. Make it a test at n=1, 2 and 4 on the gate mesh.
- No restart test exists because no restart path exists (below).

### Portability, I/O, ergonomics

- The decomposed writer is ASCII with one `println` per value ([Distribute_7_io.jl:249](src/Distribute/Distribute_7_io.jl#L249), header at [:50](src/Distribute/Distribute_7_io.jl#L50)). Parallel per rank, so it scales, but at 10M cells per rank it is minutes per write. Binary OpenFOAM format is a small change.
- No distributed VTK (blocker 7), no checkpoint or restart, no field read-back. A production run of days needs restart; parallel HDF5 through HDF5.jl with MPI-IO is the standard answer and also solves the part format.
- `Project.toml` raises the Julia floor from 1.8 to 1.10 (weak dependencies need 1.9 or later). Neither the CHANGELOG nor `CLAUDE.md`, which still says 1.8+, records it.
- `petsc_options` string doubles as start-up options that apply on the first `PETSc.initialize` only ([XCALibrePETScExt.jl:123](ext/XCALibrePETScExt.jl#L123)); a user who sets `-log_view` on the second equation's options gets nothing and no warning.

## Suggested order

1. Blockers 1, 2, 4, 7, 8 (days each), then the O(N + P) `extract_subdomain` rewrite (a week, no output change).
2. P1-M16 memory breakdown as planned, then change 2 on the shared-code PR, with zero-copy PETSc vectors and shared sparsity in the extension.
3. Change 3: shared schedule and tags, fused U and residual exchanges, then overlap, then GPU stream handoff and node-local binding; a real multi-node, multi-GPU run recorded at the end.
4. Change 1 in full: parallel decomposed-OpenFOAM reader first, parallel partitioner second, binary part format and restart together.
