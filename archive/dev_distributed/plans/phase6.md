# Phase 6 — Multi-GPU + CUDA-aware MPI (High; local machines only, not CI)

Umbrella: `distributed_plan_detailed.md` Phase 6. Per Q6: NO silent CPU fallback anywhere.

## Deliverables
1. `adapt(backend, dmesh)` — adapts local mesh + ProcessorPatch index arrays + halo
   buffers; `Adapt.@adapt_structure` for `DistributedMesh`, `Partition`, `ProcessorPatch`,
   `HaloExchange` (getproperty forwarding must survive adaptation — adapt the wrapped
   mesh, keep the wrapper).
2. Rank→device binding: `CUDA.device!(rank % ndevices)` helper called from `distribute`
   (or an explicit `bind_device!(dmesh)`); one MPI rank per GPU.
3. CUDA-aware path: device buffers passed straight to `Isend/Irecv!` when
   `MPI.has_cuda()`; otherwise pinned-host staging (built Phase 2). Detection once at
   HaloExchange construction.
4. PETSc GPU: `-mat_type mpiaijcusparse -vec_type cuda` when backend is `CUDABackend()`.
   If CUDA-enabled PETSc unavailable → HARD ERROR naming fixes (env vars, system PETSc,
   or explicit opt-in `prun!(...; solve_on=CPU())` which copies A/b host-side each solve).
5. Optional (time-permitting): interior-first overlap — process interior faces while halo
   messages fly, processor faces after `Waitall`.

## Cluster/env notes (docs + STATE)
- `MPIPreferences.use_system_binary()` for CUDA-aware MPI; `JULIA_CUDA_MEMORY_POOL=none`,
  `MPICH_GPU_SUPPORT_ENABLED=1`; verify `MPI.has_cuda()`.
- Local dev box: single RTX 4070 8GB → multi-rank single-GPU runs validate the code path
  (2 ranks sharing one device); true multi-GPU scaling measured on lab machines.

## Tests (local, gated out of CI)
- Cavity: multi-GPU (or 2-ranks-1-GPU) matches CPU-distributed and serial to 1e-5 (F64).
- Host-staging vs CUDA-aware paths produce identical results.
- Sanity scaling on a large-enough mesh: runtime decreases with GPUs, iterations identical.

## Exit criteria
GPU-distributed cavity green; explicit error paths verified (fake-missing CUDA PETSc →
error message names the remedy, per Q6 no-silent-fallback).
