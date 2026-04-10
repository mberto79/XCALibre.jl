  In src/Solve/AMG/, we have a CPU-based AMG solver. The  per-iteration bottleneck is _galerkin_update! in          
  AMG_6_api.jl, which for large problems (e.g. 1.7M DOF fine
   level) is slow because _spgemm_nzval! in                 
  AMG_3_galerkin.jl scatters into a dense scratch array of
  size ncols_B × nthreads — for the fine level that's ~586K
  × nthreads floats per thread, which exceeds L2/L3 cache
  and causes heavy cache misses.

  Please implement both of the following optimisations:

  1. Compact accumulator for _spgemm_nzval!                 
   
  Currently cpu_tmps is sized ncols_B × nthreads and each   
  thread scatters into random positions within a
  586K-element slice. At setup time, the sparsity pattern of
   C is already known (fixed throughout the run). Build a
  per-row col_to_local::Vector{Int32} index map at setup
  time (stored in LevelExtras alongside cpu_tmps) that maps
  each column index j appearing in row i of C to a compact
  local index 0..nnz_row_i-1. Then resize cpu_tmps to
  max_nnz_per_row(C) × nthreads (typically ~14 for FVM
  Laplacian vs 586K now). In _spgemm_nzval!, use
  col_to_local to scatter into the compact buffer instead of
   the full-width one. This keeps the accumulator in L1
  cache.

  2. GPU-native Galerkin update (CUDA extension)            
   
  Currently _galerkin_update! does a CPU round-trip:        
  download A.nzval from device → CPU SpGEMM → upload
  Ac.nzval to device. For GPU runs this is a bottleneck. Add
   a GPU-native path in ext/XCALibre_CUDAExt.jl using
  cuSPARSE SpGEMM (CUSPARSE.spgemm or
  CUSPARSE.CuSparseMatrixCSR multiply) so the Galerkin
  products stay on-device. The CPU path should remain as
  fallback for non-CUDA backends.

  The _galerkin_update! function is in AMG_6_api.jl (around 
  line 414). LevelExtras is defined in AMG_0_types.jl. The
  CUDA extension is in ext/XCALibre_CUDAExt.jl. Check the   
  existing _build_sparse_device dispatch there as a pattern
  for adding CUDA-specific overloads.