 Scoped to correctness and performance only, no new features. Findings
   split by impact area; within each section items are ranked by       
  expected impact.
                                                                       
  1. Correctness                                                       
  
  C1. _estimate_lambda_max underestimates λ(D⁻¹A) for Laplacian-like   
  operators — 1_AMG_setup.jl:143-165                                 
  Power iteration starts from v = ones(...) with only 3 iters. For FVM 
  Poisson / diffusion, the constant vector is (near-)nullspace of A, so
   D⁻¹A*ones ≈ 0. Three iterations from a near-nullspace seed give a
  poor estimate. Concrete failure mode: when the returned λ is clamped 
  up to 1 via max(lambda, one(T)), the Jacobi-ω cap min(ω, 4/(3λ))   
  becomes inactive and coarse levels can apply an ω that is too large
  for the actual spectrum. Fine-level Laplacian survives because ω=2/3
  is a safe default, but coarse Galerkin operators are at risk. Fix:
  random or Rademacher init + more iters, or a Gershgorin row-sum
  bound.

  C2. _diag_inverse silently defaults aii = 1 when a row has no        
  diagonal entry — 1_AMG_setup.jl:119-131
  No caller relies on this fallback (_allocate_level, _refresh_level!, 
  _smooth_prolongation only). A missing diagonal means a broken setup. 
  Replace silent fallback with error(...).
                                                                       
  C3. Ruge–Stüben direct interpolation row search is quadratic and has 
  a dead "safety" branch — 2_AMG_coarsening.jl:395-500
  For each F row, for each j in strong[i], the code linearly scans CSR 
  row i to find a_ij. That is O(|strong_i| × row_nnz). For moderately  
  dense strength graphs it is slow; not wrong, but worth flagging
  alongside the Vector{Vector} issue in P5. The isempty(strong_c)      
  fallback (line 439) is also dead after Pass 2, since Pass 2 promotes
  any F with no C strong neighbour. Safe, but remove for clarity.

  C4. amg_cg_solve! α and β have no divide-by-zero guard —             
  6_AMG_cg.jl:67,81
  SPD + nonzero p implies p'Ap > 0 so this is academic, but there is no
   guard if the Krylov space stagnates at floating-point level.        
  Mention, don't prioritise.
                                                                       
  C5. Duplicate level.x zero-fill — 5_AMG_cycle.jl:22 +                
  5_AMG_cycle.jl:43
  amg_apply_preconditioner! zeros root.x; _cycle! re-zeros level.x on  
  entry. Footnote-level redundancy.                                    
  
  2. Semantic / API surprises                                          
                                                                     
  S1. SolverSetup.preconditioner is a pre-AMG sweep, not "AMG's        
  preconditioner" — 7_AMG_update.jl:108                              
  solve_system! calls apply_smoother!(setup.smoother, values, A, b,    
  hardware) before AMG, so the user-supplied preconditioner=Jacobi() in
   examples/2D_cylinder_U_AMG.jl runs a Jacobi sweep against values and
   then AMG starts from that improved guess. This is reasonable        
  alignment with the generic path, but the user API suggests "AMG    
  preconditioner = Jacobi" which is not what is happening.
  Documentation gap, and arguably preconditioner should default to /
  allow nothing when solver isa AMG.

  S2. Jacobi ping-pong under device adapt — verified non-issue, but    
  worth keeping in the code as a comment
  _apply_level_smoother! swaps level.x and level.tmp pointers each     
  sweep. On GPU, hierarchy.levels[k] is a separately-adapted AMGLevel  
  so mutation does not touch host_levels[k]. _cycle! zero-fills level.x
   on entry, so cross-cycle pointer state does not affect correctness. 
  Document to preempt reviewer concern.                              

  S3. aggregate_ids is stored on every level but only read during      
  setup. Minor memory waste, footnote.
                                                                       
  3. Performance — CPU single-thread (the currently benchmarked path)  
  
  Status doc (2026-04-23) shows AMG-CG apply = 4.50 s / 344 applies,   
  i.e. the per-cycle cost dominates. These items target that hot path.
                                                                       
  P-CPU-1. _pattern_matches is O(nnz) per update! even on CPU —        
  7_AMG_update.jl:1-15
  Full elementwise equality of rowptr+colval against stored pattern on 
  every pressure solve. For 1.67M rows × ~7 nnz/row, that's ~12M       
  comparisons per update, per non-ortho corrector, per outer iter.
  Replace with: dims/nnz cheap check → pointer equality of the         
  underlying arrays → hash verified on setup.                        

  P-CPU-2. _estimate_lambda_max reallocates v, w per level per refresh 
  — 1_AMG_setup.jl:143-165
  Called from _refresh_level! for every level on every numeric refresh.
   Pre-allocate per-level scratch once at setup.                       
  
  P-CPU-3. _csr_to_csc runs every refresh on P and R inside            
  _regalerkin! — 1_AMG_setup.jl:256-262                              
  P and R are fixed between rebuilds. Cache their CSC form at setup;   
  only rebuild A's CSC each refresh.                                   
  
  P-CPU-4. _diag_inverse! allocates then copies —                      
  1_AMG_setup.jl:136-141                                             
  Have _diag_inverse! write directly into the destination buffers; drop
   the allocating _diag_inverse wrapper on the refresh path.           
  
  P-CPU-5. Vector{Vector{Int}} for strength graph, coarse graph,       
  aggregate adjacency, strong transpose — 2_AMG_coarsening.jl:1-106, 
  127-143, 315-324                                                     
  On rebuild, millions of tiny Vector{Int} allocations at 1.67M cells.
  Cache-unfriendly, heavy GC pressure. Replace with flat CSR (rowptr,  
  colval). Touches rebuild cost, not per-cycle apply, but rebuild is a
  non-negligible contributor under the adaptive-rebuild policy.        
                                                                     
  P-CPU-6. Coarse LU re-factorises from scratch each refresh —         
  1_AMG_setup.jl:282-294
  UMFPACK's lu! supports pattern reuse — re-use the symbolic factor and
   only refactor numerics when pattern matches.                        
  
  P-CPU-7. _drop_coarse_matrix builds I/J/V + sparse(...) + dropzeros! 
  — 1_AMG_setup.jl:204-254                                           
  Rebuild-only path, but quadrupled work vs a direct CSR-in-CSR-out    
  filter. Down-rank.                                                   
  
  P-CPU-8. Internal KA synchronize per kernel call —                   
  3_AMG_transfer.jl:15-21                                            
  On CPU these syncs are essentially free; still, the per-call overhead
   of constructing kernel! and launch is non-trivial for tiny-n coarse 
  levels. Keep the outer sync before norm/dot reductions; drop the
  per-kernel sync from _launch_amg_kernel!.                            
                                                                     
  P-CPU-9. RS direct interpolation row scan is quadratic (see C3 above)
   — Row-scan once and mark strong C contributions by stamping
  coarse_index[j] != 0 && strong[i] contains j.                        
                                                                     
  P-CPU-10. max_prolongation_entries=2 default for SA. Not a bug but   
  possibly over-truncating smoothed P. For smoothed aggregation, 4–6 is
   more typical; the current setting may be contributing to the        
  observed weak convergence on cylinder. Worth an experiment before  
  calling it a perf issue.

  4. Performance — GPU (latent, not measured in current runs)          
  
  These do not affect the current single-thread CPU benchmark but are  
  concrete GPU cliffs whenever the CUDA path is exercised.           
                                                                       
  P-GPU-1. refresh_finest_level! triggers full re-adapt of every level 
  via _sync_device_levels! — 1_AMG_setup.jl:315-320,
  1_AMG_setup.jl:348-355                                               
  Even when only fine-level diagonal/invdiag/λmax change,            
  _sync_device_levels! re-adapts every level, i.e. re-allocates every  
  device buffer and H2D-copies every array on every update. Hundreds of
   MB per refresh on a 1.67M-cell 9-level hierarchy. Fix: allocate     
  device levels once at setup, then in refresh copy only the specific
  arrays that changed (nzval, diagonal, inv_diagonal) into the existing
   device buffers.

  P-GPU-2. _pattern_matches D2H-copies rowptr+colval every update —    
  7_AMG_update.jl:5-6
  On CUDA, _cpu_vector(_rowptr(A)) and _cpu_vector(_colval(A)) allocate
   new Arrays and do full D2H copies per pressure solve. Same fix as   
  P-CPU-1 (cheap signature check first).
                                                                       
  P-GPU-3. _sync_finest_matrix! uses _cpu_vector(src) which allocates  
  intermediate Array on GPU — 7_AMG_update.jl:17-21
  _cpu_copyto!(dest, src) calls copyto!(dest, _cpu_vector(src)),       
  allocating an intermediate Array via Array(gpu).                     
  copyto!(cpu_buf::Vector, src::CuArray) works directly — drop the
  intermediate.                                                        
                                                                     
  P-GPU-4. Naive one-thread-per-row SpMV — 3_AMG_transfer.jl:1-8       
  Uncoalesced x[colval[p]] reads, no warp cooperation. On CUDA the fine
   matrix lives in CuSparseMatrixCSR upstream; using CUSPARSE mul! on  
  the finest level (the hot path) is a large win. Classified as      
  "requires structural change" because the current AMGMatrixCSR type is
   hand-rolled — mention as a GPU item the CUDA extension could address
   by wrapping CUSPARSE mul!. If this is outside scope, keep the note
  and de-prioritise.

  P-GPU-5. Per-kernel synchronize — same _launch_amg_kernel! location  
  On GPU each sync is 5–20 μs × many kernels per V-cycle. Keep sync
  only before host-observable operations (norm, dot, D2H for coarse    
  solve).                                                            
                                                                       
  P-GPU-6. Full CPU setup for GPU backend                              
  Coarsening, strength graph, Galerkin all run on CPU regardless of
  backend (_amg_setup_backend(::BACKEND) = CPU()). This is known (Stage
   C). Not a small fix — mention as structural.                      
                                                                       
#  Changes already implemented

1. C1             
2. P-CPU-1     
3. P-CPU-2                      
4. P-CPU-3                                                
5. P-CPU-4
6. P-CPU-6 — lu! in _refresh_coarse_cpu! (reuses symbolic factor)
7. P-GPU-3 — _cpu_copyto! uses copyto! directly (no intermediate Array)