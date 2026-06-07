# AMG GPU Impl Archive — Decisions & Gotchas

## D0 Architecture
- GREENFIELD: device/ owns GPU pipeline (setup/RAP/refresh/cycle); reference/ = CPU track + oracle
- Gate: _use_greenfield_amg = _greenfield_implemented()&&backend isa KA.GPU&&fuse_levels>0
  Also requires Geometric+AMGJacobi. _greenfield_implemented()=true (device/0). fuse_levels default 0.
- Layout: AMG.jl includes reference/0..7 then device/0..7
  device/: 0=gate,1=macro,2=5a,3=5b,4=cycle+sc,5=G1,6=G3gate,7=integration
- AMGHierarchy.greenfield::Ref{Any} holds MFGreenfield(st,plan,perm,r_perm)

## Index types
- LOCAL within-agg: UInt8 (<=128, cuSPARSE-free); GLOBAL colval: Int32 signed (cuSPARSE wants Cint)

## Geometric P/R normalization (CRITICAL)
- NORMALIZED piecewise-const: restrict rc[J]=(Σr[i])/sqrt(w), prolong x[i]+=xc[J]/sqrt(w)
- Kernel must use 1/sqrt(w). Source: 2_AMG_coarsening.jl:435,535

## KA gotchas (CRITICAL for kernel work)
- @localmem T = where-param only (eltype in body -> StaticArrays crash CPU)
- Recompute lo/hi/w/g/l from @index after EVERY @synchronize (body-split drops locals)
- Val{W} for workgroup size; in-agg test = contiguous lo<=col<=hi

## G3 — dead in preconditioner mode
- z=0 + no cross-wg sync -> off-agg=0 -> EXACTLY block-Jacobi; CAN'T warm-start a preconditioner
- 2.6-2.8x iters poisson, PCG DIVERGES F1 (final_rel 46.8); NOT omega issue
- UNTESTED in FAS/standalone mode (off-agg=lagged iterate); see G6 gate

## scale_correction
- sf=(r·c)/(c·Ac); required for AMGSolver (F1 stalls without, relres ~1e-2 at 300 iters)
- Gated mode isa AMGSolver. fused_top>0 + sc -> error (Option-C-with-refresh deferred)
- OPEN: AMGSolver F1 gf 334 vs ref 262 (direction gap; NOT omega, NOT sc bug)

## VRAM & lean refresh
- Lean refresh (a+b+c): plan device 374->0.4MB; gf 518 vs ref 681 = -163MB F1 ml=1
- ~97MB moved VRAM->host RAM; refresh 106ms vs 56.5ms G1 (per-timestep not per-iter)
- F1 CFD ml=2/mc=100/F32: gf 3510MB vs ref 4438MB = -928MB (more ops erased at those settings)

## Frozen refresh gotcha (CRITICAL)
- _geometric_aggregates is VALUE-DEPENDENT -> frozen refresh = approximate (standard transient AMG)
- Oracle MUST be frozen-aggregation RAP on A2 values, NOT full-rebuild(A2) (would re-aggregate)
- _assert_rap_total non-allocating, one-time. FALLBACK: derive coarse pattern structurally

## F32/UMFPACK
- UMFPACK promotes F32->F64; TF!=T for F32 -> split host bufs (solve TF, transfer T, alias when equal)

## Option C (matrix-free Galerkin coarse ops)
- fused_top knob: coarse A_l NEVER stored for levels 2..fused_top+1; applied via R-chain*A0*P-chain
- Cycle +37%/+73% at k=1/2. EXACT (relerr ~eps). fused_top=0 guard enforces.

## G2 zero-alloc
- coarse_n<=512: device dense-inverse GEMV (inv(), no host sync); else reusable-buffer host LU ldiv!
- KA launch adds fixed host alloc (scales with #levels not coarse_n) -> @allocated!=0 GPU is OK

## Deferred
- Option-C-with-refresh (G1 fused_top==0 assert is the blocker; only if OOM binds post-perf)
- Default-ON (perf gap + AMGSolver direction gap)
- G6 FAS gate (cheap CPU run; user decision for speed-SOTA fork)
