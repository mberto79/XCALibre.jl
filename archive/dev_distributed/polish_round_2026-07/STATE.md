WORKING BRANCH: HM/distributed-draft
status: polish round COMPLETE | clean, 6 wip commits (3806ea79..144cb78b)
## position
All 5 phases done+gated: 1 config echo/fallback, 2 distribute(Function), 3 rank logger,
4 BoomerAMG(;kwargs)+docs, 5 KOmegaSST+wall_distance distributed. KOmega regression green.
Awaiting user: archive, or next (LKE/LES distributed, wall-function SST).
## gotchas
- run! routes to distributed transparently (is_distributed_mesh seams); serial untouched.
- PETSc solve reads ONLY rtol/atol/itmax (KSPSetTolerances); `convergence` field is IGNORED
  in distributed. `relax` is equation under-relaxation (pre-solve), not a PETSc knob — works.
- _ksp_type/_pc_type (XCALibrePETScExt) curate Cg/Bicgstab/Gmres + Jacobi/BoomerAMG; anything
  else needs petsc_options string passthrough.
- BoomerAMG serial ctor errors on purpose (PETSc/distributed only).
## gotchas (added this round)
- runtests_mpi.jl SWALLOWS child stdout on test success — verify @info/echo via a direct
  `julia -e 'using MPI; run(\`$(MPI.mpiexec()) -n N $(Base.julia_cmd()) ... file\`)'` run.
- Logging is stdlib but MUST be in [deps] (matched Printf/Serialization: no compat entry);
  after adding, `Pkg.resolve()` dev/petscenv (and main) or precompile errors "does not have Logging".
- quiet_nonroot! sets global_logger on non-root at distribute() — @info once, @warn/@error all ranks.
## identified recommendations
- wall_distance! prints "did not converge" (residual ~9e-10 vs 1e-15 threshold) — pre-existing
  serial artifact (solvers.y.convergence hard-compared to 1e-15); harmless but noisy. Not this round.
- SST distributed needs sync!(nut) (gradU tensor-grad unsynced) + sync!(phi) in wall_distance!
  (relaxation corrupts ghosts) — KOmega didn't. LKE/LES will need the same audit when wired.
- convergence silently ignored in distributed — map or warn (D2).
- No curated BoomerAMG tuning; only petsc_options string (D3).
- Per-rank @info noise; need rank-aware logger not guards (D3/logging).
- SST/KOmegaLKE/LES not distributed (pending from archive).
- Multi-GPU scaling / CUDA-aware MPI / AMD parity / AD = lab/hardware, out of scope here.
## scripts
- (none yet this round)
