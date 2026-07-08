status: phase 8 §3 HYPRE COMPLETE 2026-07-05 | committed 6954fcb2, verified clean
## position
Phase 8: §1 unification, §2 offline partitioning, §3 HYPRE all CLOSED (gates green).
Remaining: §4 AD (optional add-on) and §5 docs — user to pick.
## gotchas
- MPI/PETSc gates: source dev/local_stack.sh AND --project=dev/petscenv (F32: petscenv_f32)
- residual = post-solve LINEAR residual → gates: conv=1e-15 + fixed iters + tight rtol
- never device arrays into MPI collectives; adapt keeps partition/procs on host
- BoomerAMG: PETSc-only (serial ctor errors), SPD only (no transpose apply); missing-hypre
  build errors at PETScSolver construction; "-pc_type hypre" passthrough also guarded
- PETSc rebuild: JOBS=6 + --with-make-np on 14GB box (-j32 OOM-kills unrelated processes)
- F32 hypre build needs -Wno-implicit-function-declaration via
  --download-hypre-configure-arguments (ParaSails hard-codes dcopy_; unfixed upstream)
- PETSc 3.24 pins hypre v3.0.0; one --download-hypre flag covers F64+F32+CUDA
- new Project.toml dep/weakdep needs [compat] (General registry blocks releases)
## scripts
- test/distributed/runtests_mpi.jl [files...] — XCAL_MPI_RANKS="1 2 4"
- test/distributed/test_hypre.jl — self-skipping; prove solve branch via "HYPRE cavity" line
- examples/3D_cascade_mpi_GPU.jl — uses REAL periodics
