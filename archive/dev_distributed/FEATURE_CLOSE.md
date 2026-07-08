# Feature close: distributed (MPI multi-node/multi-GPU) — branch HM/distributed-draft

## delivered
- New `src/Distribute` module: MPI distributed XCALibre; serial code paths untouched.
- Partitioning + local mesh, halo exchange, PETSc assembly/SpMV (`ext/XCALibrePETScExt.jl`).
- Distributed solvers `plaplace!`/`psimple!`/`ppiso!` unified with serial (p-twins deleted;
  seams = sync!/wrap_eqn/global_max/writer). KOmega turbulence distributed.
- Multi-GPU: GPU-native `mpiaijcusparse` solve (CUDA); F32 support via `dev/petscenv_f32`.
- I/O: decomposed OpenFOAM writer (no pvtu). Offline partitioning. Colocated periodics.
- HYPRE (BoomerAMG) via PETSc `--download-hypre` — no HYPRE.jl dep; missing-hypre build guarded.
- Gates green n=1,2,4(,8): serial 811/811, laplace/psimple/ppiso/turbulence/perf/io/scaling;
  GPU n=1,2. Numbers in `baselines.json`.

## pending
- Phase 8 §4 AD (optional add-on) — not started; user to decide §4 vs §5.
- Phase 8 §5 docs — recommend BoomerAMG for pressure in docs (NOT a code default: breaks
  non-hypre builds).
- Turbulence SST / KOmegaLKE / LES not wired for distributed.
- Lab-deferred (phase 6): cusparse mat-path variants, CUDA-aware MPI, multi-GPU scaling.
- AMD ext parity: mirrored but parse-check only, lab-unverified.

## safe-close (needed to build/run/resume this branch)
- **`dev/petscenv` and `dev/petscenv_f32` MUST be preserved.** These env project folders
  (Project.toml/Manifest wrapping the system PETSc F64/F32 builds) are required to run ALL
  MPI/PETSc code in this branch. They live in gitignored `dev/` (machine-local). If lost,
  rebuild via `dev/setup_petscenv_system.jl` (also kept in `dev/`, verify with
  `dev/verify_stack.jl`).
- Run MPI/PETSc gates: `source dev/local_stack.sh` AND `--project=dev/petscenv`
  (F32 work uses `--project=dev/petscenv_f32`). `local_stack.sh` also stays in `dev/`.
- PETSc must be built with `--download-hypre` (one flag covers F64+F32+CUDA; PETSc 3.24 pins
  hypre v3.0.0). Rebuild on the 14GB box: `JOBS=6` + `--with-make-np` (`-j32` OOM-kills).
  F32 hypre needs `-Wno-implicit-function-declaration` (ParaSails `dcopy_`, unfixed upstream).
- Full gotchas + decisions: `STATE.md` (## gotchas) and `DECISIONS.md` in this folder.
