# GPU-native distributed testing — what's needed on this machine

> **STATUS 2026-07-03: INSTALLED.** UCX 1.21 + OpenMPI 5.0.10 + PETSc 3.25.3, all CUDA
> (system toolkit 12.2), live in /home/humberto/customInstalls. Use `source dev/local_stack.sh`
> before every run; one-time Julia config = `dev/setup_petscenv_system.jl` (already applied
> to dev/petscenv). Verify: `dev/verify_stack.jl`. Sections below kept for reference.
> Note: this PETSc is **Int32-indexed** (no --with-64-bit-indices) — code must not assume
> PetscInt=Int64 (fixed in XCALibrePETScExt via `_petsclib(TF)`).
> CUDA.jl stays on the ARTIFACT runtime, NOT local_toolkit: system cusparse 12.2 rejects
> the NULL DnVec descriptor KrylovPreconditioners creates (needs ≥12.3); driver 580
> (CUDA 13.0) runs both runtimes side by side and device pointers interop at driver level.
> OMPI 5.0 quirk: MPIX_Query_cuda_support()==false despite working device p2p →
> JULIA_MPI_HAS_CUDA=1 in local_stack.sh so MPI.has_cuda()/HaloExchange pick CUDA-aware.

## OPEN BLOCKER (2026-07-04): cusparse device solve segfaults from Julia
`PETScSolver` on GPU fields dies in `MatCreateVecs` right after `MatConvert→mpiaijcusparse`.
Isolated in dev-scratch repros:
- The *identical* PETSc call sequence in pure C (`MatCreateMPIAIJWithArrays`→`MatConvert`
  MATMPIAIJCUSPARSE→`MatCreateVecs`) **works** → library + CUDA stack are fine.
- From Julia it segfaults at the cusparse `MatCreateVecs`; host `MatCreateVecs` + `MatConvert`
  both succeed. Not CUDA.jl (crashes with it unloaded), not the `MatCreateVecs` binding
  (signature `(CMat,Ptr{CVec},Ptr{CVec})` is correct), not fixed by forcing the CUDA context
  first or `-cuda_initialize 0`.
- **Root suspect: PETSc.jl fork bindings are `PETSC_WRAPPERS_VERSION` 3.24.0 but the library
  is 3.25.3** — a struct-layout skew that only bites the cusparse Vec path.
Fix (user decision, do NOT overwrite the working 3.25.3 build unasked):
  (a) rebuild PETSc at 3.24.x (`PETSC_VERSION` in build_cuda_ucx_openmpi_petsc.sh) to match, or
  (b) regenerate PETSc.jl wrappers for 3.25.3.
Until then, `solve_on=CPU()` (host-staged solve, device fields) is the working fallback.

## Local stack limitations and -use_gpu_aware_mpi (laptop-only workarounds)

Observed on this machine (Open MPI 5.0.10 + UCX 1.21 + CUDA 12.2, single RTX 4070):
- Device-pointer point-to-point (Sendrecv/Isend/Irecv) works **only with UCX forced**
  (`OMPI_MCA_pml=ucx` etc. — in local_stack.sh).
- Device-pointer **collectives segfault** (Open MPI's CUDA coll path). PETSc's GPU-aware
  probe does a device-buffer Allreduce → fails → PETSc must run `-use_gpu_aware_mpi 0`
  (set via PETSC_OPTIONS in local_stack.sh).

Implications — why this does NOT compromise the development:
1. `-use_gpu_aware_mpi 0` only changes **PETSc-internal** comms (VecScatter/PetscSF in
   MatMult during KSP iterations): device buffers staged via host. The solve itself
   (cuSPARSE SpMV, device dots/axpy, Jacobi) still runs fully on GPU → mpiaijcusparse
   correctness validation done locally is fully valid; only PETSc's comm bandwidth differs.
2. **Our halo exchange is independent of PETSc's flag**: MPI.jl Isend/Irecv on device
   buffers, auto-selected via MPI.has_cuda(). That p2p path works locally with UCX forced,
   so the real CUDA-aware halo path IS exercised locally.
3. All XCALibre collectives (pnorm/pdot/pmean, flux corrections, Courant) reduce **host
   scalars** — the broken device-collective path is never hit. RULE: never pass device
   arrays to MPI collectives (matters for the parallel-I/O gather: stage to host first).
4. **Nothing is baked into code.** The flag lives only in dev/local_stack.sh. On HPC with a
   proper CUDA-aware stack (Cray MPICH / OpenMPI+UCX+gdrcopy) PETSc's probe passes and the
   default enables full device MPI — zero code change.

Left unvalidated locally (HPC bring-up checklist):
- PETSc's GPU-aware VecScatter path (perf-only; PETSc's own code) — smoke-test with
  default `-use_gpu_aware_mpi` on HPC before production runs.
- True multi-GPU (2 local ranks share one RTX 4070) and inter-node transport.
- If HPC probe also fails (misconfigured stack), symptom = PETSc error at first CUDA solve;
  remedy is fixing the MPI stack, NOT forcing 0 in production (host staging halves
  effective PETSc comm bandwidth).

Goal: run the distributed tests genuinely on the GPU (device assembly + device PETSc
solves + GPU-aware halo exchange). Phase 6 validated GPU fields only via
`solve_on=CPU()` host staging — a stopgap, not a supported configuration.

## What blocks the native path today
1. **MPI is not CUDA-aware.** `dev/petscenv` uses the default MPICH_jll → `MPI.has_cuda() == false`,
   so `HaloExchange` auto-selects host staging and passing device pointers to MPI would segfault.
2. **PETSc has no CUDA.** PETSc_jll is built without CUDA (verified:
   `PetscHasExternalPackage(petsclib, "cuda") == false`), so the `mpiaijcusparse`
   MatConvert path in the extension is written but unverified. There is no CUDA-enabled
   PETSc JLL — a system build is required.

## What to install (both must be built against the SAME MPI and CUDA toolkit)

### 0. CUDA toolkit with nvcc
CUDA.jl works via artifacts, but source builds need a system toolkit:
`nvcc --version` must work (e.g. `sudo apt install nvidia-cuda-toolkit`, or NVIDIA's
repo for a toolkit matching driver 5xx). Call its root `$CUDA_HOME` (often `/usr/local/cuda` or `/usr`).

### 1. CUDA-aware OpenMPI
```bash
# UCX with CUDA first (OpenMPI's CUDA transport)
wget https://github.com/openucx/ucx/releases/download/v1.17.0/ucx-1.17.0.tar.gz
tar xf ucx-1.17.0.tar.gz && cd ucx-1.17.0
./configure --prefix=$HOME/opt/ucx-cuda --with-cuda=$CUDA_HOME && make -j$(nproc) install

wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.gz
tar xf openmpi-5.0.7.tar.gz && cd openmpi-5.0.7
./configure --prefix=$HOME/opt/ompi-cuda --with-cuda=$CUDA_HOME --with-ucx=$HOME/opt/ucx-cuda
make -j$(nproc) install
```
Verify: `$HOME/opt/ompi-cuda/bin/ompi_info --parsable --all | grep cuda_support:value` → `true`.

### 2. PETSc with CUDA, against that OpenMPI
```bash
export PATH=$HOME/opt/ompi-cuda/bin:$PATH
git clone -b release https://gitlab.com/petsc/petsc.git && cd petsc
./configure --prefix=$HOME/opt/petsc-cuda --with-cuda=1 --with-cudac=nvcc \
  --with-mpi-dir=$HOME/opt/ompi-cuda --download-fblaslapack --with-debugging=0
make all && make install
```

### 3. Point Julia (dev/petscenv) at both
```julia
# once, in dev/petscenv:
using MPIPreferences
MPIPreferences.use_system_binary(; library_names=["$(homedir())/opt/ompi-cuda/lib/libmpi"],
    mpiexec="$(homedir())/opt/ompi-cuda/bin/mpiexec")
```
Then set for every run (shell profile or wrapper):
```bash
export JULIA_PETSC_LIBRARY=$HOME/opt/petsc-cuda/lib/libpetsc.so
export LD_LIBRARY_PATH=$HOME/opt/ompi-cuda/lib:$HOME/opt/petsc-cuda/lib:$LD_LIBRARY_PATH
```
PETSc.jl must be rebuilt/precompiled after setting `JULIA_PETSC_LIBRARY` (it bakes the
petsclib in at precompile time): `Pkg.build("PETSc"); Pkg.precompile()`.

## Verification checklist (rank 0, in dev/petscenv)
```julia
using MPI, CUDA, PETSc
MPI.has_cuda()                        # must be true
CUDA.functional()                     # true (already is)
petsclib = PETSc.petsclibs[1]
PETSc.LibPETSc.PetscHasExternalPackage(petsclib, Vector{Int8}(codeunits("cuda\0")))  # true
```
Then: `XCAL_MPI_RANKS="1 2" julia --project=dev/petscenv test/distributed/runtests_mpi.jl test_gpu.jl`
(I will update test_gpu.jl to drop `solve_on=CPU()` automatically when PETSc reports CUDA.)

## Notes
- One RTX 4070 (8 GB): 2 ranks share it — validates correctness, not scaling. True
  multi-GPU scaling still needs the lab machine.
- Versions above are known-good examples, not pins; any recent UCX 1.1x / OpenMPI 5.x /
  PETSc 3.2x should do. Keep MPI+PETSc+toolkit consistent — mixing MPIs is the classic failure.
- Alternative: `spack install petsc +cuda ^openmpi +cuda` does 1+2 in one shot if you
  prefer spack; then point MPIPreferences/JULIA_PETSC_LIBRARY at spack's install prefixes.
