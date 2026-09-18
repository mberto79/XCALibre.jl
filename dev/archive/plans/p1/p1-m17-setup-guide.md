# P1-M17 - MPI and PETSc setup guide (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R1, R6, R10. Governing decisions: D4, D47, D52, D68.

## Problem, quantified

The distributed page says only that a custom PETSc or system MPI "is selected through preferences". A user with a GPU must compile PETSc with CUDA, which is the main barrier to GPU runs (D68).

## What the sources say (research 2026-09-18; cite these in the guide)

- MPI.jl: `MPIPreferences.use_system_binary(; library_names, mpiexec)` or `use_jll_binary("MPICH_jll"|"OpenMPI_jll"|"MPItrampoline_jll")`, written to the env's LocalPreferences.toml, then restart and `Pkg.instantiate()` (https://juliaparallel.org/MPI.jl/stable/configuration/). MPItrampoline: build MPIwrapper against the target MPI, `export MPITRAMPOLINE_LIB=.../libmpiwrapper.so`, verify with `MPI.Get_library_version()` (PETSc.jl docs/src/man/hpc.md). CUDA-aware check: `MPI.has_cuda()`, `JULIA_CUDA_MEMORY_POOL=none`.
- PETSc.jl 0.4.10: `PETSc.set_library!(path; PetscScalar, PetscInt)` writes `library_path`, `PetscScalar`, `PetscInt` preferences (src/init.jl:319); `unset_library!`, `library_info()`; read at precompile, so restart. PetscInt defaults to Int64 and must match `sizeof_PetscInt` in petscconf.h. One custom library per environment. The library must be built against the MPI that MPI.jl uses (hpc.md section 2).
- PETSc_jll follows MPIPreferences through platform augmentation (mpich, openmpi, mpitrampoline, mpiabi artifacts; Yggdrasil P/PETSc/build_tarballs.jl), but has NO CUDA in any variant. A CUDA-aware MPI through MPItrampoline therefore gives GPU-aware MPI for XCALibre's halo exchange only; PETSc solves stay on the host, and XCALibre refuses GPU fields on a host PETSc (D47).
- No-compile CUDA PETSc: conda-forge `petsc` has `cuda12_real_*` and `cuda13_real_*` builds (linux-64, aarch64; mpich and openmpi variants; Int32; hypre CPU-only, so BoomerAMG runs on the host) (https://github.com/conda-forge/petsc-feedstock). Load with `set_library!("$CONDA_PREFIX/lib/libpetsc.so"; PetscInt=Int32)` plus `use_system_binary` on the same env's libmpi. conda openmpi is CUDA-aware but needs `OMPI_MCA_opal_cuda_support=true`. UNTESTED: possible libstdc++/libgfortran clashes with Julia's.
- Also possible: Spack/E4S build cache (`spack install -b only petsc+cuda`), E4S GPU containers. NVIDIA HPC SDK ships no PETSc. Distro PETSc is CPU-only.
- Compile route: `./configure --prefix=... --with-debugging=0 --with-shared-libraries=1 --with-mpi-dir=$MPI_DIR --with-cuda=1 --with-cuda-arch=<sm> --download-hypre --download-fblaslapack COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3 CUDAOPTFLAGS=-O3`, which gives GPU BoomerAMG with aijcusparse. `-use_gpu_aware_mpi 0` stages PETSc's messages through the host when the MPI is not CUDA-aware.

## Steps

- [x] **P1-M17-S1** try the conda-forge CUDA PETSc on this machine: install into a conda env, point a fresh project env at its libpetsc and libmpi, run `test/distributed/test_gpu.jl` at n=1,2 - mechanism: measurement only - cost: one install plus one test run - verdict: if it passes, the guide presents it as the no-compile GPU route; if it fails, the guide records the failure and the compile route leads. DONE (D69): both conda variants pass n=1,2 native; mpich needs `-use_gpu_aware_mpi 0`, openmpi is CUDA-aware.
- [x] **P1-M17-S2** write the setup section (overview, choosing MPI, pointing PETSc.jl at a library, GPU routes in order of effort, CUDA-aware MPI flags and checks, troubleshooting) and fix the run-on paragraph in "When AMG helps" - mechanism: documentation - verdict: every procedure run here or cited, docs build green. DONE (D70): section written, conda route leads, BoomerAMG-on-GPU crash documented, docs build green.

## Exit criterion

The setup section exists, its GPU route is the one S1 verified (or states plainly that none was verified without compiling), and the docs build green.

## Open questions

- [x] Does the conda-forge library load alongside Julia's own runtime libraries without clashes? Yes, CUDA 12.9 libs beside CUDA.jl 13.4 (D69).
- [x] Added in S2 by user ruling (D71): GPU communication auto-detection in the PETSc extension, so neither conda MPI needs an env var.
