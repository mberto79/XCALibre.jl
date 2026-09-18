# conda-forge CUDA PETSc (P1-M17-S1)

Machine: RTX 4070 Laptop (sm_89), driver 580.126.09, Julia 1.13.0, CUDA.jl runtime 13.4 artifact. Installed with micromamba 2.9.0 (`~/.local/micromamba`), `CONDA_OVERRIDE_CUDA=12.9`, `petsc=3.25.5=cuda12_real*` (Float64, Int32, hypre 3.2.0 host-only). Envs: `dev/petscenv_conda` (mpich 5.0.1), `dev/petscenv_conda_ompi` (openmpi 5.0.10). Test: `test/distributed/test_gpu.jl` (cavity psimple! on CUDABackend vs serial CPU, 300 iterations, Jacobi).
- mpich, no options: PETSc aborts at init, "MPI is not GPU-aware" (n=1,2). `MPIR_CVAR_ENABLE_GPU=1` does not help; `MPI.has_cuda()` false.
- mpich, `PETSC_OPTIONS="-use_gpu_aware_mpi 0"`: PASS n=1,2, "native device solve", |dU| 1.2e-11, |dp| 5.6e-12 vs serial.
- openmpi, `OMPI_MCA_opal_cuda_support=true`, no PETSc option: PASS n=1,2, native device solve, same deltas; `MPI.has_cuda()` true only after `MPI.Init()`.
- CUDA.jl 13.4 runtime and conda's CUDA 12.9 libraries coexist in one process without error.
- Setup traps hit: OpenMPI ABI needs `Pkg.instantiate()` after `use_system_binary` (else `SCALAPACK32` artifact missing, `using PETSc` fails); `set_library!` then `Pkg.precompile()` before the first `mpiexec`, else a child fails "Precompiled image ... not available with flags".
- Install size: 2.9 GB per conda env.
