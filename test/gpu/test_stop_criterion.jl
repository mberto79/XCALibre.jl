# The stopping rule of unit/Solve/test_stop_criterion.jl on the GPU (not in a runtests.jl tier: needs a GPU).
#   julia --project=<env with XCALibre and CUDA> test/gpu/test_stop_criterion.jl
using XCALibre, Test, CUDA, Accessors
using LinearAlgebra, SparseArrays

const STOP_CRITERION_GPU = true
include(joinpath(@__DIR__, "..", "unit", "Solve", "test_stop_criterion.jl"))

# device preconditioners in place of DILU
const GPU_CASES = (
    (Cg(), Jacobi(), FullPrecision()), (Cg(), IC0GPU(), FullPrecision()),
    (Bicgstab(), Jacobi(), FullPrecision()), (Bicgstab(), ILU0GPU(), FullPrecision()),
    (Cgs(), Jacobi(), FullPrecision()), (Gmres(), Jacobi(), FullPrecision()), (Gmres(), ILU0GPU(), FullPrecision()),
    (Cg(), Jacobi(), MixedF32()), (Bicgstab(), Jacobi(), MixedF32()), (Gmres(), Jacobi(), MixedF32()),
    ((s, Jacobi(), FullPrecision()) for s ∈ AMG_MODES)...)

check_stop_criterion(GPU_CASES; backend=CUDABackend())
check_amg_solver(backend=CUDABackend())
