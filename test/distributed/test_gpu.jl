# Phase 6/7 GPU gate (local-only, not CI): cavity psimple! on CUDABackend vs serial CPU,
# ranks sharing local GPUs via bind_device!. With a CUDA PETSc (system build) the solve
# runs natively (mpiaijcusparse); without it, solves opt into solve_on=CPU() and the
# no-solve_on call must hard-error (no silent fallback).
using XCALibre, PETSc, MPI, Test, CUDA
using PETSc: LibPETSc

CUDA.functional() || (println("SKIP test_gpu: CUDA not functional"); exit(0))

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

petsclib = PETSc.petsclibs[findfirst(l -> l.PetscScalar == Float64, PETSc.petsclibs)]
PETSc.initialize(petsclib)
petsc_cuda = LibPETSc.PetscHasExternalPackage(petsclib, Vector{Int8}(codeunits("cuda\0")))
rank == 0 && println("PETSc CUDA: $petsc_cuda → solve path: $(petsc_cuda ? "native device" : "solve_on=CPU() stopgap")")

include(joinpath(@__DIR__, "psimple_case.jl"))

backend = CUDABackend()
bind_device!(backend, rank)
iterations = 300

gmesh = rank == 0 ? cavity_mesh() : nothing
ref = if rank == 0
    model_s, config_s = incompressible_case(gmesh, cavity_bcs; iterations)
    simple!(model_s, config_s; pref=0.0)
    (collect(model_s.momentum.U.x.values), collect(model_s.momentum.U.y.values),
     collect(model_s.momentum.p.values))
else
    nothing
end
Us_x, Us_y, ps = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
dm_dev = adapt(backend, dm)
model, config = incompressible_case(dm_dev, cavity_bcs; iterations, backend)
residuals = prun!(model, config; pref=0.0, solve_on=(petsc_cuda ? nothing : CPU()))

dux, duy, dp = field_errors(dm_dev, model, Us_x, Us_y, ps)
n = dm.partition.n_owned
nloc = n + dm.partition.n_ghost
orig = dm.orig_cells
px = Array(model.momentum.U.x.values)

@testset "psimple GPU cavity (rank $rank)" begin
    @test dux < 1e-5
    @test duy < 1e-5
    @test dp < 1e-5
    @test all(abs(px[i] - Us_x[orig[i]]) < 1e-5 for i ∈ n+1:nloc)
    @test maximum(residuals.p[iterations÷2:end]) < 1e-6
    @test maximum(residuals.Ux[iterations÷2:end]) < 1e-6
end
rank == 0 && println("PSIMPLE GPU cavity n=$(MPI.Comm_size(comm)) dux=$dux duy=$duy dp=$dp")

# NEW SECTION: host-staging vs auto (CUDA-aware when MPI supports it) halo paths

phi_a, phi_b = ScalarField(dm_dev), ScalarField(dm_dev)
vals = zeros(length(dm.cells))
vals[1:n] .= Float64.(orig[1:n])
copyto!(phi_a.values, vals); copyto!(phi_b.values, vals)
H_auto = HaloExchange(dm_dev, 1, backend; comm)
H_staged = HaloExchange(dm_dev, 1, backend; comm, cuda_aware=false)
halo_exchange!(phi_a, H_auto, backend, 64)
halo_exchange!(phi_b, H_staged, backend, 64)

@testset "halo staging vs auto (rank $rank)" begin
    @test Array(phi_a.values) == Array(phi_b.values)
    @test all(Array(phi_a.values)[n+1:nloc] .== Float64.(orig[n+1:nloc]))
end

# NEW SECTION: no-silent-fallback error path

if !petsc_cuda
    @testset "GPU fields + non-CUDA PETSc errors (rank $rank)" begin
        err = try (prun!(model, config; pref=0.0); nothing) catch e e end
        @test err isa ErrorException
        @test occursin("solve_on=CPU()", err.msg)
    end
else
    rank == 0 && println("PETSc has CUDA: error-path test skipped (device path active)")
end
