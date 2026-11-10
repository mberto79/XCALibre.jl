# GPU gate (local-only, not CI): the density-based solver on CUDABackend, one GPU per rank (ranks
# share local GPUs via bind_device!), vs the serial CPU run. Laminar, so no linear solve and no
# CUDA-enabled PETSc is needed.
using XCALibre, PETSc, MPI, Test, CUDA

CUDA.functional() || (println("SKIP test_gpu_godunov: CUDA not functional"); exit(0))

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "godunov_case.jl"))

backend = CUDABackend()
bind_device!(backend; comm)
kw = (iterations=30, adaptive=LocalTimeStepping(maxCo=0.8, startCo=0.2, ramp=10))
forces = (patches=[:cylinder], interval=5)

gmesh = rank == 0 ? cylinder_mesh() : nothing
ref = if rank == 0
    ms, cs = godunov_case(gmesh; kw...)
    rs = run!(ms, cs; progress=false, forces)
    (map(f -> collect(f.values), godunov_fields(ms)), rs)
else
    nothing
end
fields_s, res_s = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
model, config = godunov_case(adapt(backend, dm); backend, kw...)
res = run!(model, config; progress=false, forces)

n = dm.partition.n_owned
nloc = n + dm.partition.n_ghost
orig = dm.orig_cells
fields_d = map(f -> Array(f.values), godunov_fields(model))
owned = maximum(relerr(fd[1:n], fs[orig[1:n]], fs) for (fd, fs) ∈ zip(fields_d, fields_s))
ghost = maximum(relerr(fd[n+1:nloc], fs[orig[n+1:nloc]], fs) for (fd, fs) ∈ zip(fields_d, fields_s))
F_s, F_d = res_s.forces.cylinder, res.forces.cylinder
dF = maximum(relerr(F_d.pressure[i], F_s.pressure[i]) for i ∈ eachindex(F_s.pressure))

@testset "godunov! on GPU ranks (rank $rank)" begin
    @test model.momentum.p.values isa CuArray
    @test owned < 1e-8
    @test ghost < 1e-8
    @test relerr(res.rho, res_s.rho) < 1e-8
    @test dF < 1e-8
end
rank == 0 && println("godunov GPU n=$(MPI.Comm_size(comm)): owned=$owned ghost=$ghost force=$dF")
