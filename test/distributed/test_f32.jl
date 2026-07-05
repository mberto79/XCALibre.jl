# Phase 7 F32 gate (local-only): cavity psimple! with Float32 mesh/fields vs serial F64
# reference, loose tol. PETSc wrappers are precompile-time per-preference, so this needs
# --project=dev/petscenv_f32 (single-precision build: see build_cuda_ucx_openmpi_petsc.sh).
using XCALibre, PETSc, MPI, Test

any(l -> l.PetscScalar == Float32, PETSc.petsclibs) ||
    (println("SKIP test_f32: no Float32 PETSc lib (use --project=dev/petscenv_f32)"); exit(0))

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))
cavity_mesh_f32() = UNV2D_mesh(
    joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "quad40.unv"), scale=0.001,
    float_type=Float32)
iterations = 300

# serial F64 reference on rank 0
ref = if rank == 0
    model_s, config_s = incompressible_case(cavity_mesh(), cavity_bcs; iterations)
    simple!(model_s, config_s; pref=0.0)
    (collect(model_s.momentum.U.x.values), collect(model_s.momentum.U.y.values),
     collect(model_s.momentum.p.values))
else
    nothing
end
Us_x, Us_y, ps = MPI.bcast(ref, comm; root=0)

gmesh32 = rank == 0 ? cavity_mesh_f32() : nothing
dm = distribute(gmesh32; comm=comm)
model, config = incompressible_case(dm, cavity_bcs; iterations)
residuals = run!(model, config; pref=0.0)

dux, duy, dp = field_errors(dm, model, Us_x, Us_y, ps)
n = dm.partition.n_owned
nloc = n + dm.partition.n_ghost
orig = dm.orig_cells

@testset "psimple F32 cavity (rank $rank)" begin
    @test eltype(model.momentum.U.x.values) === Float32
    @test dux < 5e-3
    @test duy < 5e-3
    @test dp < 5e-3
end
rank == 0 && println("PSIMPLE F32 cavity n=$(MPI.Comm_size(comm)) dux=$dux duy=$duy dp=$dp")

# NEW SECTION: halo stays F32-consistent (ghosts exact copies of owner values)

phi = ScalarField(dm)
phi.values[1:n] .= Float32.(orig[1:n])
H = HaloExchange(dm, 1, CPU(); comm)
halo_exchange!(phi, H, CPU(), 64)

@testset "halo F32 (rank $rank)" begin
    @test eltype(phi.values) === Float32
    @test all(phi.values[n+1:nloc] .== Float32.(orig[n+1:nloc]))
end
