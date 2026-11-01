# Mixed precision on distributed meshes: Float64 fields, PETSc_jll's Float32 library (loaded beside the
# Float64 one) solving the correction. Must match the serial MixedF32() run.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# PETSc's GAMG rounds global row ids held in PetscReal: Float32 builds before 3.26.99 must refuse
# matrices above 2^24 + 1 rows at setup instead of failing inside PCSetUp
@testset "GAMG global-id limit (rank $rank)" begin
    ext = Base.get_extension(XCALibre, :XCALibrePETScExt)
    gamg = merge((; ksp_type="cg", pc_type=ext._pc_type(GAMG())), ext._pc_options(GAMG()))
    lim = Int(maxintfloat(Float32)) + 1
    @test ext._uses_gamg(gamg)
    @test ext._uses_gamg((pc_type="jacobi", sub_pc_type="gamg"))
    @test !ext._uses_gamg((pc_type="jacobi",))
    @test ext._gamg_ids_exact(v"3.22.0", Float32, lim)
    @test !ext._gamg_ids_exact(v"3.22.0", Float32, lim + 1)
    @test !ext._gamg_ids_exact(v"3.26.0", Float32, 33_402_240)
    @test ext._gamg_ids_exact(v"3.26.99", Float32, 33_402_240)
    @test ext._gamg_ids_exact(v"3.22.0", Float64, 33_402_240)
    @test isnothing(ext._check_gamg_ids(v"3.22.0", Float32, (pc_type="jacobi",), 33_402_240))
    @test isnothing(ext._check_gamg_ids(v"3.22.0", Float32, gamg, lim))
    err = try ext._check_gamg_ids(v"3.22.0", Float32, gamg, lim + 1); nothing catch e e end
    @test err isa ErrorException && occursin("FullPrecision()", err.msg)
    @test_throws ErrorException ext._check_gamg_ids(v"3.22.0", Float16, gamg, 4096)
    lib = ext._petsclib(Float64, 0)
    PETSc.initialize(lib)
    @test ext._petsc_version(lib) >= v"3.0"
end

if !any(l -> l.PetscScalar == Float32, PETSc.petsclibs)
    @testset "MixedF32() distributed BFS (rank $rank)" begin
        @test_skip "no Float32 PETSc library loaded"
    end
else
include(joinpath(@__DIR__, "psimple_case.jl"))

const ITERS = 100
const SETUP = (precision=MixedF32(), rtol=1e-2)
fields(model) = (collect(model.momentum.U.x.values), collect(model.momentum.U.y.values), collect(model.momentum.p.values))
function serial_run(mesh; iterations=ITERS, kwargs...)
    model, config = incompressible_case(mesh, bfs_bcs; iterations, kwargs...)
    run!(model, config)
    fields(model)
end

# below Float32's attainable rtol each solve refines its correction, so few outer iterations match Float64
const TIGHT = (iterations=20, rtol=1e-8)
gmesh = rank == 0 ? bfs_mesh() : nothing
ref = MPI.bcast(rank == 0 ? (serial_run(gmesh; SETUP...),
    serial_run(gmesh; TIGHT..., precision=FullPrecision())) : nothing, comm; root=0)
dm = distribute(gmesh; comm=comm)

model, config = incompressible_case(dm, bfs_bcs; iterations=ITERS, SETUP...)
run!(model, config)
dux, duy, dp = field_errors(dm, model, ref[1]...)
rank == 0 && println("MIXED Float32 n=$(MPI.Comm_size(comm)) vs serial mixed: dux=$dux duy=$duy dp=$dp")

model, config = incompressible_case(dm, bfs_bcs; TIGHT..., precision=MixedF32())
run!(model, config)
tight = field_errors(dm, model, ref[2]...)
rank == 0 && println("MIXED Float32 rtol=$(TIGHT.rtol) vs F64: $tight")

# PETSc's and Krylov.jl's Float32 Cg take slightly different paths under the loose rtol (~1e-5)
@testset "MixedF32() distributed BFS (rank $rank)" begin
    @test dux < 5e-5
    @test duy < 5e-5
    @test dp < 5e-5
    @test all(<(1e-6), tight)
    @test_throws ArgumentError incompressible_case(dm, bfs_bcs; iterations=1,
        precision=MixedF16()) |> c -> run!(c...)
end
end
