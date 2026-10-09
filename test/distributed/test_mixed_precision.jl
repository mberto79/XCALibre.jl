# Mixed precision on distributed meshes: Float64 fields, PETSc_jll's Float32 library (loaded beside the
# Float64 one) solving the correction. Must match the serial MixedF32() run.
using XCALibre, PETSc, MPI, Test

any(l -> l.PetscScalar == Float32, PETSc.petsclibs) ||
    (println("SKIP test_mixed_precision: no Float32 PETSc library loaded"); exit(0))

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

const ITERS = 300
const SETUP = (precision=MixedF32(), rtol=1e-2)
fields(model) = (collect(model.momentum.U.x.values), collect(model.momentum.U.y.values), collect(model.momentum.p.values))
function serial_run(mesh; kwargs...)
    model, config = incompressible_case(mesh, bfs_bcs; iterations=ITERS, kwargs...)
    run!(model, config)
    fields(model)
end

gmesh = rank == 0 ? bfs_mesh() : nothing
ref = MPI.bcast(rank == 0 ? (serial_run(gmesh; SETUP...), serial_run(gmesh)) : nothing, comm; root=0)
dm = distribute(gmesh; comm=comm)

model, config = incompressible_case(dm, bfs_bcs; iterations=ITERS, SETUP...)
run!(model, config)
dux, duy, dp = field_errors(dm, model, ref[1]...)
rank == 0 && println("MIXED Float32 n=$(MPI.Comm_size(comm)) vs serial mixed: dux=$dux duy=$duy dp=$dp; " *
    "vs F64: $(field_errors(dm, model, ref[2]...))")

# PETSc's and Krylov.jl's Float32 Cg take slightly different paths under the loose rtol (~1e-5)
@testset "MixedF32() distributed BFS (rank $rank)" begin
    @test dux < 5e-5
    @test duy < 5e-5
    @test dp < 5e-5
    @test_throws ArgumentError incompressible_case(dm, bfs_bcs; iterations=1,
        precision=MixedBF16()) |> c -> run!(c...)
end
