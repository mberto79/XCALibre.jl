# A solver or model with no distributed linear-solve seam must error rather than let each rank
# solve its own block. The supported set is declared by `distributed_ready` methods.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

check = XCALibre.Solvers.check_distributed_support

gmesh = rank == 0 ? bfs_mesh() : nothing
dm = distribute(gmesh; comm=comm)
model, config = incompressible_case(dm, bfs_bcs; iterations=1)

les_model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu=1e-3),
    turbulence = LES{Smagorinsky}(),
    energy = Energy{Isothermal}(),
    domain = dm)

@testset "unsupported distributed combinations error (rank $rank)" begin
    # supported: the set this branch wired and tested
    @test check(:SIMPLE, model) === nothing
    @test check(:PISO, model) === nothing
    @test check(:potential_flow, model) === nothing

    # unsupported solver, supported models
    @test_throws ErrorException check(:CSIMPLE, model)
    @test_throws ErrorException check(:CPISO, model)
    @test_throws ErrorException check(:SIMPLE_MRF, model)
    @test_throws ErrorException check(:Godunov, model)
    @test_throws ErrorException check(:multiphase, model)
    @test_throws ErrorException check(:FilmModel, model)

    # supported solver, unsupported model
    @test_throws ErrorException check(:SIMPLE, les_model)

    # the message names what is missing, so it doubles as the implementation list
    msg = try; check(:CSIMPLE, les_model); ""; catch e; sprint(showerror, e); end
    @test occursin("CSIMPLE", msg) && occursin("Smagorinsky", msg)
end

# a serial mesh is never refused
if rank == 0
    smodel, _ = incompressible_case(gmesh, bfs_bcs; iterations=1)
    @testset "serial meshes are not refused" begin
        @test check(:CSIMPLE, smodel) === nothing
        @test check(:Godunov, smodel) === nothing
    end
end
