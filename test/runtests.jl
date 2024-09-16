using XCALibre
using Test

TEST_CASES_DIR = pkgdir(XCALibre, "test/0_TEST_CASES")

@testset "Mesh conversion" begin
    include("test_mesh_conversion.jl")
end

@testset "Incompressible (steady)" begin
    test_name = "2d_incompressible_laminar_BFSjl.jl"
    test_file = joinpath(TEST_CASES_DIR, test_name)
    include(test_file)

    test_name = "2d_incompressible_flatplate_KOmega_lowRe.jl"
    test_file = joinpath(TEST_CASES_DIR, test_name)
    include(test_file)

    test_name = "2d_incompressible_flatplate_KOmega_HighRe.jl"
    test_file = joinpath(TEST_CASES_DIR, test_name)
    include(test_file)
end

@testset "Gradient schemes" begin
    # Write your own tests here.
    @test 2 + 2 == 4
    @test 2 + 2 == 4
    @test 2 + 2 == 4
end

@testset "Upwind schemes" begin
    # Write your own tests here.
    @test 2 + 2 == 4
    @test 2 + 2 == 4
    @test 2 + 2 == 4
end



@testset "Incompressible (transient)" begin
    # Write your own tests here.
    @test 2 + 2 == 4
end

@testset "Compressible (steady)" begin
    # Write your own tests here.
    @test 2 + 2 == 4
end

@testset "Compressible (transient)" begin
    # Write your own tests here.
    @test 2 + 2 == 4
end

@testset "Boundary conditions" begin
    # Write your own tests here.
    @test 2 + 2 == 4
end