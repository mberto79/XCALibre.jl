using XCALibre
using Test

@testset "Mesh conversion" begin
    include("test_mesh_conversion.jl")
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

@testset "Incompressible (steady)" begin
    # Write your own tests here.
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


a = """2D Mesh\n-> 1800 cells\n-> 3720 faces\n-> 1921 nodes\n\nBoundaries \n-> inlet (faces: 1:10)\n-> outlet (faces: 11:30)\n-> wall (faces: 31:140)\n-> top (faces: 141:240)\n\n\n"""

b = """2D Mesh\n-> 1800 cells\n-> 3720 faces\n-> 1921 nodes\n\nBoundaries\n-> inlet (faces: 1:10)\n-> outlet (faces: 11:30)\n-> wall (faces: 31:140)\n-> top (faces: 141:240)\n\n\n"""

a == b