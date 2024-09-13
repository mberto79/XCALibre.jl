using XCALibre
using Test

@testset "Mesh conversion" begin
    examples_dir = pkgdir(XCALibre, "examples")
    unv2_file = joinpath(
        examples_dir, 
        "2d_incompressible_laminar_backwards_step/backward_facing_step_10mm.unv"
        )
    mesh = UNV2D_mesh(unv2_file, scale=0.001)
    msg = IOBuffer(); println(msg, mesh)
    outputTest = String(take!(msg))
    @testset "UNV" begin
        @test outputTest == """
2D Mesh
-> 1800 cells
-> 3720 faces
-> 1921 nodes

Boundaries
-> inlet (faces: 1:10)
-> outlet (faces: 11:30)
-> wall (faces: 31:140)
-> top (faces: 141:240)
"""
    end
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