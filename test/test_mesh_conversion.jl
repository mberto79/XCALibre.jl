using Test

test_grids_dir = pkgdir(XCALibre, "test", "grids")

meshFile = joinpath(test_grids_dir, "trig40.unv")
mesh = UNV2D_mesh(meshFile, scale=0.001)
msg = IOBuffer(); println(msg, mesh)
outputTest = String(take!(msg))

@test outputTest == "2D Mesh\n-> 3484 cells\n-> 5306 faces\n-> 1823 nodes\n\nBoundaries \n-> inlet (faces: 1:40)\n-> outlet (faces: 41:80)\n-> bottom (faces: 81:120)\n-> top (faces: 121:160)\n\n"