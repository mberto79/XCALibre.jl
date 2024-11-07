test_grids_dir = pkgdir(XCALibre, "test", "grids")

# test tri mesh
meshFile = joinpath(test_grids_dir, "trig40.unv")
mesh = UNV2D_mesh(meshFile, scale=0.001)
msg = IOBuffer(); println(msg, mesh)
outputTest = String(take!(msg))

@test outputTest == "2D Mesh with:\n-> 3484 cells\n-> 5306 faces\n-> 1823 nodes\n"

# test quad mesh
meshFile = joinpath(test_grids_dir, "quad40.unv")
mesh = UNV2D_mesh(meshFile, scale=0.001)
msg = IOBuffer(); println(msg, mesh)
outputTest = String(take!(msg))

@test outputTest == "2D Mesh with:\n-> 1600 cells\n-> 3280 faces\n-> 1681 nodes\n"

# 3D UNV cavity mesh
meshFile = joinpath(test_grids_dir, "OF_cavity_hex", "cavity_hex.unv")
mesh = UNV3D_mesh(meshFile, scale=0.001)
msg = IOBuffer(); println(msg, mesh)
outputTest_UNV3D = String(take!(msg))

@test outputTest_UNV3D == "3D Mesh with:\n-> 125 cells\n-> 450 faces\n-> 216 nodes\n"

# 3D FOAM cavity mesh
meshFile = joinpath(test_grids_dir, "OF_cavity_hex", "polyMesh")
mesh = FOAM3D_mesh(meshFile, scale=0.001)
msg = IOBuffer(); println(msg, mesh)
outputTest_FOAM3D = String(take!(msg))

@test outputTest_FOAM3D == "3D Mesh with:\n-> 125 cells\n-> 450 faces\n-> 216 nodes\n"

# Test 3D UNV and FOAM meshes are equal
@test outputTest_UNV3D == outputTest_FOAM3D