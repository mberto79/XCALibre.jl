test_grids_dir = pkgdir(XCALibre, "test", "grids")

function test_mesh_precision(mesh, integer_type, float_type)
    @test eltype(mesh.get_int) === integer_type
    @test eltype(mesh.get_float) === float_type
    @test eltype(mesh.cell_nodes) === integer_type
    @test eltype(mesh.cell_faces) === integer_type
    @test eltype(mesh.cell_neighbours) === integer_type
    @test eltype(mesh.cell_nsign) === integer_type
    @test eltype(mesh.face_nodes) === integer_type
    @test eltype(mesh.node_cells) === integer_type
    @test eltype(mesh.boundary_cellsID) === integer_type
    @test eltype(mesh.cells[1].nodes_range) === integer_type
    @test eltype(mesh.cells[1].faces_range) === integer_type
    @test eltype(mesh.cells[1].centre) === float_type
    @test typeof(mesh.cells[1].volume) === float_type
    @test eltype(mesh.faces[1].nodes_range) === integer_type
    @test eltype(mesh.faces[1].ownerCells) === integer_type
    @test eltype(mesh.faces[1].centre) === float_type
    @test eltype(mesh.faces[1].normal) === float_type
    @test typeof(mesh.faces[1].area) === float_type
    @test eltype(mesh.boundaries[1].IDs_range) === integer_type
    @test eltype(mesh.nodes[1].coords) === float_type
    @test eltype(mesh.nodes[1].cells_range) === integer_type
end

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

precision_cases = (
    (Int32, Float32),
    (Int64, Float32),
    (Int32, Float64),
)

mesh_converters = (
    ("UNV2D", UNV2D_mesh, joinpath(test_grids_dir, "quad40.unv")),
    ("UNV3D", UNV3D_mesh, joinpath(test_grids_dir, "OF_cavity_hex", "cavity_hex.unv")),
    ("FOAM3D", FOAM3D_mesh, joinpath(test_grids_dir, "OF_cavity_hex", "polyMesh")),
)

for (name, converter, meshFile) in mesh_converters
    @testset "$name precision options" begin
        for (integer_type, float_type) in precision_cases
            mesh = converter(
                meshFile;
                scale=float_type(0.001),
                integer_type=integer_type,
                float_type=float_type,
            )
            test_mesh_precision(mesh, integer_type, float_type)
        end
    end
end

@testset "single precision mesh validation" begin
    mesh = FOAM3D_mesh(
        joinpath(test_grids_dir, "OF_cavity_hex", "polyMesh");
        scale=Float32(0.001),
        integer_type=Int32,
        float_type=Float32,
    )
    cell = mesh.cells[1]
    mesh.cells[1] = typeof(cell)(
        cell.centre,
        -abs(cell.volume),
        cell.nodes_range,
        cell.faces_range,
    )
    @test_throws ArgumentError XCALibre.Mesh.validate_single_precision_mesh(mesh; source="test")
end
