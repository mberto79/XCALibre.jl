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
unv3_mesh = mesh
msg = IOBuffer(); println(msg, mesh)
outputTest_UNV3D = String(take!(msg))

@test outputTest_UNV3D == "3D Mesh with:\n-> 125 cells\n-> 450 faces\n-> 216 nodes\n"

# 3D FOAM cavity mesh
meshFile = joinpath(test_grids_dir, "OF_cavity_hex", "polyMesh")
mesh = FOAM3D_mesh(meshFile, scale=1.0)
foam3_mesh = mesh
msg = IOBuffer(); println(msg, mesh)
outputTest_FOAM3D = String(take!(msg))

@test outputTest_FOAM3D == "3D Mesh with:\n-> 125 cells\n-> 450 faces\n-> 216 nodes\n"

@testset "OpenFOAM boundary groups are ignored" begin
    boundaries = XCALibre.FoamMesh.read_boundary(
        joinpath(test_grids_dir, "OF_cavity_hex", "polyMesh", "boundary"),
        Int32,
        Float64,
    )
    @test getproperty.(boundaries, :name) == [:walls, :top]
    @test getproperty.(boundaries, :nFaces) == Int32[125, 25]
    @test getproperty.(boundaries, :startFace) == Int32[301, 426]

    mktempdir() do directory
        malformed = joinpath(directory, "boundary")
        write(malformed, """
        FoamFile { version 2.0; class polyBoundaryMesh; object boundary; }
        2
        (
            onlyPatch
            {
                type patch;
                nFaces 1;
                startFace 0;
            }
        )
        """)
        @test_throws ArgumentError XCALibre.FoamMesh.read_boundary(
            malformed,
            Int32,
            Float64,
        )
    end
end

@testset "OpenFOAM polyhedral geometry" begin
    warped_nodes = [
        XCALibre.Mesh.Node(SVector(0.0, 0.0, 0.0), 0:0),
        XCALibre.Mesh.Node(SVector(2.0, 0.0, 0.2), 0:0),
        XCALibre.Mesh.Node(SVector(2.0, 1.0, 0.0), 0:0),
        XCALibre.Mesh.Node(SVector(0.0, 1.0, -0.1), 0:0),
    ]
    geometry1 = XCALibre.Mesh.face_geometry(
        warped_nodes,
        1:4,
        SVector(1.0, 0.5, 0.025),
    )
    @test geometry1[1] ≈ SVector(
        -0.07396705090823069,
        0.14793410181646136,
        0.9862273454430758,
    )
    @test geometry1[2] ≈ 2.027929979067325
    @test geometry1[3] ≈ SVector(
        1.0024316109422493,
        0.49969604863221884,
        0.025227963525835864,
    )

    mktempdir() do directory
        write(joinpath(directory, "points"), """
        12
        (
        (-1 -1 0)
        (1 -1 0)
        (1 1 0)
        (-1 1 0)
        (-0.5 -0.5 1)
        (0.5 -0.5 1)
        (0.5 0.5 1)
        (-0.5 0.5 1)
        (-0.5 -0.5 2)
        (0.5 -0.5 2)
        (0.5 0.5 2)
        (-0.5 0.5 2)
        )
        """)
        write(joinpath(directory, "faces"), """
        11
        (
        4(4 5 6 7)
        4(0 3 2 1)
        4(0 1 5 4)
        4(1 2 6 5)
        4(2 3 7 6)
        4(3 0 4 7)
        4(8 9 10 11)
        4(4 5 9 8)
        4(5 6 10 9)
        4(6 7 11 10)
        4(7 4 8 11)
        )
        """)
        write(joinpath(directory, "owner"), """
        11
        (
        0 0 0 0 0 0 1 1 1 1 1
        )
        """)
        write(joinpath(directory, "neighbour"), """
        1
        (
        1
        )
        """)
        write(joinpath(directory, "boundary"), """
        1
        (
            walls
            {
                type wall;
                nFaces 10;
                startFace 1;
            }
        )
        """)

        skew_mesh = FOAM3D_mesh(directory)
        @test skew_mesh.cells[1].centre ≈ SVector(0.0, 0.0, 11/28)
        @test skew_mesh.cells[1].volume ≈ 7/3
        @test skew_mesh.cells[2].centre ≈ SVector(0.0, 0.0, 1.5)
        @test skew_mesh.cells[2].volume ≈ 1.0
    end
end

@testset "boundary assignment requires each patch exactly once" begin
    valid = assign(
        region=mesh,
        (T=[Dirichlet(:walls, 0.0), Dirichlet(:top, 1.0)],),
    )
    @test getproperty.(valid.T, :ID) == (1, 2)

    duplicate_error = try
        assign(
            region=mesh,
            (T=[Dirichlet(:walls, 0.0), Dirichlet(:walls, 1.0)],),
        )
        nothing
    catch error
        error
    end
    @test duplicate_error isa ArgumentError
    @test contains(sprint(showerror, duplicate_error), "missing top")
    @test contains(sprint(showerror, duplicate_error), "assigned more than once walls")
end

# Test 3D UNV and FOAM meshes are equal
@test outputTest_UNV3D == outputTest_FOAM3D
@test getproperty.(unv3_mesh.cells, :centre) ≈ getproperty.(foam3_mesh.cells, :centre)
@test getproperty.(unv3_mesh.cells, :volume) ≈ getproperty.(foam3_mesh.cells, :volume)

@testset "OpenFOAM writer preserves input mesh and coordinate precision" begin
    mktempdir() do directory
        cd(directory) do
            mesh_directory = joinpath("constant", "polyMesh")
            mkpath(mesh_directory)
            mesh_files = ("points", "faces", "owner", "neighbour", "boundary")
            for name in mesh_files
                write(joinpath(mesh_directory, name), "sentinel-$name")
            end

            XCALibre.initialise_writer(OpenFOAM(), foam3_mesh)
            @test all(
                read(joinpath(mesh_directory, name), String) == "sentinel-$name"
                for name in mesh_files
            )
        end

        generated_directory = joinpath(directory, "generated")
        mkpath(generated_directory)
        cd(generated_directory) do
            XCALibre.initialise_writer(OpenFOAM(), foam3_mesh)
            written_mesh = FOAM3D_mesh(joinpath("constant", "polyMesh"))
            @test getproperty.(written_mesh.nodes, :coords) ==
                getproperty.(foam3_mesh.nodes, :coords)
        end
    end
end

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
