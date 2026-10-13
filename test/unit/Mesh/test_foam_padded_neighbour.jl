using XCALibre
using Test

# Some mesh generators write the neighbour list with one entry per face, padding boundary faces
# with -1. Such a mesh must load exactly as the same mesh with the short (internal-only) list.

let src = pkgdir(XCALibre, "test", "grids", "OF_cavity_hex", "polyMesh")
    read_quiet(dir) = redirect_stdout(() -> FOAM3D_mesh(dir, scale=0.001), devnull)
    ref = read_quiet(src)

    owner = redirect_stdout(() -> XCALibre.FoamMesh.read_owner(joinpath(src, "owner"), Int, Float64), devnull)
    neighbour = redirect_stdout(
        () -> XCALibre.FoamMesh.read_neighbour(joinpath(src, "neighbour"), Int, Float64), devnull)
    n_faces, n_ifaces = length(owner), length(neighbour)
    padded = vcat(neighbour .- 1, fill(-1, n_faces - n_ifaces)) # zero-based labels as in the file

    header(format, arch) = """
        FoamFile
        {
            version     2.0;
            format      $format;
            $(arch)class       labelList;
            location    "constant/polyMesh";
            object      neighbour;
        }
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        """

    function padded_copy(write_neighbour)
        dir = mktempdir()
        for name in ("points", "faces", "owner", "boundary")
            Base.cp(joinpath(src, name), joinpath(dir, name)) # qualified: keeps Main.cp free (Julia 1.10)
        end
        open(write_neighbour, joinpath(dir, "neighbour"), "w")
        return dir
    end

    ascii_dir = padded_copy() do io
        print(io, header("ascii", ""), n_faces, "\n(\n")
        foreach(v -> println(io, v), padded)
        print(io, ")\n")
    end
    binary_dir = padded_copy() do io
        print(io, header("binary", "arch        \"LSB;label=32;scalar=64\";\n    "), n_faces, "\n(")
        write(io, Int32.(padded))
        print(io, ")\n")
    end

    for dir in (ascii_dir, binary_dir)
        mesh = read_quiet(dir)
        @test length(mesh.cells) == length(ref.cells)
        @test length(mesh.faces) == length(ref.faces)
        @test count(f -> f.ownerCells[1] != f.ownerCells[2], mesh.faces) == n_ifaces
        @test mesh.boundary_cellsID == ref.boundary_cellsID
        @test [f.ownerCells for f in mesh.faces] == [f.ownerCells for f in ref.faces]
        @test [c.volume for c in mesh.cells] == [c.volume for c in ref.cells]
        @test mesh.cell_neighbours == ref.cell_neighbours
        @test [b.IDs_range for b in mesh.boundaries] == [b.IDs_range for b in ref.boundaries]
        @test !redirect_stdout(() -> XCALibre.IOFormats._polyMesh_order_mismatch(dir, mesh), devnull)
    end
end
