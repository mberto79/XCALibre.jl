using XCALibre
using StaticArrays
using SparseArrays, SparseMatricesCSR
using Test

# bounding_box must start from ±Inf (not the origin), be race-free, and work on a mesh or
# partition without boundary faces; the LES Empty-patch distance must not depend on where the
# mesh sits relative to the origin.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
pitz() = FOAM3D_mesh(joinpath(grids_dir, "OF_pitzDaily", "polyMesh"), scale=1)

pitz_bcs(mesh) = assign(region=mesh, (U = [Dirichlet(:inlet, [1.0, 0.0, 0.0]),
    Zerogradient(:outlet), Wall(:upperWall, [0.0, 0.0, 0.0]), Wall(:lowerWall, [0.0, 0.0, 0.0]),
    Empty(:frontAndBack)],)).U

coord_extrema(mesh) = (
    SVector{3}(ntuple(d -> minimum(c[d] for c ∈ mesh.node_coords), 3)),
    SVector{3}(ntuple(d -> maximum(c[d] for c ∈ mesh.node_coords), 3)))

@testset "bounding_box and LES Empty distance" begin
    mesh = pitz()
    pmin, pmax = bounding_box(mesh)
    @test (pmin, pmax) == coord_extrema(mesh)

    offset = SVector(1.0, 2.0, 3.0)
    moved = pitz()
    moved.node_coords .+= Ref(offset)
    mmin, mmax = bounding_box(moved)
    @test all(mmin .> 0) # the origin lies outside this mesh
    @test mmin ≈ pmin + offset atol=1e-12
    @test mmax ≈ pmax + offset atol=1e-12

    thickness = pmax[3] - pmin[3]
    for m ∈ (mesh, moved)
        distance, power = XCALibre.ModelPhysics.delta_scaling(m, pitz_bcs(m))
        @test distance ≈ thickness rtol=1e-12
        @test power == 0.5
    end

    # a partition holding no boundary face (an interior MPI rank)
    m = mesh
    interior = Mesh3(m.cells, m.cell_nodes, m.cell_faces, m.cell_neighbours, m.cell_nsign,
        m.faces, m.face_nodes, similar(m.boundaries, 0), m.nodes, m.node_cells,
        m.get_float, m.get_int, m.boundary_cellsID)
    @test total_boundary_faces(interior) == 0
    @test bounding_box(interior) == (pmin, pmax)
    @test XCALibre.ModelPhysics.delta_scaling(interior, ()) == (1.0, 1/3)

    f32 = FOAM3D_mesh(joinpath(grids_dir, "OF_pitzDaily", "polyMesh"), scale=1, float_type=Float32)
    @test eltype(bounding_box(f32)[1]) == Float32
end

@testset "DILU diagonal extraction on an empty matrix" begin
    A = sparsecsr(Int[], Int[], Float64[], 0, 0)
    config = (; hardware = (backend=CPU(), workgroup=32))
    D = Float64[]
    XCALibre.Solve.extract_diagonal!(D, Int[], A, config)
    @test isempty(D)
end
