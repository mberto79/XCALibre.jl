using XCALibre
using Accessors
using LinearAlgebra
using Test

# A reordered mesh is the same mesh numbered differently: a solve on it must match the solve on the
# original mesh once permuted, and its connectivity must stay consistent.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")

function reorder_solve_T(mesh)
    hardware = Hardware(backend=CPU(), workgroup=1024)
    model = Physics(time=Steady(), solid=Solid{Uniform}(k=1.0), energy=Energy{Conduction}(), domain=mesh)
    BCs = assign(region=mesh, (T = [
        Dirichlet(:inlet, 50.0), Zerogradient(:outlet), Dirichlet(:bottom, 10.0), Zerogradient(:top)],))
    schemes = (T = Schemes(laplacian = Linear),)
    solvers = (T = SolverSetup(
        solver=Cg(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0, rtol=1e-13, atol=1e-14, itmax=20000),)
    config = Configuration(solvers=solvers, schemes=schemes,
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1), hardware=hardware, boundaries=BCs)
    T = model.energy.T
    T_eqn = (
        - Laplacian{schemes.T.laplacian}(model.solid.rDf, T) == - Source(ScalarField(mesh))
    ) → ScalarEquation(T, config.boundaries.T)
    initialise!(T, 0.0)
    discretise!(T_eqn, T, config)
    apply_boundary_conditions!(T_eqn, config.boundaries.T, nothing, 0.0, config)
    @reset T_eqn.preconditioner = set_preconditioner(solvers.T.preconditioner, T_eqn)
    @reset T_eqn.solver = _workspace(solvers.T.solver, _b(T_eqn))
    update_preconditioner!(T_eqn.preconditioner, mesh, config)
    solve_system!(T_eqn, solvers.T, T, nothing, config)
    copy(T.values)
end

function consistent_connectivity(m)
    owners = m.face_ownerCells
    all(begin
        o = owners[m.cell_faces[j]]
        m.cell_neighbours[j] == (o[1] == i ? o[2] : o[1]) && m.cell_nsign[j] == (o[1] == i ? 1 : -1)
    end for i ∈ eachindex(m.cell_volume) for j ∈ m.cell_faces_range[i]) &&
    all(i ∈ view(m.node_cells, m.node_cells_range[n])
        for i ∈ eachindex(m.cell_volume) for n ∈ view(m.cell_nodes, m.cell_nodes_range[i])) &&
    all(m.boundary_cellsID[f] == owners[f][1] for f ∈ eachindex(m.boundary_cellsID))
end

neighbour_distance(m) = sum(f -> abs(Int(f[1]) - Int(f[2])), m.face_ownerCells[length(m.boundary_cellsID)+1:end])

@testset "reorder_mesh! keeps the solution ($method)" for method ∈ (:rcm, :morton)
    mesh = UNV2D_mesh(joinpath(grids_dir, "trig100.unv"); reorder=false)
    reordered = deepcopy(mesh)
    perm = XCALibre.Mesh._reorder_mesh!(reordered, method)
    @test perm !== nothing
    @test reorder_solve_T(reordered) ≈ reorder_solve_T(mesh)[perm] rtol=1e-10
    @test consistent_connectivity(reordered)
end

@testset "reorder_mesh! on a 3D mesh" begin
    mesh = UNV3D_mesh(joinpath(grids_dir, "bfs_unv_tet_15mm.unv"); scale=0.001, reorder=false)
    reordered = deepcopy(mesh)
    perm = XCALibre.Mesh._reorder_mesh!(reordered, :rcm)
    @test reordered.cell_volume == mesh.cell_volume[perm]
    @test sort(reordered.node_coords) == sort(mesh.node_coords)
    @test [b.IDs_range for b ∈ reordered.boundaries] == [b.IDs_range for b ∈ mesh.boundaries]
    @test consistent_connectivity(reordered)
    @test neighbour_distance(reordered) < neighbour_distance(mesh)/10
    # an ordered mesh is left as it is, so reading back a mesh written after reordering is stable
    @test XCALibre.Mesh._reorder_mesh!(reordered, :rcm) === nothing
    @test UNV3D_mesh(joinpath(grids_dir, "bfs_unv_tet_15mm.unv"); scale=0.001).cell_volume == reordered.cell_volume
end

@testset "polyMesh written from a reordered mesh" begin
    mesh = UNV3D_mesh(joinpath(grids_dir, "bfs_unv_tet_15mm.unv"); scale=0.001, reorder=false)
    reordered = reorder_mesh!(deepcopy(mesh))
    mktempdir() do dir
        cd(dir) do
            IOFormats = XCALibre.IOFormats
            IOFormats.initialise_writer(OpenFOAM(), mesh)
            @test !IOFormats._polyMesh_order_mismatch("constant/polyMesh", mesh)
            @test IOFormats._polyMesh_order_mismatch("constant/polyMesh", reordered)
            # same counts, different numbering: the files are rewritten in the order of the mesh
            IOFormats.initialise_writer(OpenFOAM(), reordered)
            back = FOAM3D_mesh("constant/polyMesh"; reorder=false)
            @test back.cell_volume ≈ reordered.cell_volume
            @test !IOFormats._polyMesh_order_mismatch("constant/polyMesh", reordered)
            @test consistent_connectivity(back)
        end
    end
end
