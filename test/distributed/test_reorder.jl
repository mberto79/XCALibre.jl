# A mesh reordered with reorder_mesh! before distribute gives the serial solution of the same mesh
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "laplace_case.jl"))

trig_mesh() = UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "trig100.unv"))
trig_bcs(mesh) = assign(region=mesh, (
    T = [Dirichlet(:inlet, 50.0), Zerogradient(:outlet), Dirichlet(:bottom, 10.0), Zerogradient(:top)],))

function solve_T(mesh)
    model, config = laplace_case(mesh, trig_bcs)
    run!(model, config)
    model.energy.T.values
end

# serial reference on rank 0: file order and reordered (the same field, renumbered)
ref = if rank == 0
    mesh = trig_mesh()
    reordered = deepcopy(mesh)
    perm = XCALibre.Mesh._reorder_mesh!(reordered, :rcm)
    (collect(solve_T(reordered)), collect(solve_T(mesh))[perm])
else
    nothing
end
Treordered, Tfile = MPI.bcast(ref, comm; root=0)

dm = distribute(() -> reorder_mesh!(trig_mesh(); polymesh=nothing); comm)
Tloc = solve_T(dm)
n = dm.partition.n_owned
orig = dm.orig_cells

@testset "reordered mesh, distributed (rank $rank)" begin
    @test Treordered ≈ Tfile rtol=1e-8
    @test maximum(abs.(Tloc[1:n] .- Treordered[orig[1:n]]); init=0.0) < 1e-6
end
