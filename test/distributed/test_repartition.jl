# Cell migration behind repartition: moving a Metis distribution to another partition must give the part
# extraction gives for that partition, and solve like serial. The PETSc partitioner needs PT-Scotch or ParMETIS, so
# the target here is an x-slab partition computed on every rank.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

gmesh = bfs_mesh()
order = sortperm([c.centre[1] for c ∈ gmesh.cells])
slabs = zeros(Int, length(order))
for (i, c) ∈ enumerate(order)
    slabs[c] = 1 + (i - 1) * nranks ÷ length(order)
end

dm0 = distribute(rank == 0 ? gmesh : nothing; comm)
dm = XCALibre.Distribute._migrate(dm0, slabs[dm0.orig_cells[1:dm0.partition.n_owned]] .- 1)
ref = extract_subdomain(gmesh, slabs, rank + 1; comm)

@testset "migrate == extract (rank $rank)" begin
    for k ∈ fieldnames(Partition)
        @test getfield(dm.partition, k) == getfield(ref.partition, k)
    end
    @test dm.orig_cells == ref.orig_cells
    @test [(pp.neighbour, pp.send_cells, pp.recv_ghosts, length(pp.faces)) for pp ∈ dm.procs] ==
          [(pp.neighbour, pp.send_cells, pp.recv_ghosts, length(pp.faces)) for pp ∈ ref.procs]
    @test [(c.centre, c.volume) for c ∈ dm.cells] == [(c.centre, c.volume) for c ∈ ref.cells]
    @test [(b.name, length(b.IDs_range)) for b ∈ dm.boundaries] == [(b.name, length(b.IDs_range)) for b ∈ ref.boundaries]
    @test length(dm.faces) == length(ref.faces) && length(dm.nodes) <= length(ref.nodes)
    @test sort([f.area for f ∈ dm.faces]) == sort([f.area for f ∈ ref.faces])
    @test sort(dm.orig_cells[dm.boundary_cellsID]) == sort(ref.orig_cells[ref.boundary_cellsID])
end

iterations = 100
sref = if rank == 0
    m, c = incompressible_case(gmesh, bfs_bcs; iterations)
    run!(m, c)
    (collect(m.momentum.U.x.values), collect(m.momentum.U.y.values), collect(m.momentum.p.values))
end
Us_x, Us_y, ps = MPI.bcast(sref, comm; root=0)
model, config = incompressible_case(dm, bfs_bcs; iterations)
run!(model, config)
dux, duy, dp = field_errors(dm, model, Us_x, Us_y, ps)
@testset "migrated mesh solves like serial (rank $rank)" begin
    @test dux < 1e-6 && duy < 1e-6 && dp < 1e-6
    @test check_ghosts(model.momentum.p, dm, config) == 0
end
rank == 0 && println("REPARTITION migrate n=$nranks dux=$dux duy=$duy dp=$dp")

# the PETSc partitioner: balanced where the build has PT-Scotch, the build hint where it has not
@testset "repartition (rank $rank)" begin
    r = try repartition(dm0) catch e e end
    if r isa Exception
        @test r isa ErrorException && occursin("ptscotch", r.msg)
    else
        @test MPI.Allreduce(r.partition.n_owned, +, comm) == length(gmesh.cells)
        @test sort(MPI.Allgather(r.partition.n_owned, comm))[end] <= 1.1 * length(gmesh.cells) / nranks
    end
end
