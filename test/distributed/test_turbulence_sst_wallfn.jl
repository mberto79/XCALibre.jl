# Distributed KOmegaSST with wall functions vs serial simple!, per rank under mpiexec.
# :wall and :top carry K/Omega/Nut wall functions. At three ranks the
# partition is three y-bands, so :top and :wall each empty out on some rank by construction.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "sst_case.jl"))

iterations = 50
wf_opts = (walls=(:wall, :top), init=(wf_vel[1], wf_k, wf_w))

# every rank holds the small global mesh: wall cells (y < 0.01) | middle band | top cells (y > 0.05)
gmesh = bfs_mesh()
nranks = MPI.Comm_size(comm)
if nranks != 3
    @testset "KOmegaSST wall functions distributed (rank $rank)" begin
        @test_skip "the y-band partition needs exactly 3 ranks"
    end
else
parts = [(y = c.centre[2]; y < 0.01 ? 1 : y < 0.05 ? 2 : 3) for c ∈ gmesh.cells]
ref = if rank == 0
    ms, cs = sst_case(gmesh, bfs_sst_wallfn_bcs; iterations, wf_opts...)
    simple!(ms, cs)
    (collect(ms.momentum.U.x.values), collect(ms.momentum.U.y.values),
     collect(ms.momentum.p.values), collect(ms.turbulence.k.values),
     collect(ms.turbulence.omega.values), collect(ms.turbulence.nut.values))
else
    nothing
end
Us_x, Us_y, ps, ks, ws, nuts = MPI.bcast(ref, comm; root=0)

dm = extract_subdomain(gmesh, parts, rank + 1; comm)
model, config = sst_case(dm, bfs_sst_wallfn_bcs; iterations, wf_opts...)
run!(model, config)

n = dm.partition.n_owned
orig = dm.orig_cells
owned_err(loc, ser) = maximum(abs.(Array(loc)[1:n] .- ser[orig[1:n]]); init=0.0)
scaled(loc, ser) = owned_err(loc, ser) / max(maximum(abs.(ser)), eps())

# every rank must run the wall functions, including one holding a patch with no faces
local_patches = Dict(String(b.name) => length(b.IDs_range) for b in dm.mesh.boundaries)
n_empty(name) = MPI.Allreduce(Int(local_patches[name] == 0), +, comm)

dux = scaled(model.momentum.U.x.values, Us_x)
duy = scaled(model.momentum.U.y.values, Us_y)
dp = scaled(model.momentum.p.values, ps)
dk = scaled(model.turbulence.k.values, ks)
dw = scaled(model.turbulence.omega.values, ws)
dnut = scaled(model.turbulence.nut.values, nuts)

@testset "KOmegaSST wall functions distributed (rank $rank)" begin
    @test dux < 1e-6
    @test duy < 1e-6
    @test dp < 1e-6
    @test dk < 1e-6
    @test dw < 1e-6
    @test dnut < 1e-6
    # non-vacuous: some rank really holds a wall-function patch with no faces
    @test n_empty("wall") >= 1
    @test n_empty("top") >= 1
    # non-vacuous: the wall functions actually wrote something
    @test any(>(0), Array(model.turbulence.nut.values)[1:n])
end
rank == 0 && println("SST wallfn bfs n=$(MPI.Comm_size(comm)) dux=$dux dp=$dp dk=$dk dw=$dw dnut=$dnut")
println("  rank $rank patches: ", join(["$k=$v" for (k,v) in sort(collect(local_patches))], " "))
end
