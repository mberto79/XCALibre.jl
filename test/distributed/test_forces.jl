# pressure_force/viscous_force on a distributed mesh: each rank integrates its own patch faces and
# the sum over ranks matches the serial integral, also for a patch some ranks hold no faces of.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))

# analytic cell fields, so serial and distributed see the same values with no solve
function set_fields!(model)
    mesh = model.domain
    xc = [c.centre for c ∈ mesh.cells]
    model.momentum.p.values .= [1.0 + 3.0x[1] - 2.0x[2] for x ∈ xc]
    model.momentum.U.x.values .= [0.5 + 40.0x[2] for x ∈ xc]
    model.momentum.U.y.values .= [-20.0x[1] for x ∈ xc]
end

patches = (:wall, :top, :inlet, :outlet)

# viscous_force takes the wall velocity of each face from the U condition of every patch
# (Dirichlet inlet, zero-gradient outlet, wall, symmetry top)
forces(model, config) = [(pressure_force(p, model, config), viscous_force(p, model, config))
    for p ∈ patches]

gmesh = rank == 0 ? bfs_mesh() : nothing
ref = if rank == 0
    ms, cs = incompressible_case(gmesh, bfs_bcs; iterations=1)
    set_fields!(ms)
    forces(ms, cs)
else
    nothing
end
ref = MPI.bcast(ref, comm; root=0)

dm = distribute(gmesh; comm=comm)
model, config = incompressible_case(dm, bfs_bcs; iterations=1)
set_fields!(model)
dist = forces(model, config)

@testset "distributed patch forces (rank $rank)" begin
    for (k, patch) ∈ enumerate(patches)
        (Fp_s, Fv_s), (Fp_d, Fv_d) = ref[k], dist[k]
        @test maximum(abs.(Fp_d .- Fp_s)) <= 1e-12 * max(1.0, maximum(abs.(Fp_s)))
        @test maximum(abs.(Fv_d .- Fv_s)) <= 1e-12 * max(1.0, maximum(abs.(Fv_s)))
    end
end
rank == 0 && println("forces n=$(MPI.Comm_size(comm)): ", dist)
