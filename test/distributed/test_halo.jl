# Halo tests; run per rank under mpiexec (see runtests_mpi.jl)
using XCALibre, MPI, Test
using LinearAlgebra, Random, StaticArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
gmesh = rank == 0 ?
    UNV3D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "3d_box_1000x1000x1000mm_10.unv"), scale=0.001) :
    nothing
dm = distribute(gmesh; comm=comm)
backend, workgroup = CPU(), 64
n_owned = dm.partition.n_owned
nloc = n_owned + dm.partition.n_ghost
ghosts = n_owned+1:nloc
f(x, y, z) = 1.3x + 2.7y - 0.9z + 0.31

# serial references computed on the global mesh, broadcast to all ranks
ref = rank == 0 ?
    (norm(f(c.centre...) for c ∈ gmesh.cells),
     sum(f(c.centre...)^2 * 2.0 for c ∈ gmesh.cells),
     sum(f(c.centre...) for c ∈ gmesh.cells) / length(gmesh.cells)) :
    nothing
ref_norm, ref_dot, ref_mean = MPI.bcast(ref, comm; root=0)

@testset "Phase 2 halo (rank $rank)" begin
    centres = dm.mesh.cells

    # scalar exchange: ghost values equal neighbour's owned centre x-coordinate bitwise
    phi = ScalarField(dm)
    H1 = HaloExchange(dm, 1, backend; comm=comm)
    for i ∈ 1:n_owned
        phi[i] = centres[i].centre[1]
    end
    phi.values[ghosts] .= NaN
    halo_exchange!(phi, H1, backend, workgroup)
    @test all(phi[i] === centres[i].centre[1] for i ∈ ghosts)

    # linear field through sync!: ghosts match analytic values exactly
    dphi = DistributedScalarField(dm, backend; comm=comm)
    initialise!(dphi, (x, y, z) -> f(x, y, z))
    dphi.field.values[ghosts] .= NaN
    config = (; hardware=(; backend, workgroup))
    sync!(dphi, config)
    @test all(dphi.field[i] === f(centres[i].centre...) for i ∈ ghosts)

    # idempotency: second exchange leaves values bitwise identical
    before = copy(dphi.field.values)
    sync!(dphi, config)
    @test dphi.field.values == before

    # vector exchange through one 3-wide buffer
    dU = DistributedVectorField(dm, backend; comm=comm)
    for i ∈ 1:n_owned
        c = centres[i].centre
        dU.field[i] = SVector{3}(c[1], 2c[2], -c[3])
    end
    sync!(dU, config)
    @test all(dU.field[i] == SVector{3}(centres[i].centre[1], 2centres[i].centre[2], -centres[i].centre[3]) for i ∈ ghosts)

    # global reductions over owned entries match serial values
    dpsi = DistributedScalarField(dm, backend; comm=comm)
    dpsi.field.values .= 2 .* dphi.field.values
    @test pnorm(dphi) ≈ ref_norm rtol = 1e-14
    @test pdot(dphi, dpsi) ≈ ref_dot rtol = 1e-14
    @test pmean(dphi) ≈ ref_mean rtol = 1e-14

    # adjoint identity: ⟨v̄, Hx⟩ == ⟨Hᵀv̄, x⟩ globally
    rng = Xoshiro(42 + rank)
    x = ScalarField(dm); v = ScalarField(dm)
    x.values .= rand(rng, nloc); v.values .= rand(rng, nloc)
    hx = ScalarField(dm)
    hx.values .= x.values
    halo_exchange!(hx, H1, backend, workgroup)
    lhs = MPI.Allreduce(dot(v.values, hx.values), +, comm)
    w = ScalarField(dm)
    w.values .= v.values
    XCALibre.Distribute.halo_exchange_adjoint!(w, H1, backend, workgroup)
    @test all(w.values[ghosts] .== 0)
    rhs = MPI.Allreduce(dot(w.values, x.values), +, comm)
    @test lhs ≈ rhs rtol = 1e-12

    # buffer/request reuse: allocations steady after warmup
    halo_exchange!(phi, H1, backend, workgroup)
    a1 = @allocated halo_exchange!(phi, H1, backend, workgroup)
    a2 = @allocated halo_exchange!(phi, H1, backend, workgroup)
    @test a2 <= a1
end

# the mesh carries its communicator: a duplicated comm gives bitwise the same ghosts through the
# self-syncing seam and the reductions, with nothing reading COMM_WORLD by name
comm2 = MPI.Comm_dup(comm)
dm2 = distribute(gmesh; comm=comm2)
@testset "communicator on the mesh (rank $rank)" begin
    @test dm2.comm == comm2
    @test dm2.orig_cells == dm.orig_cells
    phi1, phi2 = ScalarField(dm), ScalarField(dm2)
    for i ∈ 1:n_owned
        phi1[i] = f(dm.mesh.cells[i].centre...); phi2[i] = phi1[i]
    end
    phi1.values[ghosts] .= NaN; phi2.values[ghosts] .= NaN
    config = (; hardware=(; backend, workgroup))
    sync!(phi1, dm, config); sync!(phi2, dm2, config)
    @test phi2.values == phi1.values
    @test dm2.halos.w1.comm == comm2
    @test XCALibre.Solvers.global_max(Float64(rank + 1), dm2) == MPI.Comm_size(comm2)
    @test XCALibre.Solve.is_report_rank(dm2) == (rank == 0)
    @test (gather(phi2, dm2) === nothing) == (rank != 0)
end
