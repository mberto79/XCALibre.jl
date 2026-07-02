# Phase 2 halo tests; run per rank under mpiexec (see runtests_mpi.jl)
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
    halo_exchange_adjoint!(w, H1, backend, workgroup)
    @test all(w.values[ghosts] .== 0)
    rhs = MPI.Allreduce(dot(w.values, x.values), +, comm)
    @test lhs ≈ rhs rtol = 1e-12

    # buffer/request reuse: allocations steady after warmup
    halo_exchange!(phi, H1, backend, workgroup)
    a1 = @allocated halo_exchange!(phi, H1, backend, workgroup)
    a2 = @allocated halo_exchange!(phi, H1, backend, workgroup)
    @test a2 <= a1
end
