# Failure on some ranks must raise an error on every rank instead of leaving the others in a
# collective: reader and partition on rank 0, a missing part, a partial restart, a broken processor dir.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)
last_rank = nranks - 1

include(joinpath(@__DIR__, "psimple_case.jl"))

# every rank must reach the barrier after the failing call, or the file hangs until its timeout
fails_everywhere(f) = (t = (try f(); false catch; true end); MPI.Barrier(comm); t)

@testset "rank-0 reader and partition failures (rank $rank)" begin
    @test fails_everywhere(() -> distribute(() -> error("reader failed"); comm))
    tmp = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm; root=0)
    @test fails_everywhere(() -> distribute(() -> error("reader failed"); dir=joinpath(tmp, "p"), comm))
    @test fails_everywhere(() -> distribute(rank == 0 ? :not_a_mesh : nothing; comm))
    rank == 0 && partition_mesh(bfs_mesh(), nranks; dir=tmp)
    MPI.Barrier(comm)
    rank == 0 && rm(joinpath(tmp, "rank_$last_rank.xdm"))
    MPI.Barrier(comm)
    @test fails_everywhere(() -> distribute(tmp; comm))
end

mesh_path = joinpath(pkgdir(XCALibre), "examples/0_GRIDS/3d_box_1000x1000x1000mm_5.unv")
walls = (:y_min, :y_max, :z_min, :z_max)
function box_case(mesh; iterations, write_interval)
    model = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=1e-3), turbulence=RANS{Laminar}(),
        energy=Energy{Isothermal}(), domain=mesh)
    bcs = (
        U = [Dirichlet(:x_min, [0.5, 0.0, 0.0]), Extrapolated(:x_max), [Wall(w, [0.0, 0.0, 0.0]) for w ∈ walls]...],
        p = [Extrapolated(:x_min), Dirichlet(:x_max, 0.0), [Extrapolated(w) for w ∈ walls]...])
    s(solver) = SolverSetup(solver=solver, preconditioner=Jacobi(), convergence=1e-15, relax=0.5, rtol=1e-4, itmax=100)
    config = Configuration(solvers=(U=s(Bicgstab()), p=s(Cg())), schemes=(U=Schemes(), p=Schemes()),
        runtime=Runtime(iterations=iterations, time_step=1, write_interval=write_interval),
        hardware=Hardware(backend=CPU(), workgroup=64), boundaries=assign(region=mesh, bcs))
    initialise!(model.momentum.U, [0.0, 0.0, 0.0]); initialise!(model.momentum.p, 0.0)
    model, config
end

dm = distribute(() -> UNV3D_mesh(mesh_path, scale=0.001); comm)
tmp = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm; root=0)
cd(tmp)
run!(box_case(dm; iterations=2, write_interval=2)...; output=OpenFOAM(), progress=false)
MPI.Barrier(comm)
proc = "processor$last_rank"

@testset "partial restart (rank $rank)" begin
    rank == last_rank && rm(joinpath(proc, "2", "U"))
    MPI.Barrier(comm)
    @test fails_everywhere(() -> run!(box_case(dm; iterations=3, write_interval=-1)...; restart=2, progress=false))
end

@testset "broken processor directory (rank $rank)" begin
    poly = joinpath(proc, "constant", "polyMesh")
    addr = joinpath(poly, "cellProcAddressing")
    if rank == last_rank
        D = XCALibre.Distribute
        mv(addr, addr * ".orig")
        open(addr, "w") do io  # one cell short of the mesh it sits beside
            write(io, D._foam_header("labelList", "constant/polyMesh", "cellProcAddressing"))
            D._bin_list(io, Int32.(0:dm.partition.n_owned-2))
        end
    end
    MPI.Barrier(comm)
    @test fails_everywhere(() -> distribute(FOAMCase(tmp); comm))
    rank == last_rank && (mv(addr * ".orig", addr; force=true); rm(joinpath(poly, "faces")))
    MPI.Barrier(comm)
    @test fails_everywhere(() -> distribute(FOAMCase(tmp); comm))
end
MPI.Barrier(comm)
cd(pkgdir(XCALibre))
rank == 0 && rm(tmp; recursive=true)
