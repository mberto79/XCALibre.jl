# MeshWave wall distance on a distributed mesh vs the undivided mesh, per rank under mpiexec:
# the owned values must be identical (the sweep reads only face neighbours and the seed uses the
# wall faces of every rank), whatever the partitioning. 2D step (convex corner) and 3D tet step.
using XCALibre, PETSc, MPI, Test

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

grids = pkgdir(XCALibre, "examples/0_GRIDS")

function wall_distance_case(mesh, walls)
    model = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=1e-3),
        turbulence=RANS{KOmegaSST}(walls=walls), energy=Energy{Isothermal}(), domain=mesh)
    config = Configuration(schemes=(;), solvers=(;),
        runtime=Runtime(iterations=1, time_step=1, write_interval=-1),
        hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=(;))
    model, config
end

for (label, load, walls) ∈ (
        ("2D step", () -> UNV2D_mesh(joinpath(grids, "backwardFacingStep_10mm.unv"), scale=0.001), (:wall,)),
        ("3D tet step", () -> UNV3D_mesh(joinpath(grids, "bfs_unv_tet_10mm.unv"), scale=0.001), (:wall,)))
    gmesh = rank == 0 ? load() : nothing
    ref = if rank == 0
        model, config = wall_distance_case(gmesh, walls)
        wall_distance!(model, walls, config)
        collect(model.turbulence.y.values)
    else
        nothing
    end
    yserial = MPI.bcast(ref, comm; root=0)

    dm = distribute(gmesh; comm=comm)
    model, config = wall_distance_case(dm, walls)
    new_config = wall_distance!(model, walls, config)
    n = dm.partition.n_owned
    y = Array(model.turbulence.y.values)
    @testset "MeshWave distributed $label (rank $rank)" begin
        @test y[1:n] == yserial[dm.orig_cells[1:n]]
        @test check_ghosts(model.turbulence.y, dm, new_config) == 0
    end
end
MPI.Barrier(comm)
