# Distributed (MPI) version of TO_UPDATE_WONT_RUN/3D_BFS.jl. Needs an environment with
# XCALibre, PETSc and MPI. Run over 4 ranks with:
#   julia --project=<env> -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=<env> examples/3D_BFS_mpi.jl`)'

# To control multithreading per rank, use julia default mechanism

#= source dev/local_stack.sh   # optional here, required for GPU examples
julia --project=dev/petscenv -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 --bind-to core --map-by socket:PE=2 $(Base.julia_cmd()) -t 2 --project=dev/petscenv examples/3D_BFS_mpi.jl`)'

=#

# To test core pinning works
#=
julia --project=dev/petscenv -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 --bind-to core --map-by socket:PE=2 $(Base.julia_cmd()) --project=dev/petscenv -e "using MPI; MPI.Init(); r = MPI.Comm_rank(MPI.COMM_WORLD); println(r, \"  \", only(filter(l->startswith(l, \"Cpus_allowed_list\"), readlines(\"/proc/self/status\"))))"`)'

=#
using XCALibre, PETSc, MPI

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# rank 0 reads the global mesh; distribute partitions and scatters it
mesh = if rank == 0
    grids_dir = "/home/humberto/Desktop/BFS_GRIDS"
    UNV3D_mesh(joinpath(grids_dir, "bfs_unv_tet_5mm.unv"), scale=0.001)

    # grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    # UNV3D_mesh(joinpath(grids_dir, "bfs_unv_tet_10mm.unv"), scale=0.001)
else
    nothing
end
mesh_dist = distribute(mesh; comm=comm)

backend = CPU(); workgroup = AutoTune()
activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)

# Inlet conditions
velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dist
    )

BCs = assign(
    region=mesh_dist,
    (
        U = [
                Dirichlet(:inlet, velocity),
                Zerogradient(:outlet),
                Wall(:wall, noSlip),
                Zerogradient(:sides),
                Zerogradient(:top)
        ],
        p = [
                Zerogradient(:inlet),
                Dirichlet(:outlet, 0.0),
                Wall(:wall),
                Extrapolated(:sides),
                Extrapolated(:top)
        ]
    )
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 0.1
    ),
    p = SolverSetup(
        solver      = Cg(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 0.01,
        itmax = 1000
    )
)

gradScheme = Gauss
divScheme = Upwind
schemes = (
    U = Schemes(time=SteadyState, divergence=divScheme, gradient=gradScheme),
    p = Schemes(time=SteadyState, gradient=gradScheme)
)

runtime = Runtime(iterations=500, write_interval=500, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

MPI.Barrier(comm)
t = @elapsed residuals = run!(model, config, output=OpenFOAM())

rank == 0 && println("done in ", t, " s: final residuals Ux=", residuals.Ux[end],
    " Uy=", residuals.Uy[end], " Uz=", residuals.Uz[end], " p=", residuals.p[end])
