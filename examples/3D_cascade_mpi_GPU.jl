# Distributed (MPI) + GPU version of 3D_cascade_periodic.jl, exercising the decomposed
# OpenFOAM writer (processor<rank>/ dirs; open XCALibre.foam in ParaView).
# Periodic :top/:bottom: distribute(periodic_patches=...) colocates matched cell pairs
# per rank, then construct_periodic on the DistributedMesh works exactly as in serial.
# Needs an env with XCALibre, PETSc, MPI, CUDA and a CUDA-enabled MPI/PETSc stack
# (locally: `source dev/local_stack.sh`; without a CUDA PETSc pass solve_on=CPU()).
# Run over 2 ranks with:
#   julia --project=<env> -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project=<env> examples/3D_cascade_mpi_GPU.jl`)'
using XCALibre, PETSc, MPI, CUDA

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# rank 0 reads the global mesh; distribute partitions and scatters it
mesh = if rank == 0
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    UNV3D_mesh(joinpath(grids_dir, "cascade_3D_periodic_2p5mm.unv"), scale=0.001)
else
    nothing
end
mesh_dist = distribute(mesh; comm=comm, periodic_patches=[(:top, :bottom)])

backend = CUDABackend(); workgroup = 32
bind_device!(backend, rank) # ranks pick/share the local GPU(s)
hardware = Hardware(backend=backend, workgroup=workgroup)
periodic = construct_periodic(mesh_dist, backend, :top, :bottom)
mesh_dev = adapt(backend, mesh_dist)

velocity = [0.25, 0.0, 0.0]
nu = 1e-3
noSlip = [0.0, 0.0, 0.0]

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu=nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region = mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Wall(:plate, noSlip),
            Symmetry(:side1), Symmetry(:side2),
            periodic...
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Wall(:plate),
            Symmetry(:side1), Symmetry(:side2),
            periodic...
        ]
    )
)

schemes = (
    U = Schemes(divergence=Linear, gradient=Gauss),
    p = Schemes(gradient=Gauss)
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-3
    ),
    p = SolverSetup(
        solver      = Cg(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-3
    )
)

runtime = Runtime(iterations=500, time_step=1, write_interval=100)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

# native GPU solve needs a CUDA PETSc; on a host-only PETSc add solve_on=CPU()
residuals = run!(model, config, output=OpenFOAM())

rank == 0 && println("done: final residuals Ux=", residuals.Ux[end],
    " Uy=", residuals.Uy[end], " p=", residuals.p[end])
