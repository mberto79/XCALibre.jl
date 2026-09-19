# Distributed (MPI) + GPU 3D_cascade_periodic.jl; needs CUDA-enabled MPI and PETSc builds (PETSc_jll has no CUDA).
# Install the launcher once: julia --project=<env> -e 'using MPI; MPI.install_mpiexecjl()'
# Run: mpiexecjl -n 2 julia --project=<env> examples/3D_cascade_mpi_GPU.jl; open XCALibre.foam in ParaView
using XCALibre, PETSc, MPI, CUDA

# every rank makes this identical call: rank 0 reads and partitions, the others receive
mesh_dist = distribute(periodic_patches=[(:top, :bottom)]) do
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    UNV3D_mesh(joinpath(grids_dir, "cascade_3D_periodic_2p5mm.unv"), scale=0.001)
end

backend = CUDABackend(); workgroup = 32
bind_device!(backend) # ranks pick/share the local GPU(s)
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

residuals = run!(model, config, output=OpenFOAM())

is_root() && println("done: final residuals Ux=", residuals.Ux[end],
    " Uy=", residuals.Uy[end], " p=", residuals.p[end])
