# Distributed (MPI) backward-facing step; needs XCALibre, PETSc and MPI (PETSc_jll/MPI.jl binaries suffice).
# Install the launcher once: julia --project=<env> -e 'using MPI; MPI.install_mpiexecjl()'
# Run: mpiexecjl -n 4 julia --project=<env> examples/3D_BFS_mpi.jl [mesh.unv]
using XCALibre, PETSc, MPI

mesh_file = isempty(ARGS) ?
    joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "bfs_unv_tet_10mm.unv") : ARGS[1]

# every rank makes this identical call: rank 0 decomposes into `dir` the first time, the rest
# wait, and each then loads only its own part. A decomposition there is reused for the same key.
mesh_dist = distribute(dir=joinpath(pwd(), "parts"), key=(mesh_file, 0.001)) do
    UNV3D_mesh(mesh_file, scale=0.001)
end

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
        preconditioner = Jacobi(), # GAMG() needs no extra build; BoomerAMG() needs hypre
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

runtime = Runtime(iterations=100, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

t = @elapsed residuals = run!(model, config, output=OpenFOAM())

is_root() && println("done in ", t, " s: final residuals Ux=", residuals.Ux[end],
    " Uy=", residuals.Uy[end], " Uz=", residuals.Uz[end], " p=", residuals.p[end])
