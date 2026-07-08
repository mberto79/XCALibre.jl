# Distributed (MPI) version of 2D_cylinder_U.jl. Needs an environment with XCALibre,
# PETSc and MPI. Run over 4 ranks with:
#   julia --project=<env> -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=<env> examples/2D_cylinder_U_mpi.jl`)'

# To control multithreading per rank, use julia default mechanism

#= source dev/local_stack.sh   # optional here, required for GPU examples
julia --project=dev/petscenv -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 --bind-to core --map-by socket:PE=2 $(Base.julia_cmd()) -t 2 --project=dev/petscenv examples/2D_cylinder_U_mpi.jl`)'

=#

# To test core pinning works
#=
julia --project=dev/petscenv -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 --bind-to core --map-by socket:PE=2 $(Base.julia_cmd()) --project=dev/petscenv -e "using MPI; MPI.Init(); r = MPI.Comm_rank(MPI.COMM_WORLD); println(r, \"  \", only(filter(l->startswith(l, \"Cpus_allowed_list\"), readlines(\"/proc/self/status\"))))"`)'

=#
using XCALibre, PETSc, MPI

comm = MPI.COMM_WORLD

# rank 0 reads the global mesh; distribute partitions and scatters it (read only on root)
mesh_dist = distribute(comm=comm) do
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    UNV2D_mesh(joinpath(grids_dir, "cylinder_d10mm_5mm.unv"), scale=0.001)
end

backend = CPU(); workgroup = 64
hardware = Hardware(backend=backend, workgroup=workgroup)

# Inlet conditions
velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = Physics(
    time = Transient(),
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
                Wall(:cylinder, noSlip),
                Extrapolated(:bottom),
                Extrapolated(:top)
        ],
        p = [
                Zerogradient(:inlet),
                Dirichlet(:outlet, 0.0),
                Wall(:cylinder),
                Extrapolated(:bottom),
                Extrapolated(:top)
        ]
    )
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 0.0,
        atol = 1e-6
    ),
    p = SolverSetup(
        solver      = Cg(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 0.0,
        atol = 1e-6,
        itmax = 2000
    )
)

timeScheme = CrankNicolson # or Euler
schemes = (
    U = Schemes(time=timeScheme, divergence=LUST, gradient=Gauss),
    p = Schemes(time=timeScheme, gradient=Gauss)
)

# distributed writer lands in phase 7 — run only for now
runtime = Runtime(iterations=1000, write_interval=-1, time_step=0.0025)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)

MPI.Comm_rank(comm) == 0 && println("done: final residuals Ux=", residuals.Ux[end],
    " Uy=", residuals.Uy[end], " p=", residuals.p[end])
