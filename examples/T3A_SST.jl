using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
# grid = "flatplate_2D_laminar.unv"

grid = "T3A_SST/constant/polyMesh"
mesh_file = joinpath(grids_dir, grid)
mesh = FOAM3D_mesh(mesh_file, scale=1)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

velocity = [5.4, 0.0, 0.0]
nu = 1.5e-5
k_inlet = 0.047633
ω_inlet = 264.63/0.09
nut_inlet = k_inlet/ω_inlet

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmegaSST}(walls=(:plate,)),
    # turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region = mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Wall(:plate, [0.0, 0.0, 0.0]),
            Zerogradient(:top),
            Symmetry(:above),
            Empty(:defaultFaces),
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Wall(:plate),
            Zerogradient(:top),
            Symmetry(:above),
            Empty(:defaultFaces),
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Zerogradient(:outlet),
            Dirichlet(:plate, 0.0),
            Zerogradient(:top),
            Symmetry(:above),
            Empty(:defaultFaces),
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Zerogradient(:outlet),
            OmegaWallFunction(:plate),
            Zerogradient(:top),
            Symmetry(:above),
            Empty(:defaultFaces),
        ],
        nut = [
            Dirichlet(:inlet, k_inlet/ω_inlet),
            Extrapolated(:outlet),
            Dirichlet(:plate, 0.0), 
            Zerogradient(:top),
            Symmetry(:above),
            Empty(:defaultFaces),
        ]
    )
)

schemes = (
    U = Schemes(divergence=LUST),
    p = Schemes(divergence=LUST),
    h = Schemes(divergence=LUST),
    y = Schemes(gradient=Midpoint),
    k = Schemes(divergence=LUST),
    omega = Schemes(divergence=LUST)
)


solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-1
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-1
    ),
    h = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-2,
    ),
    y = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-8,
        rtol = 1e-2,
        relax       = 0.9,
    ),
    k = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.6,
        rtol = 1e-1
    ),
    omega = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.6,
        rtol = 1e-1
    )
)

runtime = Runtime(iterations=5000, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
# initialise!(model.energy.T, 300.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config, output=OpenFOAM()) # 9.39k allocs

using Plots
plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
plot!(1:length(Re), Re, yscale=:log10, label="h")