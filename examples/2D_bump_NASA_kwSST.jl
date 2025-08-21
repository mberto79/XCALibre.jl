using Plots
using XCALibre
# using CUDA

# mesh_file = "unv_sample_meshes/backwardFacingStep_5mm.unv"
mesh_file = "./examples/0_GRIDS/OF_bump2d/polymesh/"
mesh = FOAM3D_mesh(mesh_file, scale=1, integer_type=Int64, float_type=Float64)

# mesh_dev = adapt(CUDABackend(), mesh)
# mesh_dev = adapt(CPUBackend(), mesh)
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(CPU(), mesh)

L = 50
nu = 1e-3
# u_mag = 1.5 # 5mm mesh
u_mag = 69.44 # 2mm mesh
velocity = [u_mag, 0.0, 0.0]
Tu = 0.01
nuR = 10
ReL = u_mag*L/nu
k_inlet = 3/2*(Tu*u_mag)^2
ω_inlet = k_inlet/(nuR*nu)

νt_inlet = k_inlet/ω_inlet
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmegaSST}(walls=(:bump,)),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region = mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Symmetry(:top),
            Symmetry(:symUp),
            Wall(:bump, [0.0, 0.0, 0.0]),
            Symmetry(:symDown),
            Empty(:frontAndBack)
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Symmetry(:top),
            Symmetry(:symUp),
            Wall(:bump),
            Symmetry(:symDown),
            Empty(:frontAndBack)
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Zerogradient(:outlet),
            Symmetry(:top),
            Symmetry(:symUp),
            Dirichlet(:bump, 0.0),
            Symmetry(:symDown),
            Empty(:frontAndBack)
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Zerogradient(:outlet),
            Symmetry(:top),
            Symmetry(:symUp),
            OmegaWallFunction(:bump),
            Symmetry(:symDown),
            Empty(:frontAndBack)
        ],
        nut = [
            Dirichlet(:inlet, k_inlet/ω_inlet),
            Extrapolated(:outlet),
            Symmetry(:top),
            Symmetry(:symUp),
            Dirichlet(:bump, 0.0),
            Symmetry(:symDown),
            Empty(:frontAndBack)
        ],
        y = [
            Zerogradient(:inlet, k_inlet/ω_inlet),
            Zerogradient(:outlet),
            Zerogradient(:top),
            Zerogradient(:symUp),
            Dirichlet(:bump, 0.0),
            Zerogradient(:symDown),
            Zerogradient(:frontAndBack)
        ]
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes(divergence=Upwind),
    y = Schemes(),#gradient=Midpoint),
    k = Schemes(divergence=Upwind),
    omega = Schemes(divergence=Upwind)
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.5,
        rtol = 1e-1
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-1
    ),
    y = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-10,
        rtol = 0.1,
        relax       = 0.9,
    ),
    k = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-1
    ),
    omega = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-1
    )
)

runtime = Runtime(iterations=2000, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)


GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config, output=OpenFOAM()) # 36.90k allocs

# Reff = stress_tensor(model.momentum.U, nu, model.turbulence.nut)
# Fp = pressure_force(:wall, model.momentum.p, 1.25)
# Fv = viscous_force(:wall, model.momentum.U, 1.25, nu, model.turbulence.nut)


# plot(; xlims=(0,494))
# plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
# plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
# plot!(1:length(Rp), Rp, yscale=:log10, label="p")

# # PROFILING CODE

# using Profile, PProf

# GC.gc()

# initialise!(model.momentum.U, velocity)
# initialise!(model.momentum.p, 0.0)
# initialise!(model.turbulence.k, k_inlet)
# initialise!(model.turbulence.omega, ω_inlet)
# initialise!(model.turbulence.nut, νt_inlet)

# residuals = run!(model, config)

# Profile.Allocs.clear()
# Profile.Allocs.@profile sample_rate=0.1 begin 
# residuals = run!(model, config)
# end

# PProf.Allocs.pprof()