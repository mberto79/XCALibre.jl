using Plots
using XCALibre
# using CUDA

# mesh_file = "unv_sample_meshes/backwardFacingStep_5mm.unv"
mesh_file = "./examples/0_GRIDS/OF_bump2d/polymesh/"
mesh = FOAM3D_mesh(mesh_file, scale=1, integer_type=Int64, float_type=Float64)

# mesh_dev = adapt(CUDABackend(), mesh)
# mesh_dev = adapt(CPUBackend(), mesh)
mesh_dev = adapt(CPU(), mesh)

L = 50
nu = 2.31e-5
# u_mag = 1.5 # 5mm mesh
u_mag = 69.44 # 2mm mesh
velocity = [u_mag, 0.0, 0.0]
Tu = 0.01
nuR = 100
ReL = u_mag*L/nu
k_inlet = 0.001*u_mag^2/ReL#3/2*(Tu*u_mag)^2
ω_inlet = 2*u_mag/L #k_inlet/(nuR*nu)

νt_inlet = k_inlet/ω_inlet
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmegaSST}(walls=(:wall,)),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )


@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Symmetry(:top),
    Symmetry(:symUp),
    Wall(:bump, [0.0, 0.0, 0.0]),
    Symmetry(:symDown)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Symmetry(:top, 0.0),
    Symmetry(:symUp, 0.0),
    Wall(:bump, 0.0),
    Symmetry(:symDown, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:symUp, 0.0),
    KWallFunction(:bump),
    Neumann(:symDown, 0.0)
)

@assign! model turbulence omega (
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:symUp, 0.0),
    OmegaWallFunction(:bump),
    Neumann(:symDown, 0.0)
)

@assign! model turbulence nut (
    Dirichlet(:inlet, νt_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:symUp, 0.0),
    NutWallFunction(:bump),
    Neumann(:symDown, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind),
    p = set_schemes(divergence=Upwind),
    y = set_schemes(gradient=Midpoint),
    k = set_schemes(divergence=Upwind),
    omega = set_schemes(divergence=Upwind)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.5,
        rtol = 1e-2,
        atol = 1e-10
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.2,
        rtol = 1e-3,
        atol = 1e-10
    ),
    y = set_solver(
        model.turbulence.y;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.9,
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-3,
        atol = 1e-10
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-3,
        atol = 1e-10
    )
)

runtime = set_runtime(iterations=3000, write_interval=100, time_step=1)
# runtime = set_runtime(iterations=2, write_interval=-1, time_step=1)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)


GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, νt_inlet)

residuals = run!(model, config) # 36.90k allocs

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