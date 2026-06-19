# In this case, a hydrostatic water-air column is initialised.
#       After 10,000 iterations velocity currents must not develop if discretisation is consistent.
#       *Note: test is currently reduced to 1,000 iterations to save time.


using XCALibre
using Test
using LinearAlgebra

scaling = 0.001 

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "quad100.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=scaling)

backend = CPU(); workgroup = AutoTune(); activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

noSlipVelocity = [0.0, 0.0, 0.0]

gravity = Gravity([0.0, -9.81, 0.0])


model = Physics(
    time = Transient(),
    fluid = Fluid{Multiphase}(
        model = VOF(cAlpha=0.0, sigma=0.0),
        phases = (
            Phase(rho=1000.0, mu=1.0e-3),
            Phase(rho=1.2,    mu=1.8e-5),
        ),
        gravity = gravity
    ),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

operating_pressure = 0.0


BCs = assign(
    region = mesh_dev,
    (
        U = [
            Wall(:inlet, noSlipVelocity),
            Wall(:outlet, noSlipVelocity),
            Extrapolated(:top), 
            Wall(:bottom, noSlipVelocity),
        ],
        p_rgh = [
            Zerogradient(:inlet),
            Zerogradient(:outlet),
            Zerogradient(:bottom),
            # Zerogradient(:top),
            Dirichlet(:top, 0.0),
        ],
        alpha = [
            Zerogradient(:inlet),
            Zerogradient(:outlet),
            Zerogradient(:bottom),
            Extrapolated(:top),
        ]
    )
)

schemes = (
    U =     Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
    p =     Schemes(time=Euler, gradient=Gauss,    laplacian=Linear),
    p_rgh = Schemes(time=Euler, gradient=Gauss,    laplacian=Linear),
    alpha = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), # ILU0GPU, Jacobi, DILU
        convergence = 1e-7,
        relax       = 1.0,
        rtol        = 0.0,
        atol        = 1.0e-5
    ),
    p_rgh = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres(), Cg()
        preconditioner = Jacobi(), # IC0GPU, Jacobi, DILU
        convergence = 1e-7,
        relax       = 1.0,
        rtol        = 0.0,
        atol        = 1.0e-7
        
    ),
    alpha = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres(), Cg()
        preconditioner = Jacobi(), # IC0GPU, Jaco•bi, DILU
        convergence = 1e-7,
        relax       = 1.0,
        rtol        = 0.0,
        atol        = 1.0e-5
    )
)

hardware = Hardware(backend=backend, workgroup=workgroup)

min_corner_vec = [0.0, 0.0, -0.5]
max_corner_vec = [1.0, 0.5,  0.5]

runtime = Runtime(iterations=1000, time_step=1.0e-5, write_interval=20000)
config  = Configuration(solvers=solvers, schemes=schemes,
                        runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

min_corner_vec = [0.0, 0.0, -0.5]
max_corner_vec = [1.0, 0.5,  0.5]

initialise!(model.fluid.p_rgh, 0.0)
initialise!(model.momentum.U, noSlipVelocity)
initialise!(model.fluid.alpha, 0.0)
setField_Box!(mesh=mesh, field=model.fluid.alpha, value=1.0,
              min_corner=min_corner_vec, max_corner=max_corner_vec)


residuals = run!(model, config)

U = model.momentum.U
vel_mag = sqrt.(U.x.values.^2 .+ U.y.values.^2 .+ U.z.values.^2)
mean_vel_mag = sum(vel_mag) / length(vel_mag)
@test mean_vel_mag < 1e-8