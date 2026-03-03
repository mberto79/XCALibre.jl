# The idea is to run four cases: without adaptive time-stepping, and with it when maxCo=0.25,0.5,0.75
# The number of iterations for each individual case was selected so that the final simulation time is approximately the same (10 seconds):

    # 2000 iterations: non-adaptive
    # 2867 iterations: maxCo=0.25
    # 1434 iterations: maxCo=0.5
    # 957 iterations: maxCo=0.75

# Then we compare if the average velocity magnitude at the outlet is identical across all these cases despite different dt


using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_10mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

function inlet_velocity(coords, time, index)
    ux = 0.5 + 0.1 * sin(2π*0.1*time)
    return SVector(ux, 0.0, 0.0)
end

BCs = assign(
    region = mesh_dev,
    (
        U = [
            DirichletFunction(:inlet, inlet_velocity),
            Zerogradient(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Wall(:top, [0.0, 0.0, 0.0])
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Wall(:wall),
            Wall(:top)
        ]
    )
)

schemes = (
    U = Schemes(time=Euler),
    p = Schemes()
)


solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
    )
)

runtime = Runtime(
    iterations=2000, time_step=0.005, write_interval=10)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config) # 9.39k allocs

outlet_result_1 = boundary_average(:outlet, model.momentum.U, BCs.U, config)



adaptive = AdaptiveTimeStepping(
        maxCo=0.25,
        minShrink=0.1,
        maxGrow=1.2
    )
    
runtime = Runtime(
    iterations=2867, time_step=0.005, write_interval=10, adaptive=adaptive)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config) # 9.39k allocs

outlet_result_2 = boundary_average(:outlet, model.momentum.U, BCs.U, config)



adaptive = AdaptiveTimeStepping(
        maxCo=0.5,
        minShrink=0.1,
        maxGrow=1.2
    )
    
runtime = Runtime(
    iterations=1434, time_step=0.005, write_interval=10, adaptive=adaptive)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config) # 9.39k allocs

outlet_result_3 = boundary_average(:outlet, model.momentum.U, BCs.U, config)



adaptive = AdaptiveTimeStepping(
        maxCo=0.75,
        minShrink=0.1,
        maxGrow=1.2
    )
    
runtime = Runtime(
    iterations=957, time_step=0.005, write_interval=10, adaptive=adaptive)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config) # 9.39k allocs

outlet_result_4 = boundary_average(:outlet, model.momentum.U, BCs.U, config)

n1 = norm(outlet_result_1)
n2 = norm(outlet_result_2)
n3 = norm(outlet_result_3)
n4 = norm(outlet_result_4)

@test isapprox(n2, n1; rtol=0.01)
@test isapprox(n3, n1; rtol=0.01)
@test isapprox(n4, n1; rtol=0.01)