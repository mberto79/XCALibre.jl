# Purpose of this test is to confirm solver is running in mixture mode without errors.

using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "quad40.unv")
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = AutoTune()
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

noSlipVelocity = [0.0, 0.0, 0.0]
gravity = Gravity([0.0, -9.81, 0.0])

model = Physics(
    time = Transient(),
    fluid = Fluid{Multiphase}(
        model = Mixture(diameter=2.0e-4),
        phases = (
            Phase(rho=1000.0, mu=1.0e-3),
            Phase(rho=2500.0, mu=1.8e-5),
        ),
        gravity = gravity
    ),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
)

BCs = assign(
    region = mesh_dev,
    (
        U = [
            Wall(:inlet, noSlipVelocity),
            Wall(:outlet, noSlipVelocity),
            Zerogradient(:top),
            Wall(:bottom, noSlipVelocity),
        ],
        p_rgh = [
            Zerogradient(:inlet),
            Zerogradient(:outlet),
            Zerogradient(:bottom),
            Dirichlet(:top, 0.0),
        ],
        alpha = [
            Zerogradient(:inlet),
            Zerogradient(:outlet),
            Zerogradient(:bottom),
            Zerogradient(:top),
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
        solver=Bicgstab(), preconditioner=Jacobi(),
        convergence=1e-7, relax=1.0, rtol=0.0, atol=1.0e-5),
    p_rgh = SolverSetup(
        solver=Cg(), preconditioner=Jacobi(),
        convergence=1e-7, relax=1.0, rtol=0.0, atol=1.0e-5),
    alpha = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(),
        convergence=1e-7, relax=1.0, rtol=0.0, atol=1.0e-5),
)

runtime = Runtime(iterations=5, time_step=1.0e-4, write_interval=-1)
config  = Configuration(solvers=solvers, schemes=schemes,
                        runtime=runtime, hardware=hardware, boundaries=BCs)

initialise!(model.momentum.p, 0.0)
initialise!(model.momentum.U, noSlipVelocity)
initialise!(model.fluid.alpha, 0.0)
setField_Box!(mesh=mesh, field=model.fluid.alpha, value=1.0,
                min_corner=[0.0, 0.0, -0.5], max_corner=[0.3, 0.4, 0.5])

residuals = run!(model, config)

@test all(isfinite, residuals.Ux)
@test all(isfinite, residuals.Uy)
@test all(isfinite, residuals.p)