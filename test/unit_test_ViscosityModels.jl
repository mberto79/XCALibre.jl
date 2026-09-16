using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "quad40.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file)

# backend = CPU(); workgroup = AutoTune(); activate_multithread(backend)
backend = CPU(); workgroup = AutoTune()

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

cp = 1000.0
gamma = 1.4
Pr = 0.7

mu_ref = 1.8e-5
T_ref = 288.15
S = 110.4

# Setup dummy model
model = Physics(
    time = Steady(),
    fluid = Fluid{WeaklyCompressible}(
        nu = Viscosity{SutherlandViscosity}(mu_ref=mu_ref, T_ref=T_ref, S=S),
        cp = cp,
        gamma = gamma,
        Pr = Pr
        ),
    turbulence = RANS{Laminar}(),
    energy = Energy{SensibleEnthalpy}(Tref=288.15),
    domain = mesh_dev
    )

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Symmetry(:inlet, [0.0, 0.0, 0.0]),
            Symmetry(:outlet, [0.0, 0.0, 0.0]),
            Symmetry(:bottom, [0.0, 0.0, 0.0]),
            Symmetry(:top, [0.0, 0.0, 0.0])
    ],
        p = [
            Symmetry(:inlet),
            Symmetry(:outlet),
            Symmetry(:bottom),
            Symmetry(:top)
        ]
    )
)

schemes = (
    U = Schemes(),
    p = Schemes()
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(),
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-3
    ),
    p = SolverSetup(
        solver      = Gmres(),
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-3
    )
)

runtime = Runtime(iterations=10, time_step=0.1, write_interval=-1)
hardware = Hardware(backend=backend,workgroup = workgroup)

config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

initialise!(model.energy.T, 300.0)
initialise!(model.fluid.rho, 1.0)

update_viscosity!(model.fluid, model.energy, config)
print(model.fluid.nu.nu.values)  # Debugging output

expected_nu = mu_ref * (300.0 / T_ref)^(3/2) * (T_ref + S) / (300.0 + S) / 1.0
@test model.fluid.nu.nu.values[0] ≈ expected_nu

initialise!(model.energy.T, 150.0)
update_viscosity!(model.fluid, model.energy, config)
print(model.fluid.nu.nu.values)  # Debugging output

expected_nu = mu_ref * (150.0 / T_ref)^(3/2) * (T_ref + S) / (150.0 + S) / 1.0
@test model.fluid.nu.nu.values[0] ≈ expected_nu