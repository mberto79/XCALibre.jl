using XCALibre
#this test is solely to check the capability to post-process multiple fields at a time for a steady case 

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "quad40.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

backend = CPU()
workgroup = 1024
mesh_dev = adapt(backend, mesh)

U0 = 0.3
velocity = [U0, 0.0, 0.0]
nu = 1e-3

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu=nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev)

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Extrapolated(:outlet),
            Symmetry(:bottom, [0.0, 0.0, 0.0]),
            Symmetry(:top, [0.0, 0.0, 0.0])
    ],
        p = [
            Extrapolated(:inlet),
            Dirichlet(:outlet, 0.0),
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

postprocess = [FieldAverage(model.momentum.U; name = "U_mean_freq_5", update_interval = 5), 
FieldAverage(model.momentum.U; name = "U_mean_start_5",start =5, update_interval = 5), 
FieldAverage(model.momentum.U; name = "U_mean_stop_5",stop =5, update_interval = 7) ,FieldAverage(model.momentum.p;name = "p_mean"),
ReynoldsStress(model.momentum.U), FieldRMS(model.momentum.U; name = "U_rms", start = 2, stop=6, update_interval = 2) ]

config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs, postprocess=postprocess)

@test initialise!(model.momentum.U, velocity) === nothing
@test initialise!(model.momentum.p, 0.0) === nothing


residuals = run!(model, config);


@test postprocess[1].field isa VectorField
@test postprocess[2].field isa VectorField
@test postprocess[3].field isa VectorField
@test postprocess[4].field isa ScalarField
@test postprocess[5].field isa VectorField
@test postprocess[6].field isa VectorField


@test postprocess[1].mean isa VectorField
@test postprocess[2].mean isa VectorField
@test postprocess[3].mean isa VectorField
@test postprocess[4].mean isa ScalarField
@test postprocess[5].rs isa SymmetricTensorField
@test postprocess[6].rms isa VectorField