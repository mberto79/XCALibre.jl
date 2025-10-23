using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "quad40.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

backend = CPU()
workgroup = 1024
mesh_dev = adapt(backend, mesh)

U0 = 0.3
frequency = 1
velocity = [2*U0, U0, 0.0]
A = 0.1
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu=nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

@inline inflow(vec, t, i) = begin
    u = U0*(1 + A * cospi(2*frequency*t))
    v = U0*(1+ A * sinpi(2*frequency*t))
    return velocity = SVector{3}(u, v, 0.0)
end

BCs = assign(
    region=mesh_dev,
    (
        U = [
            DirichletFunction(:inlet, inflow),
            Extrapolated(:outlet),
            # Wall(:wall, [0.0, 0.0, 0.0]),
            Symmetry(:bottom, [0.0, 0.0, 0.0]),
            Symmetry(:top, [0.0, 0.0, 0.0])
    ],
        p = [
            Extrapolated(:inlet),
            Dirichlet(:outlet, 0.0),
            # Extrapolated(:wall),
            Symmetry(:bottom),
            Symmetry(:top)
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
        preconditioner = Jacobi(), # DILU(), TEMPORARY!
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-3
    ),
    p = SolverSetup(
        solver      = Gmres(), #Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), #LDL(), TEMPORARY!
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-3
    )
)
iterations = 100

runtime = Runtime(iterations=iterations, time_step=0.05, write_interval=-1)
hardware = Hardware(backend=backend,workgroup = workgroup)

postprocess = FieldAverage(model.momentum.U;name="Umean",start = 51*0.05, update_interval = 2*0.05 )
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs,postprocess=postprocess)

@test initialise!(model.momentum.U, velocity) === nothing
@test initialise!(model.momentum.p, 0.0) === nothing
foreach(rm, (f for f in readdir() if endswith(f, ".vtk")))
residuals = run!(model, config);

u_mean_exact = U0
v_mean_exact = U0
u_mean = sum(postprocess.mean.x.values[end-30:end-10])/length(postprocess.mean.x.values[end-30:end-10])
v_mean = sum(postprocess.mean.y.values[end-30:end-10])/length(postprocess.mean.y.values[end-30:end-10])

@test u_mean ≈ u_mean_exact atol = 0.005
@test v_mean ≈ v_mean_exact atol = 0.05