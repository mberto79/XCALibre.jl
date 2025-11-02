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
A = 0.5
frequency = 1
velocity = [1.5*U0, U0, 0.0]
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
    u = U0*(1 + A*cospi(2*frequency*t))
    v = U0*(1+ A*sinpi(2*frequency*t))
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
timestep = 0.01
runtime = Runtime(iterations=iterations, time_step=timestep, write_interval=-1)
hardware = Hardware(backend=backend,workgroup = workgroup)

postprocess = [FieldAverage(model.momentum.U; name="Umean"),FieldAverage(model.momentum.U; name="Umean_stop50", stop=50*timestep, update_interval = 3*timestep),FieldAverage(model.momentum.U;name="Umean_start51", start= 51*timestep, update_interval = timestep/2)]
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs,postprocess=postprocess)

@test initialise!(model.momentum.U, velocity) === nothing
@test initialise!(model.momentum.p, 0.0) === nothing
residuals = run!(model, config);


#check middle 10 cells of inlet agree with analytical mean
u_mean_exact = U0
v_mean_exact = U0
u_mean = sum(postprocess[1].mean.x.values[end-25:end-15])/length(postprocess[1].mean.x.values[end-25:end-15])
v_mean = sum(postprocess[1].mean.y.values[end-25:end-15])/length(postprocess[1].mean.y.values[end-25:end-15])

@test u_mean ≈ u_mean_exact atol = 0.005
@test v_mean ≈ v_mean_exact atol = 0.005

#testing start and end and update_interval logic
u_mean_first_half = sum(postprocess[2].mean.x.values[end-25:end-15])/length(postprocess[2].mean.x.values[end-25:end-15])
u_mean_second_half = sum(postprocess[3].mean.x.values[end-25:end-15])/length(postprocess[3].mean.x.values[end-25:end-15])

@test u_mean ≈ u_mean_first_half atol = 0.005
@test u_mean ≈ u_mean_second_half atol = 0.005