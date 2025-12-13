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

postprocess = [ReynoldsStress(model.momentum.U), FieldRMS(model.momentum.U;name="U_rms")]
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs,postprocess=postprocess)

@test initialise!(model.momentum.U, velocity) === nothing
@test initialise!(model.momentum.p, 0.0) === nothing
residuals = run!(model, config);

u_rms_exact = U0*A/sqrt(2) 
v_rms_exact = U0*A/sqrt(2)
u_rms = sum(postprocess[2].rms.x.values[end-25:end-15])/length(postprocess[2].rms.x.values[end-25:end-15])
v_rms = sum(postprocess[2].rms.y.values[end-25:end-15])/length(postprocess[2].rms.y.values[end-25:end-15])

@test u_rms ≈ u_rms_exact atol = 0.005
@test v_rms ≈ v_rms_exact atol = 0.03

#check that the square root of the diagonal values of the RST agree with the rms
RST = postprocess[1].rs
u_rms_RST = sqrt(sum(RST.xx.values[end-25:end-15])/length(RST.xx.values[end-25:end-15]))
v_rms_RST = sqrt(sum(RST.yy.values[end-25:end-15])/length(RST.yy.values[end-25:end-15]))

@test u_rms_RST ≈ u_rms atol = 0.00001
@test v_rms_RST ≈ v_rms atol = 0.0003