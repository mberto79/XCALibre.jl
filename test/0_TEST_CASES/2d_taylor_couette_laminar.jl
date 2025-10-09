using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "taylor_couette_200_10.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = AutoTune(); activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

rpm = 100
centre = [0,0,0]
axis = [0,0,1]
nu = 1e-3

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region = mesh_dev,
    (
        U = [
            RotatingWall(:inner_wall, rpm=rpm, centre=centre, axis=axis),
            Wall(:outer_wall, [0,0,0]),
        ],
        p = [
            Zerogradient(:inner_wall),
            Wall(:outer_wall),
        ]
    )
)

@test typeof(BCs.U[1]) <: RotatingWall
@test typeof(BCs.U[1].value.centre) <: SVector
@test typeof(BCs.U[1].value.axis) <: SVector
@test typeof(BCs.U[1].value.rpm) <: AbstractFloat

schemes = (
    U = Schemes(divergence = Upwind),
    p = Schemes()
)


solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), # ILU0GPU, Jacobi, DILU
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-2
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres(), Cg()
        preconditioner = Jacobi(), # IC0GPU, Jacobi, DILU
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-3
    )
)

runtime = Runtime(
    iterations=100, time_step=1, write_interval=-1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, [0,0,0])
initialise!(model.momentum.p, 0.0)

@time residuals = run!(model, config)

wall_avg = boundary_average(:inner_wall, model.momentum.U, BCs.U, config)

# Test that the average of individual velocity components is zero
@test wall_avg[1] ≈ zero(eltype(wall_avg)) atol=1e-10
@test wall_avg[2] ≈ zero(eltype(wall_avg)) atol=1e-10
@test wall_avg[3] ≈ zero(eltype(wall_avg)) atol=1e-10


# Check that the mean of the magnitude is similar to the analytical expression
wall_mean_mag = 0.0
wall_faces_range = BCs.U[1].IDs_range
for fID ∈ wall_faces_range
    global wall_mean_mag += norm(model.momentum.Uf[fID])
end
wall_mean_mag /= length(wall_faces_range)

@test rpm*2π/60*0.1 ≈ wall_mean_mag atol = 1e-3

