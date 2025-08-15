using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")

# grid = "finer_mesh_laplace.unv"
grid = "laplace_2d_mesh.unv"
# grid = "laplace_unit_3by3.unv"

mesh_file = joinpath(grids_dir, grid)

# mesh = UNV2D_mesh(mesh_file, scale=0.001)
mesh = UNV2D_mesh(mesh_file)
@test typeof(mesh) <: Mesh2

backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)

mesh_dev = adapt(backend, mesh)



# Steel coefficients
k_coeffs = [-1.4087, 1.3982, 0.2543, -0.6260, 0.2334, 0.4256, -0.4658, 0.1650, -0.0199]
cp_coeffs = [22.0061, -127.5528, 303.6470, -381.0098, 274.0328, -112.9212, 24.7593, -2.239153, 0.0]

model = Physics(
    time = Transient(),
    solid = Solid{NonUniform}(k_coeffs=k_coeffs, cp_coeffs=cp_coeffs, rho=7850.0),
    energy = Energy{Conduction}(),
    domain = mesh_dev
    )


left_wall_temp = 100.0       # 0.0
bottom_wall_temp = 200.0     # 1.0

BCs = assign(
    region = mesh_dev,
    (
        T = [     
            Dirichlet(:left_wall, left_wall_temp),
            Zerogradient(:right_wall),
            Dirichlet(:bottom_wall, bottom_wall_temp),
            Zerogradient(:upper_wall)
        ],
    )
)



solvers = (
    T = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), # Jacobi(), #NormDiagonal(), # DILU()
        convergence = 1e-8,
        relax       = 0.8,
        rtol = 1e-4,
        atol = 1e-5
    )
)

schemes = (
    T = Schemes(laplacian = Linear)
)


runtime = Runtime(
    iterations=10, write_interval=10, time_step=0.1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

@test initialise!(model.energy.T, 100.0) === nothing

residuals = run!(model, config)

wall_L = boundary_average(:left_wall, model.energy.T, BCs.T, config)
wall_R = boundary_average(:right_wall, model.energy.T, BCs.T, config)
wall_B = boundary_average(:bottom_wall, model.energy.T, BCs.T, config)
wall_U = boundary_average(:upper_wall, model.energy.T, BCs.T, config)

diagonal_temp = (left_wall_temp + bottom_wall_temp) / 2.0 # const temperature at diagonal cells
cold_average = (left_wall_temp + diagonal_temp) / 2.0 # hottest cell is at the diagonal edge, coldest cell is the cold wall edge
warm_average = (bottom_wall_temp + diagonal_temp) / 2.0 # coldest cell is at the diagonal edge, hottest cell is the hot wall edge


tolerance = diagonal_temp / 10.0

@test wall_R ≈ warm_average atol=tolerance
@test wall_U ≈ cold_average atol=tolerance