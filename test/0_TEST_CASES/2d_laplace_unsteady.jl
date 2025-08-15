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
k_coeffs = MaterialCoefficients(
    c1=-1.4087, c2=1.3982, c3=0.2543, c4=-0.6260, c5=0.2334,
    c6=0.4256, c7=-0.4658, c8=0.1650, c9=-0.0199
)
cp_coeffs = MaterialCoefficients(
    c1=22.0061, c2=-127.5528, c3=303.6470, c4=-381.0098, c5=274.0328,
    c6=-112.9212, c7=24.7593, c8=-2.239153, c9=0.0
)

model = Physics(
    time = Transient(),
    solid = Solid{NonUniform}(k=k_coeffs, cp=cp_coeffs, rho=7850.0),
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