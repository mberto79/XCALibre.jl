using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")

grid = "finer_mesh_laplace.unv"
# grid = "laplace_2d_mesh.unv"

mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)

mesh_dev = adapt(backend, mesh)


model = Physics(
    time = Transient(),
    solid = Solid{Uniform}(k=54.0, cp=480.0, rho=7850.0),
    energy = Energy{LaplaceEnergy}(),
    domain = mesh_dev
    )

BCs = assign(
    region = mesh_dev,
    (
        T = [     
            Dirichlet(:left_wall, 10),
            Zerogradient(:right_wall),
            Dirichlet(:bottom_wall, 20),
            Zerogradient(:upper_wall)
        ],
    )
)



solvers = (
    T = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = DILU(), # Jacobi(), #NormDiagonal(), # DILU()
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

initialise!(model.energy.T, 15)

residuals = run!(model, config)

# # Results can be seen in figures "grid_laplace.png" and "testing_laplace.png"

# @kwdef struct Coeffs{T}
#   c1::T
#   c2::T
#   c3::T
#   c4::T
#   c5::T
#   c6::T
#   c7::T
#   c8::T
#   c9::T
# end

# (c::Coeffs)(T) = c.c1.^8 + .... 

# SolidMaterials # this is an absytract time
# struct Aluminium end

# get_coeffs(material::Aluminium) = Coeffs(
#     c1 = as,
#     c2 = 4,
#     c3 = 4,
#     ...
# )
