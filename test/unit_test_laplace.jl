using XCALibre
using KernelAbstractions
using Accessors

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")

grid = "finer_mesh_laplace.unv"
grid = "laplace_2d_mesh.unv"

mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

T_test  = ScalarField(mesh)
initialise!(T_test, 300.0)

k_test, cp_test = XCALibre.ModelPhysics.get_coefficients(:Steel,T_test)





using CUDA


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
            Dirichlet(:left_wall, 0),
            Zerogradient(:right_wall),
            Dirichlet(:bottom_wall, 1),
            Zerogradient(:upper_wall)
        ],
    )
)



solvers = (
    T = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
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

iterations=10

runtime = Runtime(
    iterations=iterations, write_interval=1, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)



source_field = ScalarField(mesh) #0.0 field
initialise!(model.energy.T, 15)

T_eqn = (
    Time{schemes.time}(model.energy.rhocp, model.energy.T)
    - Laplacian{schemes.laplacian}(model.energy.rDf, model.energy.T)
    ==
    - Source(source_field)
) → ScalarEquation(model.energy.T, BCs.T)




@info "Initialising preconditioners..."

@reset T_eqn.preconditioner = XCALibre.Solve.set_preconditioner(solvers.preconditioner, T_eqn)

@info "Pre-allocating solvers..."

@reset T_eqn.solver = XCALibre.Solve._workspace(solvers.solver, _b(T_eqn))


output=VTK()

outputWriter = initialise_writer(output, model.domain) # needs to be passed on
    
interpolate!(model.energy.Tf, model.energy.T, config)

n_cells = length(mesh.cells)
TF = _get_float(mesh)
prev = KernelAbstractions.zeros(backend, TF, n_cells) 
R_T = ones(TF, iterations)

# Initial calculations
time = zero(TF) # assuming time=0

XCALibre.Solvers.LAPLACE(
    model, T_eqn, config;
    output=output, pref=nothing, ncorrectors=0, inner_loops=0,
    outputWriter, R_T, time, isCoupled=false
)