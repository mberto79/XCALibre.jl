using XCALibre
using KernelAbstractions
using Accessors
using Test

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")

# grid = "finer_mesh_laplace.unv"
grid = "laplace_unit_3by3.unv"
# grid = "laplace_unit_5by5.unv"

mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file)

T_test  = ScalarField(mesh)
initialise!(T_test, 300.0)

k_test, cp_test = XCALibre.ModelPhysics.get_coefficients(Aluminium(),T_test)


backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)

mesh_dev = adapt(backend, mesh)


model = Physics(
    time = Steady(),
    solid = Solid{Uniform}(k=1.0),
    energy = Energy{Conduction}(),
    domain = mesh_dev
    )

BCs = assign(
    region = mesh_dev,
    (
        T = [     
            Dirichlet(:left_wall, 50.0),
            Zerogradient(:right_wall),
            Dirichlet(:bottom_wall, 10.0),
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

iterations=10

runtime = Runtime(
    iterations=iterations, write_interval=1, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)



source_field = ScalarField(mesh) #0.0 field
initialise!(model.energy.T, 15)


 @info "Defining models..."
    T_eqn = (
        Time{schemes.time}(model.solid.rhocp, model.energy.T) #0.0 by default
        - Laplacian{schemes.laplacian}(model.solid.rDf, model.energy.T)
        ==
        - Source(source_field)
    ) → ScalarEquation(model.energy.T, config.boundaries.T)


@info "Initialising preconditioners..."

@reset T_eqn.preconditioner = set_preconditioner(solvers.preconditioner, T_eqn)

@info "Pre-allocating solvers..."

@reset T_eqn.solver = _workspace(solvers.solver, _b(T_eqn))


# The part that was previously inside the solver

output=VTK()
outputWriter = initialise_writer(output, model.domain) 
interpolate!(model.energy.Tf, model.energy.T, config)

@info "Allocating working memory..."

n_cells = length(mesh.cells)
TF = _get_float(mesh)
prev = KernelAbstractions.zeros(backend, TF, n_cells) 
R_T = ones(TF, iterations)

# Initial calculations
time = zero(TF) # assuming time=0

XCALibre.Solvers.LAPLACE(
    model, T_eqn, config;
    output=output, pref=nothing, ncorrectors=0, inner_loops=0,
    outputWriter, R_T, time
)

solution_A_matrix = T_eqn.equation.A.parent
solution_b_vector = T_eqn.equation.b

expected_A_matrix = [
     4.0  -1.0   0.0  -1.0   0.0   0.0   0.0   0.0   0.0;
    -1.0   5.0  -1.0   0.0  -1.0   0.0   0.0   0.0   0.0;
     0.0  -1.0   6.0   0.0   0.0  -1.0   0.0   0.0   0.0;
    -1.0   0.0   0.0   3.0  -1.0   0.0  -1.0   0.0   0.0;
     0.0  -1.0   0.0  -1.0   4.0  -1.0   0.0  -1.0   0.0;
     0.0   0.0  -1.0   0.0  -1.0   5.0   0.0   0.0  -1.0;
     0.0   0.0   0.0  -1.0   0.0   0.0   2.0  -1.0   0.0;
     0.0   0.0   0.0   0.0  -1.0   0.0  -1.0   3.0  -1.0;
     0.0   0.0   0.0   0.0   0.0  -1.0   0.0  -1.0   4.0
]

expected_b_vector = [100.0, 100.0, 120.0, 0.0, 0.0, 20.0, 0.0, 0.0, 20.0]

@test solution_A_matrix ≈ expected_A_matrix atol=0.1
@test solution_b_vector ≈ expected_b_vector