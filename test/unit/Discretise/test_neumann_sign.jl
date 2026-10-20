using XCALibre
using Test

mesh = UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "laplace_unit_3by3.unv"))
hardware = Hardware(backend=CPU(), workgroup=1024)
solvers = (T = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),)
schemes = (T = Schemes(laplacian=Linear),)
BCs = assign(
    region = mesh,
    (T = [
        Dirichlet(:left_wall, 50.0),
        Neumann(:right_wall, 2.0),
        Dirichlet(:bottom_wall, 10.0),
        Zerogradient(:upper_wall)
    ],)
)
config = Configuration(solvers=solvers, schemes=schemes,
    runtime=Runtime(iterations=1, write_interval=1, time_step=1), hardware=hardware, boundaries=BCs)
gamma = ConstantScalar(1.0)

# +Laplacian == 0 and -Laplacian == 0 must assemble to negated systems with the same solution
function assemble_laplace(sign)
    T = ScalarField(mesh)
    lap = Laplacian{Linear}(gamma, T)
    eqn = (sign > 0 ? lap : -lap) == Source(ConstantScalar(0.0))
    model = eqn → ScalarEquation(T, config.boundaries.T)
    discretise!(model, T, config)
    apply_boundary_conditions!(model, config.boundaries.T, nothing, 0.0, config)
    A = Matrix(model.equation.A.parent); b = copy(model.equation.b)
    A, b, A \ b
end

Ap, bp, xp = assemble_laplace(1)
Am, bm, xm = assemble_laplace(-1)
@test Ap ≈ -Am
@test bp ≈ -bm
@test xp ≈ xm
