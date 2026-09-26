using XCALibre
using Accessors
using KernelAbstractions
using LinearAlgebra
using SparseArrays
const Krylov = XCALibre.Solve.Krylov

# Serial BiCGStab solves use the shadow vector c = M⁻¹(b - Ax0), as PETSc bcgs does, instead of
# Krylov.jl's default c = b, which fits only a zero initial guess. Every serial solve is
# warm-started, and with c = b the iterates degraded as a steady run converged (motorBike 10M
# diverged). Check that a warm-started solve_system! takes exactly the iterates of Krylov.jl's
# bicgstab given that shadow, and that they differ from the default's.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "finer_mesh_laplace.unv"))

backend = CPU(); workgroup = 1024
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

rtol, atol, itmax = 1e-2, 1e-15, 3
solvers = (
    T = SolverSetup(
        solver      = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 1.0,
        rtol = rtol,
        atol = atol,
        itmax = itmax
    ),
)
schemes = (T = Schemes(laplacian = Linear),)
runtime = Runtime(iterations=1, write_interval=-1, time_step=1)
config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

source_field = ScalarField(mesh)
T_eqn = (
    Time{schemes.T.time}(model.solid.rhocp, model.energy.T)
    - Laplacian{schemes.T.laplacian}(model.solid.rDf, model.energy.T)
    ==
    - Source(source_field)
) → ScalarEquation(model.energy.T, config.boundaries.T)

T = model.energy.T
initialise!(T, 0.0)
discretise!(T_eqn, T, config)
apply_boundary_conditions!(T_eqn, config.boundaries.T, nothing, zero(Float64), config)
@reset T_eqn.preconditioner = set_preconditioner(solvers.T.preconditioner, T_eqn)
@reset T_eqn.solver = _workspace(solvers.T.solver, _b(T_eqn))
update_preconditioner!(T_eqn.preconditioner, mesh, config)

A = _A(T_eqn)
rowptr, colval, nzval = _rowptr(A), _colval(A), _nzval(A)
b = copy(_b(T_eqn))
n = length(b)
Acsc = sparse(transpose(SparseMatrixCSC(n, n, Int.(rowptr), Int.(colval), copy(nzval))))

# warm start near the solution, as in a converging outer loop: b - Ax0 is small next to b
x_conv = Acsc \ b
x0 = x_conv .+ 1e-3 .* sin.(5 .* (1:n) ./ n)
T.values .= x0
XCALibre.Solve.solve_system!(T_eqn, solvers.T, T, nothing, config)

M = Diagonal(copy(T_eqn.preconditioner.storage))
reference(c) = Krylov.bicgstab(Acsc, b, x0; c=c, M=M, rtol=rtol, atol=atol, itmax=itmax)[1]
x_shadow = reference(M*(b - Acsc*x0))
x_default = reference(b)

@test norm(T.values - x_shadow) <= 1e-10*norm(x_shadow - x0)
@test norm(x_default - x_shadow) > 1e-6*norm(x_shadow - x0)
