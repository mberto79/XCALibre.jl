using XCALibre
using Accessors
using KernelAbstractions
using LinearAlgebra

# rtol in serial Krylov solves is the reduction of the unpreconditioned residual,
# ||b - Ax|| <= rtol*||b - Ax0||, not of the preconditioned residual Krylov.jl monitors by default (CG's M-norm sqrt(r'Mr), or
# M(b - Ax) with left preconditioning). A Jacobi-preconditioned solve and its preconditioned
# residual are unchanged by a symmetric scaling A -> SAS, b -> Sb, but ||b - Ax|| is not:
# shrinking the rows of the few cells that hold most of the initial error makes the
# preconditioned test stop long before ||b - Ax|| has dropped by rtol. This is what happened on
# the 10M-cell motorBike mesh, where the pressure CG stopped after about 7 iterations instead
# of about 180.

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

function stopping_case(linear_solver, preconditioner)
    rtol = 1e-2
    solvers = (
        T = SolverSetup(
            solver      = linear_solver,
            preconditioner = preconditioner,
            convergence = 1e-8,
            relax       = 1.0,
            rtol = rtol,
            atol = 1e-15
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

    A = _A(T_eqn)
    rowptr, colval, nzval = _rowptr(A), _colval(A), _nzval(A)
    b = _b(T_eqn)
    n = length(b)

    # Symmetric scaling: three cells with most of the initial error get a tiny diagonal
    hot = [1, n ÷ 2, n]
    s = ones(n); s[hot] .= 1e-3
    for i in 1:n, k in rowptr[i]:(rowptr[i+1]-1)
        nzval[k] *= s[i]*s[colval[k]]
    end

    spmv(x) = [sum(nzval[k]*x[colval[k]] for k in rowptr[i]:(rowptr[i+1]-1)) for i in 1:n]
    x_exact = [sin(3i/n) + 2 for i in 1:n]
    b .= spmv(x_exact)
    x0 = copy(x_exact); x0[hot] .+= 1e4    # large error in the tiny-diagonal cells
    x0 .+= 1e-2 .* cos.(7 .* (1:n) ./ n)   # and a small smooth error everywhere else
    T.values .= x0

    r0 = norm(b - spmv(x0))
    update_preconditioner!(T_eqn.preconditioner, mesh, config)
    XCALibre.Solve.solve_system!(T_eqn, solvers.T, T, nothing, config)
    r1 = norm(b - spmv(T.values))

    return r0, r1, rtol, solvers.T.atol
end

# Jacobi exercises the elementwise route to ||b - Ax||, DILU the recomputed one
for (linear_solver, preconditioner) in
        ((Cg(), Jacobi()), (Bicgstab(), Jacobi()), (Gmres(), Jacobi()), (Cgs(), Jacobi()), (Bicgstab(), DILU()))
    r0, r1, rtol, atol = stopping_case(linear_solver, preconditioner)
    @test r1 <= rtol*r0 + atol
end
