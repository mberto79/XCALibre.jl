using Plots
using FVM_1D
using Krylov
using KernelAbstractions
using CUDA


# bfs_unv_tet_15mm, 10mm, 5mm, 4mm, 3mm
mesh_file = "unv_sample_meshes/bfs_unv_tet_5mm.unv"
@time mesh = build_mesh3D(mesh_file, scale=0.001)

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U (
    # Dirichlet(:inlet, velocity),
    # Neumann(:outlet, 0.0),
    # Dirichlet(:wall, [0.0, 0.0, 0.0]),
    # Dirichlet(:top, [0.0, 0.0, 0.0]),
    # Dirichlet(:sides, [0.0, 0.0, 0.0])
    Dirichlet(:inlet, velocity),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:sides, 0.0)
)

 @assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0),
    Neumann(:sides, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind),
    p = set_schemes()
)


solvers = (
    U = set_solver(
        model.U;
        solver      = CgSolver, #BicgstabSolver, # BicgstabSolver, GmresSolver, #CgSolver
        # preconditioner = CUDA_ILU2(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-4,
        atol = 1e-2
    ),
    p = set_solver(
        model.p;
        solver      = CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        # preconditioner = CUDA_IC0(),
        preconditioner = Jacobi(),

        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-4,
        atol = 1e-2


    )
)

runtime = set_runtime(
    iterations=500, time_step=1, write_interval=500)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)

backend = CPU()
backend = CUDABackend()

Rx, Ry, Rz, Rp, model1, peqn = simple!(model, config, backend)

using CUDA.CUSPARSE
using LinearOperators
using SparseArrays
using LinearAlgebra

P = ic02(peqn.equation.A)
P = ilu02(peqn.equation.A)
test = similar(P)
test .= ic02(peqn.equation.A)

CUDA.@allowscalar test1 = UpperTriangular(P)'

n = length(peqn.equation.b)
type = eltype(peqn.equation.b)
z = CUDA.zeros(type, n)

function ldiv_ic0!(P::CuSparseMatrixCSC, x, y, z)
    ldiv!(z, UpperTriangular(P)', x)  # Forward substitution with L
    ldiv!(y, UpperTriangular(P), z)   # Backward substitution with Lᴴ
    return y
end

sym = her = true
opM = LinearOperator(T, n, n, sym, her, (y, x) -> ldiv_ic0!(P, x, y, z))

x, stats = cg(peqn.equation.A, peqn.equation.b, M=opM)

A_cpu = sprand(100, 100, 0.3)

U1 = UpperTriangular(A_cpu)'
U11 = UpperTriangular(A_cpu)'