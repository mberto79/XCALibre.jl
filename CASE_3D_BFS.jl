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
        solver      = CgSolver, #QmrSolver, # BicgstabSolver, GmresSolver, #CgSolver
        # preconditioner = CUDA_ILU2(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 0.0,
        atol = 1e-2
    ),
    p = set_solver(
        model.p;
        solver      = CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        # preconditioner = CUDA_IC0(),
        preconditioner = Jacobi(),

        convergence = 1e-7,
        relax       = 0.3,
        rtol = 0.0,
        atol = 1e-3


    )
)

runtime = set_runtime(
    iterations=2000, time_step=1, write_interval=2000)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, [0.0,0.0,0.0])
initialise!(model.U, velocity)
initialise!(model.p, 0.0)

backend = CPU()
backend = CUDABackend()

Rx, Ry, Rz, Rp, model = simple!(model, config, backend)

term_scheme_calls = Expr(:block)
    for t in 1:3
        func_calls = quote
            # scheme!(terms[$t], nzval_array, cell1, face,  cell2, ns, cIndex1, nIndex1, fID, prev, runtime)
            # scheme!(terms[$t], nzval_array, cell2, face,  cell1, -ns, cIndex2, nIndex2, fID, prev, runtime)

            Ac1, An1 = scheme!(terms[$t], cell1, face,  cell2, ns, cIndex1, nIndex1, fID, prev, runtime)
            Ac1 += Ac1 
            An1 += An1
            Ac2, An2 = scheme!(terms[$t], cell2, face,  cell1, -ns, cIndex2, nIndex2, fID, prev, runtime)
            Ac2 += Ac2 
            An2 += An2
        end
        push!(term_scheme_calls.args, func_calls.args...)
    end
    return_quote = quote
        z = zero(typeof(face.area))
        Ac1 = 0.0
        Ac2 = 0.0
        An1 = 0.0 
        An2 = 0.0 
        $(term_scheme_calls.args...)
        return Ac1, An1, Ac2, An2
    end
    return_quote