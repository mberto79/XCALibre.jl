using Plots
using FVM_1D
using Krylov
using KernelAbstractions
using CUDA


# bfs_unv_tet_15mm, 10mm, 5mm, 4mm, 3mm
mesh_file = "unv_sample_meshes/3D_cylinder.unv"
@time mesh = build_mesh3D(mesh_file, scale=0.001)

velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U ( 
    # Dirichlet(:inlet, velocity),
    # Neumann(:outlet, 0.0),
    # Dirichlet(:cylinder, noSlip),
    # Dirichlet(:freestream, velocity)
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip),
    Neumann(:freestream, 0.0)
)

@assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:cylinder, 0.0),
    Neumann(:freestream, 0.0)
)

schemes = (
    U = set_schemes(time=Euler, divergence=Upwind),
    p = set_schemes(time=Euler, divergence=Upwind)
)


solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = NormDiagonal(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-4,
        atol = 1e-3
    ),
    p = set_solver(
        model.p;
        solver      = CgSolver, #SymmlqSolver, #CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = NormDiagonal(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-4,
        atol = 1e-3
    )
)

runtime = set_runtime(
    iterations=1000, write_interval=50, time_step=0.005)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)
# model2vtk(model, "WRITE_TEST")

backend = CPU()
backend = CUDABackend()

Rx, Ry, Rz, Rp, model1 = piso!(model, config, backend)

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
