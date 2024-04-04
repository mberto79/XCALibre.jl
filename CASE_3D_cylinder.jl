using Plots
using FVM_1D
using Krylov
using KernelAbstractions
using CUDA


# bfs_unv_tet_15mm, 10mm, 5mm, 4mm, 3mm
mesh_file = "unv_sample_meshes/3D_cylinder.unv"
@time mesh = build_mesh3D(mesh_file, scale=0.001)

velocity = [0.50, 0.0, 0.0]
velocity = [0.50, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U ( 
    # Dirichlet(:inlet, velocity),
    # Neumann(:outlet, 0.0),
    # Dirichlet(:cylinder, noSlip),
    # Dirichlet(:bottom, velocity),
    # Dirichlet(:top, velocity),
    # Dirichlet(:sides, velocity)
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0),
    Neumann(:sides, 0.0)
)

@assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:cylinder, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0),
    Neumann(:sides, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Midpoint),
    p = set_schemes(gradient=Midpoint)
)


solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-3
    ),
    p = set_solver(
        model.p;
        solver      = CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-3

    )
)

runtime = set_runtime(
    iterations=500, time_step=1, write_interval=100)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)
# model2vtk(model, "WRITE_TEST")

backend = CPU()
backend = CUDABackend()

Rx, Ry, Rz, Rp, model1 = simple!(model, config, backend)

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")

# # PROFILING CODE

using Profile, PProf

GC.gc()
initialise!(model.U, velocity)
initialise!(model.p, 0.0)

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1 begin 
    Rx, Ry, Rp = simple!(model, config)
end

PProf.Allocs.pprof()
