using Plots
using FVM_1D
using Krylov
using KernelAbstractions
using CUDA


# bfs_unv_tet_15mm, 10mm, 5mm, 4mm, 3mm
mesh_file = "unv_sample_meshes/bfs_unv_tet_4mm.unv"
mesh_file = "unv_sample_meshes/bfs_unv_tet_5mm.unv"
mesh_file = "unv_sample_meshes/bfs_unv_tet_10mm.unv"

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
        solver      = BicgstabSolver, #CgSolver, # BicgstabSolver, GmresSolver, #CgSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-1,
        atol = 1e-5
    ),
    p = set_solver(
        model.p;
        solver      = CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-2,
        atol = 1e-5


    )
)

runtime = set_runtime(
    iterations=500, time_step=1, write_interval=500)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)

Rx, Ry, Rz, Rp, model = simple!(model, config)

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
    Rx, Ry, Rz, Rp, model = simple!(model, config)
end

PProf.Allocs.pprof()
