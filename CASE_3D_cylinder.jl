using Plots
using FVM_1D
using Krylov
using KernelAbstractions
using CUDA


# bfs_unv_tet_15mm, 10mm, 5mm, 4mm, 3mm
mesh_file = "unv_sample_meshes/3D_cylinder.unv"
mesh_file = "unv_sample_meshes/3D_cylinder_extruded_HEX_PRISM.unv"
mesh_file = "unv_sample_meshes/3D_cylinder_extruded_HEX_PRISM_FIXED.unv"
mesh_file = "unv_sample_meshes/3D_cylinder_extruded_HEX_PRISM_FIXED_2mm.unv"
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
    Neumann(:freestream, 0.0),
)

@assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:cylinder, 0.0),
    Neumann(:freestream, 0.0),
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Midpoint),
    p = set_schemes(gradient=Midpoint)
)


solvers = (
    U = set_solver(
        model.U;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.6,
        rtol = 1e-4,
        atol = 1e-2
    ),
    p = set_solver(
        model.p;
        solver      = CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.4,
        rtol = 1e-4,
        atol = 1e-3
    )
)

runtime = set_runtime(
    iterations=500, time_step=1, write_interval=500)

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
