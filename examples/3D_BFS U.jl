using Plots
using XCALibre
using Krylov
using KernelAbstractions
using CUDA


# bfs_unv_tet_15mm, 10mm, 5mm, 4mm, 3mm
mesh_file = "unv_sample_meshes/bfs_unv_tet_4mm.unv"
mesh_file = "unv_sample_meshes/bfs_unv_tet_5mm.unv"
mesh_file = "unv_sample_meshes/bfs_unv_tet_10mm.unv"

@time mesh = UNV3D_mesh(mesh_file, scale=0.001)

mesh_dev = adapt(CUDABackend(), mesh)

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

@assign! model momentum U (
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

 @assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0),
    Neumann(:sides, 0.0)
)

schemes = (
    U = set_schemes(time=Euler, divergence=Upwind, gradient=Midpoint),
    p = set_schemes(time=Euler, gradient=Midpoint)
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = Bicgstab(), #Bicgstab(), # Bicgstab(), Gmres(), #Cg()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-4,
        atol = 1e-3
    ),
    p = set_solver(
        model.momentum.p;
        solver      = Cg(), #Gmres(), #Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-4,
        atol = 1e-3


    )
)

runtime = set_runtime(
    iterations=1000, time_step=0.001, write_interval=100)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")

# # PROFILING CODE

using Profile, PProf

GC.gc()
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1 begin 
    residuals = run!(model, config)
end

PProf.Allocs.pprof()
