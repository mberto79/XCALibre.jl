using Plots
using XCALibre
using Krylov
using KernelAbstractions
using CUDA


# bfs_unv_tet_15mm, 10mm, 5mm, 4mm, 3mm
mesh_file = "unv_sample_meshes/3D_cylinder.unv"
mesh_file = "unv_sample_meshes/3D_cylinder_extruded_HEX_PRISM_FIXED_2mm.unv"
@time mesh = UNV3D_mesh(mesh_file, scale=0.001)

mesh_dev = adapt(CUDABackend(), mesh)

velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

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
    # Dirichlet(:cylinder, noSlip),
    # Dirichlet(:bottom, velocity),
    # Dirichlet(:top, velocity),
    # Dirichlet(:sides, velocity)
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip),
    Neumann(:freestream, 0.0),
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:cylinder, 0.0),
    Neumann(:freestream, 0.0),
)

schemes = (
    U = set_schemes(time=Euler, divergence=Upwind, gradient=Midpoint),
    p = set_schemes(time=Euler, gradient=Midpoint)
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 0.0,
        atol = 1e-2
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, #SymmlqSolver, #CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 0.0,
        atol = 1e-2
    )
)

runtime = set_runtime(
    iterations=10000, write_interval=100, time_step=0.0025)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
# save_output(model, "WRITE_TEST")

backend = CPU()
backend = CUDABackend()

residuals = run!(model, config)

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
