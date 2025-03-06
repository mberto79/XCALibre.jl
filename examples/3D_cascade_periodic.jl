using XCALibre
# using Adapt
using CUDA

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cascade_3D_periodic_2p5mm.unv"
# grid = "cascade_3D_periodic_4mm.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV3D_mesh(mesh_file, scale=0.001)

backend = CUDABackend(); workgroup=32
# backend = CPU(); workgroup = cld(length(mesh.cells), Threads.nthreads())

periodic1 = construct_periodic(mesh, backend, :top, :bottom)
periodic2 = construct_periodic(mesh, backend, :side1, :side2)

mesh_dev = adapt(backend, mesh)

velocity = [0.25, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu=nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )


    
@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:plate, [0.0, 0.0, 0.0]),
    # Symmetry(:side1, 0.0),
    # Symmetry(:side2, 0.0),
    # Neumann(:side1, 0.0),
    # Neumann(:side2, 0.0),
    periodic1...,
    periodic2...
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:plate, 0.0),
    # Neumann(:side1, 0.0),
    # Neumann(:side2, 0.0),
    periodic1...,
    periodic2...
)

schemes = (
    # U = set_schemes(divergence=Linear, gradient=Midpoint),
    # U = set_schemes(divergence=Upwind, gradient=Midpoint),
    U = set_schemes(divergence=Linear, gradient=Orthogonal),
    # p = set_schemes(gradient=Midpoint)
    p = set_schemes(gradient=Orthogonal)
    # p = set_schemes()
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, #CgSolver, # BicgstabSolver, GmresSolver, #CgSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-1
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
        rtol = 1e-2
    )
)

runtime = set_runtime(
    iterations=1000, time_step=1, write_interval=100)

hardware = set_hardware(backend=backend, workgroup=workgroup)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
# initialise!(model.momentum.U, [0.0, 0.0, 0.0 ])
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)#, ncorrectors=2)

using Plots
fig = plot(; xlims=(0,runtime.iterations), ylims=(1e-10, 1e-4))
plot!(fig, 1:runtime.iterations, residuals.Ux, yscale=:log10, label="Ux")
plot!(fig, 1:runtime.iterations, residuals.Uy, yscale=:log10, label="Uy")
plot!(fig, 1:runtime.iterations, residuals.p, yscale=:log10, label="p")
fig

q =@macroexpand XCALibre.Discretise.@define_boundary Union{PeriodicParent,Periodic} Divergence{Upwind} begin
    a = 0
end