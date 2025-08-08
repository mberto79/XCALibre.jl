using Plots
using XCALibre
using CUDA
using StaticArrays

@inline inflow(vec, t, i) = begin
    vmax = 0.5
    y = vec[2]
    H = 0.1
    H2 = H/2.0
    h = y - H2
    xdir = SVector{3}(1,0,0)
    xvel = vmax*(1.0 - (h/H2)^2.0)
    velocity = xvel*xdir
    return velocity
end

# backwardFacingStep_2mm, 5mm or 10mm
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_10mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)

mesh_dev = adapt(backend, mesh)

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region = mesh_dev,
    (
        U = [
            DirichletFunction(:inlet, inflow),
            # Dirichlet(:inlet, velocity),
            Neumann(:outlet, 0.0),
            Dirichlet(:wall, [0.0, 0.0, 0.0]),
            Dirichlet(:top, [0.0, 0.0, 0.0]),
            # Wall(:wall, [0.0, 0.0, 0.0]),
            # Wall(:top, [0.0, 0.0, 0.0])
            # Symmetry(:top, 0.0)
        ],
        p = [
            Neumann(:inlet, 0.0),
            Dirichlet(:outlet, 0.0),
            Neumann(:wall, 0.0),
            Neumann(:top, 0.0)
            # Symmetry(:top, 0.0)
        ]
    )
)

schemes = (
    U = Schemes(divergence = Linear),
    # U = Schemes(divergence = Upwind),
    p = Schemes()
)


solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-4,
        atol = 1e-10
    )
)

runtime = Runtime(
    iterations=2000, time_step=1, write_interval=100)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config) # 9.39k allocs in 184 iterations

# plot(; xlims=(0,1000))
# plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
# plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
# plot!(1:length(Rp), Rp, yscale=:log10, label="p")
