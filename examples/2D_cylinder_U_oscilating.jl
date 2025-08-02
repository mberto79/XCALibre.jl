using Plots
using XCALibre
# using CUDA
using StaticArrays

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
# grid = "cylinder_d10mm_2mm.unv"
# grid = "cylinder_d10mm_25mm.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Inlet conditions

velocity = [0.25, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu
δt = 0.0025
iterations = 5000
total_time = iterations*δt

@inline inflow(vec, t, i) = begin
    vx = 0.25
    if t >= 8 && t <= 16 # activate oscillations only when time is within these limits
        amplitude = 0.025
        frequency = 0.25
        vx += amplitude*sin(2π*frequency*t)
        return velocity = SVector{3}(vx, 0.0, 0.0)
    else 
        return velocity = SVector{3}(vx,0.0,0.0)
    end
end

model = Physics(
    time = Transient(),
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
            Zerogradient(:outlet),
            Wall(:cylinder, noSlip),
            Extrapolated(:bottom),
            Extrapolated(:top)
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0),
            Wall(:cylinder),
            Extrapolated(:bottom),
            Extrapolated(:top)
        ]
    )
)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 0,
        atol = 1e-5
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), #NormDiagonal(),
        convergence = 1e-7,
        relax       = 0.9,
        rtol = 0,
        atol = 1e-6
    )
)

schemes = (
    U = Schemes(time=Euler, divergence=Upwind, gradient=Gauss),
    p = Schemes(time=Euler, divergence=Upwind, gradient=Gauss)
)


runtime = Runtime(iterations=iterations, write_interval=50, time_step=δt)

configure!(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model); #, pref=0.0)

# plot(; xlims=(0,runtime.iterations), ylims=(1e-12,1e-4))
# plot!(1:length(residuals.Ux), residuals.Ux, yscale=:log10, label="Ux")
# plot!(1:length(residuals.Uy), residuals.Uy, yscale=:log10, label="Uy")
# plot!(1:length(residuals.p), residuals.p, yscale=:log10, label="p")