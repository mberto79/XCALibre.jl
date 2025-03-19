using Plots
using XCALibre
using CUDA
using StaticArrays

# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40


grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
# grid = "cylinder_d10mm_2mm.unv"
# grid = "cylinder_d10mm_25mm.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

hardware = set_hardware(backend=CUDABackend(), workgroup=32) # to run with Nvidia GPUs
# hardware = set_hardware(backend=CPU(), workgroup=1024) # for CPU runs

mesh_dev = adapt(hardware.backend, mesh)

# Inlet conditions

velocity = [0.25, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu
δt = 0.005
iterations = 5000
T = iterations*δt

@inline inflow(vec, t, i) = begin
    vx = 0.25
    if t >= 8 && t <= 16 # activate oscillations only when time is within these limits
        amplitude = 0.025
        frequency = 0.25
        vx = vx + amplitude*sin(2π*frequency*t)
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

@assign! model momentum U ( 
    # Dirichlet(:inlet, velocity),
    DirichletFunction(:inlet, inflow),
    Neumann(:outlet, 0.0),
    Wall(:cylinder, noSlip),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:cylinder, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 0,
        atol = 1e-4
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), #NormDiagonal(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 0,
        atol = 1e-5
    )
)

schemes = (
    U = set_schemes(time=Euler, divergence=Upwind, gradient=Orthogonal),
    p = set_schemes(time=Euler, divergence=Upwind, gradient=Orthogonal)
)


runtime = set_runtime(
    iterations=iterations, write_interval=50, time_step=δt)
    # iterations=1, write_interval=50, time_step=0.005)

# 2mm mesh use settings below (to lower Courant number)
# runtime = set_runtime(
    # iterations=5000, write_interval=250, time_step=0.001) # Only runs on 32 bit

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config); #, pref=0.0)

plot(; xlims=(0,runtime.iterations), ylims=(1e-12,1e-4))
plot!(1:length(residuals.Ux), residuals.Ux, yscale=:log10, label="Ux")
plot!(1:length(residuals.Uy), residuals.Uy, yscale=:log10, label="Uy")
plot!(1:length(residuals.p), residuals.p, yscale=:log10, label="p")