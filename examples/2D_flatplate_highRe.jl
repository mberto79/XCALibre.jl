# using Plots
using XCALibre
# using CUDA

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "flatplate_2D_highRe.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU(); # activate_multithread(backend)
mesh_dev = mesh; workgroup = 1024
# backend = CUDABackend()
# mesh_dev = adapt(backend, mesh); workgroup= 32

velocity = [10, 0.0, 0.0]
nu = 1e-5
Re = velocity[1]*1/nu
k_inlet = 0.375
ω_inlet = 1000

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, [0.0, 0.0, 0.0]),
    # Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Neumann(:top, 0.0)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    KWallFunction(:wall),
    # Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model turbulence omega (
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    OmegaWallFunction(:wall),
    Neumann(:top, 0.0)
)

@assign! model turbulence nut (
    Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    NutWallFunction(:wall), 
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind, gradient=Midpoint),
    p = set_schemes(divergence=Upwind, gradient=Midpoint),
    k = set_schemes(divergence=Upwind, gradient=Midpoint),
    omega = set_schemes(divergence=Upwind, gradient=Midpoint)
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.8,
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.2,
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.3,
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.3,
    )
)

runtime = set_runtime(iterations=500, write_interval=100, time_step=1)

hardware = set_hardware(backend=backend, workgroup=workgroup)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config) # 9.39k allocs

# using DelimitedFiles
# using LinearAlgebra

# OF_data = readdlm("flatplate_OF_wall_kOmega_highRe.csv", ',', Float64, skipstart=1)
# oRex = OF_data[:,7].*velocity[1]./nu[1]
# oCf = sqrt.(OF_data[:,12].^2 + OF_data[:,13].^2)/(0.5*velocity[1]^2)

# tauw, pos = wall_shear_stress(:wall, model)
# tauMag = [norm(tauw[i]) for i ∈ eachindex(tauw)]
# x = [pos[i][1] for i ∈ eachindex(pos)]
# Rex = velocity[1].*x./nu

# x_corr = [0:0.0002:1;]
# Rex_corr = velocity[1].*x_corr/nu
# Cf_corr = 0.074.*(Rex_corr).^(-1/5)
# plot(; xaxis="Rex", yaxis="Cf")
# plot!(Rex_corr, Cf_corr, color=:red, ylims=(0, 0.02), label="Blasius",lw=1.5)
# plot!(oRex, oCf, color=:green, lw=1.5, label="OpenFOAM") |> display
# plot!(Rex,tauMag./(0.5*velocity[1]^2), color=:blue, lw=1.5,label="Code") |> display

# plot(; xlims=(0,1000))
# plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
# plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
# plot!(1:length(Rp), Rp, yscale=:log10, label="p") |> display