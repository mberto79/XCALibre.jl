using Plots

using FVM_1D

using Krylov

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/flatplate_2D_laminar.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)

Tref = 273.25 + 25 #K reference temperature

velocity = [400, 0.0, 0.0]
mu = 1E-3
R = 287.0
Cp = 1005.0
Pr = 0.7
T_inf = 300
pressure = 100000
rho = pressure/(R*T_inf)
nu = mu/rho
Re = velocity[1]*1/nu
M = 0.5/sqrt(1.4*R*T_inf)
h_inf = T_inf*Cp
h_wall = 100*1005
gamma = 1.4
model = RANS{Laminar_rho}(mesh=mesh, viscosity=ConstantScalar(nu))
thermodel = ThermoModel{IdealGas}(mesh=mesh, Cp=Cp, gamma=gamma)

@assign! model U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    # Wall(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Neumann(:top, 0.0)
)

 @assign! model p (
    Dirichlet(:inlet, pressure),
    Neumann(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model energy (
    Dirichlet(:inlet, h_inf),
    Neumann(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

schemes = (
    # U = set_schemes(divergence=Upwind),
    # p = set_schemes(divergence=Upwind),
    # energy = set_schemes(divergence=Upwind)
    U = set_schemes(divergence=Upwind),
    p = set_schemes(divergence=Linear),
    energy = set_schemes(divergence=Upwind)
)

solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.7,
        atol        = 1e-6,
        rtol        = 1e-2,
    ),
    p = set_solver(
        model.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = LDL(),
        convergence = 1e-7,
        relax       = 0.3,
        atol        = 1e-6,
        rtol        = 1e-2,
    ),
    energy = set_solver(
        model.energy;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.7,
        atol        = 1e-6,
        rtol        = 1e-2,
    ),
)

runtime = set_runtime(iterations=100, write_interval=10, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, pressure)
initialise!(model.energy, h_inf)

Rx, Ry, Rz, Rp, Re = simple_rho_K_transonic!(model, thermodel, config)

using DelimitedFiles
using LinearAlgebra

# OF_data = readdlm("flatplate_OF_wall_laminar.csv", ',', Float64, skipstart=1)
# oRex = OF_data[:,7].*velocity[1]./nu[1]
# oCf = sqrt.(OF_data[:,9].^2 + OF_data[:,10].^2)/(0.5*velocity[1]^2)

# tauw, pos = wall_shear_stress(:wall, model)
# tauMag = [norm(tauw[i]) for i ∈ eachindex(tauw)]
# x = [pos[i][1] for i ∈ eachindex(pos)]

# Rex = velocity[1].*x/nu[1]
# Cf = 0.664./sqrt.(Rex)
# plot(; xaxis="Rex", yaxis="Cf")
# plot!(Rex, Cf, color=:red, ylims=(0, 0.05), xlims=(0,2e4), label="Blasius",lw=1.5)
# plot!(oRex, oCf, color=:green, lw=1.5, label="OpenFOAM")
# plot!(Rex,tauMag./(0.5*velocity[1]^2), color=:blue, lw=1.5,label="Code")

plot(; xlims=(0,runtime.iterations))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
plot!(1:length(Re), Re, yscale=:log10, label="energy")