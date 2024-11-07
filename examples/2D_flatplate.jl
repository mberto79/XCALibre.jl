using Plots

using XCALibre

using Krylov

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/flatplate_2D_laminar.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# mesh_dev = adapt(CUDABackend(), mesh)
mesh_dev = mesh

velocity = [0.2, 0.0, 0.0]
nu = 1e-5
Re = velocity[1]*1/nu

model = Physics(
    time = Steady(),
    fluid = Incompressible(
        nu = nu, 
        rho = ConstantScalar(1.0)
        ),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Neumann(:top, 0.0)
)

 @assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence=Linear),
    p = set_schemes(divergence=Linear)
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
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
    )
)

runtime = set_runtime(iterations=2000, write_interval=1000, time_step=1)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config) # 9.39k allocs

using DelimitedFiles
using LinearAlgebra

OF_data = readdlm("flatplate_OF_wall_laminar.csv", ',', Float64, skipstart=1)
oRex = OF_data[:,7].*velocity[1]./nu[1]
oCf = sqrt.(OF_data[:,9].^2 + OF_data[:,10].^2)/(0.5*velocity[1]^2)

tauw, pos = wall_shear_stress(:wall, model)
tauMag = [norm(tauw[i]) for i ∈ eachindex(tauw)]
x = [pos[i][1] for i ∈ eachindex(pos)]

Rex = velocity[1].*x/nu[1]
Cf = 0.664./sqrt.(Rex)
plot(; xaxis="Rex", yaxis="Cf")
plot!(Rex, Cf, color=:red, ylims=(0, 0.05), xlims=(0,2e4), label="Blasius",lw=1.5)
plot!(oRex, oCf, color=:green, lw=1.5, label="OpenFOAM")
plot!(Rex,tauMag./(0.5*velocity[1]^2), color=:blue, lw=1.5,label="Code")

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")