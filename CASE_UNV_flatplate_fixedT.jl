using Plots

using FVM_1D

using Krylov

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/flatplate_2D_laminar.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# mesh_gpu = adapt(CUDABackend(), mesh)

velocity = [0.2, 0.0, 0.0]
nu = 1e-4
Re = velocity[1]*1/nu
cp = 1005.0
gamma = 1.4
Pr = 0.7

model = Physics(
    time = Steady(),
    fluid = WeaklyCompressible(
        mu = ConstantScalar(nu),
        cp = ConstantScalar(cp),
        gamma = ConstantScalar(gamma),
        Pr = ConstantScalar(Pr)
        ),
    turbulence = RANS{Laminar}(),
    energy = ENERGY{SensibleEnthalpy}(),
    domain = mesh
    )

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Symmetry(:top, 0.0)
)

 @assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 100000.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model energy h (
    FixedTemperature(:inlet, T=300.0, model=model.energy),
    Neumann(:outlet, 0.0),
    FixedTemperature(:wall, T=310.0, model=model.energy),
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence=Upwind),
    p = set_schemes(divergence=Linear),
    h = set_schemes(divergence=Linear)
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
    ),
    p = set_solver(
        model.momentum.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
    ),
    h = set_solver(
        model.energy.h;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-2,
        atol = 1e-4
    )
)

runtime = set_runtime(iterations=1000, write_interval=100, time_step=1)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 100000.0)
initialise!(model.energy.T, 300.0)

Rx, Ry, Rz, Rp, Re, model_out = run!(model, config) # 9.39k allocs

# using DelimitedFiles
# using LinearAlgebra

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

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")
plot!(1:length(Re), Re, yscale=:log10, label="h")