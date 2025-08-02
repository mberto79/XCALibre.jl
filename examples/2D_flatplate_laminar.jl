using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "flatplate_2D_laminar.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

velocity = [0.2, 0.0, 0.0]
nu = 1e-5
Re = velocity[1]*1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Extrapolated(:top)
            # Zerogradient(:top)
            # Symmetry(:top)
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Wall(:wall),
            Extrapolated(:top)
            # Zerogradient(:top)
            # Symmetry(:top)
        ]
    )
)

div = Upwind # Linear, Upwind
schemes = (
    U = Schemes(divergence=div),
    p = Schemes()
)


solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.8,
    ),
    p = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.2,
    )
)

runtime = Runtime(iterations=2000, write_interval=1000, time_step=1)

configure!(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model) # 9.39k allocs

# using Plots
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

# plot(; xlims=(0,1000))
# plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
# plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
# plot!(1:length(Rp), Rp, yscale=:log10, label="p")