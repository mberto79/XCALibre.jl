using XCALibre
# using CUDA

# backwardFacingStep_2mm, backwardFacingStep_10mm
# mesh_file = "unv_sample_meshes/flatplate_transition.unv"
# mesh_file = "unv_sample_meshes/flatplate_2D_lowRe.unv"
# mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "kOmegaSST/constant/polymesh"
mesh_file = joinpath(grids_dir, grid)

mesh = FOAM3D_mesh(mesh_file, scale=1)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Turbulence Model
velocity = [10,0,0]
nu = 1e-5
Re = 10*1/nu
νR = 15
Tu = 0.03
k_inlet = 1.08e-3 #3/2*(Tu*velocity[1])^2
ω_inlet = 8675 #k_inlet/(νR*nu)

# model = RANS{KOmegaLKE}(mesh=mesh, viscosity=nu, Tu=Tu)

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmegaSST}(walls=(:wall,)),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Zerogradient(:outlet),
            Wall(:topWall, [0.0, 0.0, 0.0]),
            Wall(:bottomWall, [0.0, 0.0, 0.0]),
            # Extrapolated(:top)
            # Zerogradient(:top)
            Empty(:frontAndBack),
            Symmetry(:symmetry)
        ],
        p = [
            Zerogradient(:inlet),
            Dirichlet(:outlet, 0.0),
            Wall(:topWall),
            Wall(:bottomWall),
            # Extrapolated(:top)
            Empty(:frontAndBack),
            Symmetry(:symmetry)
        ],
        k = [
            Dirichlet(:inlet, k_inlet),
            Zerogradient(:outlet),
            Dirichlet(:topWall, 0.0),
            Dirichlet(:bottomWall, 0.0),
            # Extrapolated(:top)
            # Zerogradient(:top)
            Empty(:frontAndBack),
            Symmetry(:symmetry)
        ],
        omega = [
            Dirichlet(:inlet, ω_inlet),
            Zerogradient(:outlet),
            OmegaWallFunction(:topWall),
            OmegaWallFunction(:bottomWall),
            # Extrapolated(:top)
            # Zerogradient(:top)
            Empty(:frontAndBack),
            Symmetry(:symmetry)
        ],
        nut = [
            Dirichlet(:inlet, k_inlet/ω_inlet),
            Extrapolated(:outlet),
            Dirichlet(:topWall, 0.0), 
            Dirichlet(:bottomWall, 0.0), 
            # Extrapolated(:top)
            # Zerogradient(:top)
            Empty(:frontAndBack),
            Symmetry(:symmetry)
        ]
    )
)

schemes = (
    U = Schemes(divergence=LUST),
    p = Schemes(divergence=LUST),
    k = Schemes(divergence=LUST),
    y = Schemes(gradient=Midpoint),
    omega = Schemes(divergence=LUST)
)


solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.7,
        rtol = 1e-2,
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres(), Cg()
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.2,
        rtol = 1e-3,
    ),
    y = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-8,
        rtol = 1e-2,
        relax       = 0.9,
    ),
    k = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-2,
    ),
    omega = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-2,
    )
)

runtime = Runtime(
    iterations=1000, write_interval=10, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config); #, pref=0.0) # 9.39k allocs


# let
#     using Plots
#     p = plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
#     plot!(1:length(residuals.Ux), residuals.Ux, yscale=:log10, label="Ux")
#     plot!(1:length(residuals.Uy), residuals.Uy, yscale=:log10, label="Uy")
#     plot!(1:length(residuals.p), residuals.p, yscale=:log10, label="p")
#     display(p)
# end

# using DelimitedFiles
# using LinearAlgebra
# using Plots 
# # OF_data = readdlm("flatplate_OF_wall_kOmega_lowRe.csv", ',', Float64, skipstart=1)
# # oRex = OF_data[:,7].*velocity[1]./nu[1]
# # oCf = sqrt.(OF_data[:,12].^2 + OF_data[:,13].^2)/(0.5*velocity[1]^2)

# model_cpu = adapt(CPU(), model)

# tauw, pos = wall_shear_stress(:wall, model_cpu)
# tauMag = [norm(tauw[i]) for i ∈ eachindex(tauw)]
# tauMag = [tauw.x[i] for i ∈ eachindex(tauw)]
# x = [pos[i][1] for i ∈ eachindex(pos)]
# Rex = velocity[1].*x./nu

# x_corr = [0:0.0002:2;]
# Rex_corr = velocity[1].*x_corr/nu
# Cf_corr = 0.0576.*(Rex_corr).^(-1/5)
# Cf_laminar = 0.664.*(Rex_corr).^(-1/2)

# plot(; xaxis="Rex", yaxis="Cf")
# plot!(Rex_corr, Cf_corr, color=:red, ylims=(0, 0.01), xlims=(0,6e5), label="Turbulent",lw=1.5)
# plot!(Rex_corr, Cf_laminar, color=:green, ylims=(0, 0.01), xlims=(0,6e5), label="Laminar",lw=1.5)
# # plot!(oRex, oCf, color=:green, lw=1.5, label="OpenFOAM") # |> display
# plot!(Rex,tauMag./(0.5*velocity[1]^2), color=:blue, lw=1.5,label="Code") |> display

# plot(; xlims=(0,1000))
# plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
# plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
# plot!(1:length(Rp), Rp, yscale=:log10, label="p") |> display
