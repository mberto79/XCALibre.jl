using XCALibre
# using CUDA

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "EROFATC_Plate_Example7_2.unv"
mesh_file = joinpath(grids_dir, grid)

 mesh = UNV2D_mesh(mesh_file, scale=0.001)
# mesh = UNV3D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Turbulence Model
velocity = [5.4,0,0]
nu = 1.497e-5
# Re = 10*1/nu
νR = 13.9
Tu = 0.03
k_inlet = 0.0575
# k_inlet = 3/2*(Tu*velocity[1])^2
kL_inlet = 0.0115
# kL_inlet = 1/2*(Tu*velocity[1])^2
ω_inlet = 275
# ω_inlet = k_inlet/(νR*nu) # Omega at the Inlet


# model = RANS{KOmegaLKE}(mesh=mesh, viscosity=nu, Tu=Tu)

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmegaLKE}(Tu = 0.01, walls=(:Wall,)),
    # turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
)

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:Inlet, velocity),
            Zerogradient(:Outlet),
            Wall(:Wall, [0.0, 0.0, 0.0]),
            Extrapolated(:Freestream),
            Extrapolated(:Freestream_Sym),
            # Zerogradient(:top)
            Symmetry(:Sym)
        ],
        p = [
            Zerogradient(:Inlet),
            Dirichlet(:Outlet, 0.0),
            Wall(:Wall),
            Extrapolated(:Freestream),
            Extrapolated(:Freestream_Sym),
            # Zerogradient(:top)
            Symmetry(:Sym)
        ],
        k = [
            Dirichlet(:Inlet, k_inlet),
            Zerogradient(:Outlet),
            Dirichlet(:Wall, 0.0),
            # KWallFunction(:Wall),
            Extrapolated(:Freestream),
            Extrapolated(:Freestream_Sym),
            # Zerogradient(:top)
            Symmetry(:Sym)
        ],
        kl = [
            Dirichlet(:Inlet, kL_inlet),
            Zerogradient(:Outlet),
            Dirichlet(:Wall, 0.0),
            Extrapolated(:Freestream),
            Extrapolated(:Freestream_Sym),
            # Zerogradient(:top)
            Symmetry(:Sym)
        ],
        omega = [
            Dirichlet(:Inlet, ω_inlet),
            Zerogradient(:Outlet),
            OmegaWallFunction(:Wall),
            Extrapolated(:Freestream),
            Extrapolated(:Freestream_Sym),
            # Zerogradient(:top)
            Symmetry(:Sym)
        ],
        nut = [
            Dirichlet(:Inlet, k_inlet/ω_inlet),
            Extrapolated(:Outlet),
            Dirichlet(:Wall, 0.0),  
            # NutWallFunction(:Wall),  
            Extrapolated(:Freestream),
            Extrapolated(:Freestream_Sym),
            # Zerogradient(:top)
            Symmetry(:Sym)
        ]
    )
)

schemes = (
    U = Schemes(divergence=LUST, limiter=MFaceBased(mesh_dev)),
    # U = Schemes(divergence=LUST),
    p = Schemes(divergence=LUST),
    k = Schemes(divergence=LUST),
    y = Schemes(gradient=Gauss),
    kl = Schemes(divergence=LUST),
    omega = Schemes(divergence=LUST)
)


solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 3e-6,
        relax       = 0.7,
        rtol = 1e-2,
    ),
    p = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres(), Cg()
        preconditioner = Jacobi(),
        convergence = 3e-6,
        relax       = 0.2,
        rtol = 1e-3,
    ),
    y = SolverSetup(
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-6,
        rtol = 1e-3,
        relax       = 0.9,
    ),
    kl = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 3e-6,
        relax       = 0.3,
        rtol = 1e-2,
    ),
    k = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 4e-6,
        relax       = 0.3,
        rtol = 1e-2,
    ),
    omega = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 3e-6,
        relax       = 0.3,
        rtol = 1e-2,
    )
)

runtime = Runtime(
    iterations=3000, write_interval=100, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.kl, kL_inlet)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config); #, pref=0.0) # 9.39k allocs


 let
     using Plots
     p = plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
     plot!(1:length(residuals.Ux), residuals.Ux, yscale=:log10, label="Ux")
     plot!(1:length(residuals.Uy), residuals.Uy, yscale=:log10, label="Uy")
     plot!(1:length(residuals.p), residuals.p, yscale=:log10, label="p")
     display(p)
 end

using DelimitedFiles
using LinearAlgebra
using Plots 
using LaTeXStrings
# OF_data = readdlm("flatplate_OF_wall_kOmega_lowRe.csv", ',', Float64, skipstart=1)
# oRex = OF_data[:,7].*velocity[1]./nu[1]
# oCf = sqrt.(OF_data[:,12].^2 + OF_data[:,13].^2)/(0.5*velocity[1]^2)

# Ex_data = readdlm("T3A-_Experimental_Data.csv", ',', Float64, skipstart=1)
 Ex_data = readdlm("T3A_Experimental_Results.csv", ',', Float64, skipstart=1)
# Ex_data = readdlm("T3B_Experimental_Data.csv", ',', Float64, skipstart=1)
 eRex = Ex_data[:,1]
 eCf = Ex_data[:,2]

# Walt_data = readdlm("T3A-_Walters'_Data.csv", ',', Float64, skipstart=1)
 Walt_data = readdlm("T3A_Walters'_Data.csv", ',', Float64, skipstart=1)
# Walt_data = readdlm("T3B_Walters'_Data.csv", ',', Float64, skipstart=1)
 wRex = Walt_data[:,1]
 wCf = Walt_data[:,2]

 # model_cpu = adapt(CPU(), model)

tauw, pos = wall_shear_stress(:Wall, model, config)
tauMag = [norm(tauw[i]) for i ∈ eachindex(tauw)]
# tauMag = [tauw.x[i] for i ∈ eachindex(tauw)]
x = [pos[i][1] for i ∈ eachindex(pos)]
Rex = velocity[1].*x./nu

x_corr = [0:0.0002:2;]
Rex_corr = velocity[1].*x_corr/nu
Cf_corr = 0.0576.*(Rex_corr).^(-1/5)
Cf_laminar = 0.664.*(Rex_corr).^(-1/2)

plot(; xaxis= L"Re_{x} (-)", yaxis= L"C_{f} (-)")
plot!(Rex_corr, Cf_corr, linestyle = :dot, color=:red, ylims=(0, 0.01), xlims=(0,6e5), label="Turbulent",lw=1.5)
plot!(Rex_corr, Cf_laminar, linestyle = :dot, color=:green, ylims=(0, 0.01), xlims=(0,6e5), label="Laminar",lw=2.5)
# plot!(oRex, oCf, color=:green, lw=1.5, label="OpenFOAM") # |> display
plot!(wRex, wCf, linestyle = :dash, color=:black, lw=1.5,label="Walters' Original model") |> display
plot!(Rex,tauMag./(0.5*velocity[1]^2), color=:blue, lw=1.5,label="XCALibre") |> display
scatter!(eRex, eCf, color=:green, lw=1.5, label="T3A Experimantal Data")|> display

# plot(; xlims=(0,1000))
# plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
# plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
# plot!(1:length(Rp), Rp, yscale=:log10, label="p") |> display
