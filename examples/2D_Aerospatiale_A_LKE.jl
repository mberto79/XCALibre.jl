using XCALibre
# using CUDA

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "Aerospatiale-A_Airfoil.unv"
mesh_file = joinpath(grids_dir, grid)

 mesh = UNV2D_mesh(mesh_file, scale=0.001)
# mesh = UNV3D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Turbulence Model
Freestream_Velocity = 49.7
AoA = 13.3 # In degrees
velocity = [(Freestream_Velocity*cosd(AoA)),(Freestream_Velocity*sind(AoA)),0] # Adjusts the Velocity to match angle of attack of object
nu = 1.497e-5
# Re = 10*1/nu
νR = 13.9
Tu = 0.002
k_inlet = 0.01482 
# k_inlet = 3/2*(Tu*velocity[1])^2
kL_inlet = 0.00494
# kL_inlet = 1/2*(Tu*velocity[1])^2
ω_inlet = 0.00148
# ω_inlet = k_inlet/(νR*nu) # Omega at the Inlet


# model = RANS{KOmegaLKE}(mesh=mesh, viscosity=nu, Tu=Tu)

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmegaLKE}(Tu = 0.01, walls=(:Upper_Surface, :Lower_Surface)),
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
            Wall(:Upper_Surface, [0.0, 0.0, 0.0]),
            Wall(:Lower_Surface, [0.0, 0.0, 0.0]),
            Extrapolated(:Upper_Freestream),
            Extrapolated(:Lower_Freestream)

        ],
        p = [
            Zerogradient(:Inlet),
            Dirichlet(:Outlet, 0.0),
            Wall(:Upper_Surface),
            Wall(:Lower_Surface),
            Extrapolated(:Upper_Freestream),
            Extrapolated(:Lower_Freestream)

        ],
        k = [
            Dirichlet(:Inlet, k_inlet),
            Zerogradient(:Outlet),
            Dirichlet(:Upper_Surface, 0.0),
            Dirichlet(:Lower_Surface, 0.0),
            Extrapolated(:Upper_Freestream),
            Extrapolated(:Lower_Freestream)

        ],
        kl = [
            Dirichlet(:Inlet, kL_inlet),
            Zerogradient(:Outlet),
            Dirichlet(:Upper_Surface, 0.0),
            Dirichlet(:Lower_Surface, 0.0),
            Extrapolated(:Upper_Freestream),
            Extrapolated(:Lower_Freestream)

        ],
        omega = [
            Dirichlet(:Inlet, ω_inlet),
            Zerogradient(:Outlet),
            OmegaWallFunction(:Upper_Surface),
            OmegaWallFunction(:Lower_Surface),
            Extrapolated(:Upper_Freestream),
            Extrapolated(:Lower_Freestream)

        ],
        nut = [
            Dirichlet(:Inlet, k_inlet/ω_inlet),
            Extrapolated(:Outlet),
            Dirichlet(:Upper_Surface, 0.0),  
            Dirichlet(:Lower_Surface, 0.0),    
            Extrapolated(:Upper_Freestream),
            Extrapolated(:Lower_Freestream)

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
        solver      =  Bicgstab(), # Gmres(), Cg()
        preconditioner = Jacobi(),
        convergence = 3e-5,
        relax       = 0.2,
        rtol = 1e-3,
    ),
    y = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
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
        convergence = 3e-5,
        relax       = 0.3,
        rtol = 1e-2,
    ),
    omega = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 3e-5,
        relax       = 0.3,
        rtol = 1e-2,
    )
)

runtime = Runtime(
    iterations=10000, write_interval=100, time_step=1)

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
# OF_data = readdlm("flatplate_OF_wall_kOmega_lowRe.csv", ',', Float64, skipstart=1)
# oRex = OF_data[:,7].*velocity[1]./nu[1]
# oCf = sqrt.(OF_data[:,12].^2 + OF_data[:,13].^2)/(0.5*velocity[1]^2)

# Ex_data = readdlm("T3A-_Experimental_Data.csv", ',', Float64, skipstart=1)
Ex_data = readdlm("Aerospatiale_A_Experimental_Results.csv", ',', Float64, skipstart=1)
# Ex_data = readdlm("T3B_Experimental_Data.csv", ',', Float64, skipstart=1)
 es_over_C = Ex_data[:,1]
 eCf = Ex_data[:,2]

#  Walt_data = readdlm("T3A-_Walters'_Data.csv", ',', Float64, skipstart=1)
Walt_data = readdlm("Aerospatiale_A_Walters'_Data.csv", ',', Float64, skipstart=1)
#  Walt_data = readdlm("T3B_Walters'_Data.csv", ',', Float64, skipstart=1)
 ws_over_C = Walt_data[:,1]
 wCf = Walt_data[:,2]

 # model_cpu = adapt(CPU(), model)

tauw, pos = wall_shear_stress(:Upper_Surface, model, config)
tauMag = [norm(tauw[i]) for i ∈ eachindex(tauw)]
# tauMag = [tauw.x[i] for i ∈ eachindex(tauw)]
x = [pos[i][1] for i ∈ eachindex(pos)]
# Rex = velocity[1].*x./nu
s_over_C = x.*10

x_corr = [0:0.0002:2;]
Rex_corr = velocity[1].*x_corr/nu
Cf_corr = 0.0576.*(Rex_corr).^(-1/5)
Cf_laminar = 0.664.*(Rex_corr).^(-1/2)

plot(; xaxis="s/C", yaxis="Cf")
# plot!(oRex, oCf, color=:green, lw=1.5, label="OpenFOAM") # |> display
scatter!(es_over_C, eCf, color=:green, lw=1.5, label="Aerospatiale-A Experimantal Data")
plot!(ws_over_C, wCf, color=:black, lw=1.5,label="Walters' Original model") 
plot!(s_over_C,tauMag./(0.5*velocity[1]^2), color=:blue, lw=1.5,label="XCALibre", title="Aerospatiale-A Airfoil") |> display


# plot(; xlims=(0,1000))
# plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
# plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
# plot!(1:length(Rp), Rp, yscale=:log10, label="p") |> display

