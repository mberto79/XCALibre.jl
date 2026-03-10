using XCALibre
# using CUDA

# backwardFacingStep_2mm, backwardFacingStep_10mm
# mesh_file = "unv_sample_meshes/flatplate_transition.unv"
# mesh_file = "unv_sample_meshes/flatplate_2D_lowRe.unv"
# mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"

grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
grid = "EROFATC_Plate_Example7_2.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Turbulence Model
velocity = [19.3,0,0]
nu = 1.48e-5
# Re = 10*1/nu
νR = 7.8
Tu = 0.009
# k_inlet = 0.0575  
k_inlet = 3/2*(Tu*velocity[1])^2
kL_inlet = 1/2*(Tu*velocity[1])^2
ω_inlet = k_inlet/(νR*nu)
rho = 1

# model = RANS{KOmegaLKE}(mesh=mesh, viscosity=nu, Tu=Tu)

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu, rho = rho),
    turbulence = RANS{KOmegaLKE}(Tu = Tu, walls=(:Wall,)),
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
    # U = Schemes(divergence=LUST, limiter=MFaceBased(mesh_dev)),
    U = Schemes(divergence=LUST),
    p = Schemes(divergence=LUST),
    k = Schemes(divergence=LUST),
    y = Schemes(gradient=Midpoint),
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
        convergence = 3e-6,
        rtol = 1e-2,
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
        convergence = 3e-6,
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
    iterations=2500, write_interval=100, time_step=1)

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

 Ex_data = readdlm("T3A_Experimental_Results.csv", ',', Float64, skipstart=1)
 eRex = Ex_data[:,1]
 eCf = Ex_data[:,2]

 # OF_data = readdlm("T3A_Experimental_Results.csv", ',', Float64, skipstart=1)
 # oRex = OF_data[:,7].*velocity[1]./nu[1]
 # oCf = sqrt.(OF_data[:,12].^2 + OF_data[:,13].^2)/(0.5*velocity[1]^2)

 # model_cpu = adapt(CPU(), model)

  tauw, pos = wall_shear_stress(:Wall, model, config)
  tauMag = [norm(tauw[i]) for i ∈ eachindex(tauw)]
  tauMag = [tauw.x[i] for i ∈ eachindex(tauw)]
  x = [pos[i][1] for i ∈ eachindex(pos)]
  Rex = velocity[1].*x./nu

  ustar = (tauMag./rho).^(1/2); # Friction velocity
  yplus = ((2*ustar)/nu)
  #print(yplus)

 x_corr = [0:0.0002:2;]
 Rex_corr = velocity[1].*x_corr/nu
 Cf_corr = 0.0576.*(Rex_corr).^(-1/5)
 Cf_laminar = 0.664.*(Rex_corr).^(-1/2)

 p = plot(; xaxis="Rex", yaxis="Cf")
 plot!(Rex_corr, Cf_corr, color=:red, ylims=(0, 0.01), xlims=(0,6e5), label="Turbulent",lw=1.5)
 plot!(Rex_corr, Cf_laminar, color=:green, ylims=(0, 0.01), xlims=(0,6e5), label="Laminar",lw=1.5)
 scatter!(eRex, eCf, color=:green, label="Experimental T3A Data") # |> display
 # plot!(oRex, oCf, color=:green, lw=1.5,label="OpenFoam") |> display
#  plot!(Rex,tauMag./(0.5.*velocity[1]^2), color=:blue, lw=1.5,label="Code") |> display
  plot!(Rex,tauMag./(0.5.*velocity[1]^2), color=:blue, lw=1.5,label="Code", title = "T3A Validation Case") |> display
#  plot!(Rex,tauMag./(0.5.*velocity[1]^2), color=:purple, lw=1.5,label="Code wth -5% damping") |> display
#  plot!(Rex,tauMag./(0.5.*velocity[1]^2), color=:Yellow, lw=1.5,label="Code wth -10% damping", title = "T3B Damping Sensitivity Study") |> display

 #savefig(p,"EROFATC_Plate_3.svg")

# plot(; xlims=(0,1000))
# plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
# plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
# plot!(1:length(Rp), Rp, yscale=:log10, label="p") |> display
