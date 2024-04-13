using Plots, FVM_1D, Krylov, AerofoilOptimisation

#%% AEROFOIL GEOMETRY DEFINITION
foil,ctrl_p = spline_foil(FoilDef(
    chord   = 100, #[mm]
    LE_h    = 0, #[%c, at α=0°]
    TE_h    = 0, #[%c, at α=0°]
    peak    = [15,5], #[%c]
    trough  = [85,-1], #[%c]
    xover = 70, #[%c]
    α = 5 #[°]
)) #Returns aerofoil MCL & control point vector (spline method)

#%% REYNOLDS & Y+ CALCULATIONS
chord = 100.0
Re = 80000
nu,ρ = 1.48e-5,1.225
yplus_init,BL_layers = 2.0,35
laminar = false
velocity,BL_mesh = BL_calcs(Re,nu,ρ,chord,yplus_init,BL_layers,laminar) #Returns (BL mesh thickness, BL mesh growth rate)

#%% AEROFOIL MESHING
lines = update_mesh(
    chord = foil.chord, #[mm]
    ctrl_p = ctrl_p, #Control point vector
    vol_size = (16,10), #Total fluid volume size (x,y) in chord multiples [aerofoil located in the vertical centre at the 1/3 position horizontally]
    thickness = 1, #Aerofoil thickness [%c]
    BL_thick = 1, #Boundary layer mesh thickness [mm]
    BL_layers = BL_layers, #Boundary layer mesh layers [-]
    BL_stretch = 1.2, #Boundary layer stretch factor (successive multiplication factor of cell thickness away from wall cell) [-]
    ratio = 1.15,
    py_lines = (13,44,51,59,36,68,225,247,284), #SALOME python script relevant lines (notebook path, 3 B-Spline lines,chord line, thickness line, BL line .unv path)
    py_path = "/home/tim/Documents/MEng Individual Project/Julia/AerofoilOptimisation/foil_pythons/FoilMesh.py", #Path to SALOME python script
    salome_path = "/home/tim/Downloads/InstallationFiles/SALOME-9.11.0/mesa_salome", #Path to SALOME installation
    unv_path = "/home/tim/Documents/MEng Individual Project/Julia/FVM_1D_TW/unv_sample_meshes/FoilMesh.unv", #Path to .unv destination
    note_path = "/home/tim/Documents/MEng Individual Project/SALOME", #Path to SALOME notebook (.hdf) destination
    GUI = false #SALOME GUI selector
) #Updates SALOME geometry and mesh to new aerofoil MCL definition


#%% CFD CASE SETUP & SOLVE
# Aerofoil Mesh
mesh_file = "unv_sample_meshes/FoilMesh.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)

# Turbulence Model
νR = 20
Tu = 0.025
k_inlet = 3/2*(Tu*velocity[1])^2
ω_inlet = k_inlet/(νR*nu)
model = RANS{KOmega}(mesh=mesh, viscosity=ConstantScalar(nu))

# Boundary Conditions
noSlip = [0.0, 0.0, 0.0]

@assign! model U ( 
    FVM_1D.Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    FVM_1D.Dirichlet(:top, velocity),
    FVM_1D.Dirichlet(:bottom, velocity),
    FVM_1D.Dirichlet(:foil, noSlip)
)

@assign! model p (
    Neumann(:inlet, 0.0),
    FVM_1D.Dirichlet(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:foil, 0.0)
)

@assign! model turbulence k (
    FVM_1D.Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0),
    FVM_1D.Dirichlet(:foil, 1e-15)
)

@assign! model turbulence omega (
    FVM_1D.Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0),
    OmegaWallFunction(:foil) # need constructor to force keywords
)

@assign! model turbulence nut (
    FVM_1D.Dirichlet(:inlet, k_inlet/ω_inlet),
    Neumann(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0), 
    FVM_1D.Dirichlet(:foil, 0.0)
)


schemes = (
    U = set_schemes(divergence=Upwind,gradient=Midpoint),
    p = set_schemes(divergence=Upwind),
    k = set_schemes(divergence=Upwind,gradient=Midpoint),
    omega = set_schemes(divergence=Upwind,gradient=Midpoint)
)

solvers = (
    U = set_solver(
        model.U;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.7,
    ),
    p = set_solver(
        model.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = LDL(),
        convergence = 1e-7,
        relax       = 0.4,
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.4,
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = ILU0(),
        convergence = 1e-7,
        relax       = 0.4,
    )
)

runtime = set_runtime(
    iterations=1000, write_interval=250, time_step=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

Rx, Ry, Rp = simple!(model, config) #, pref=0.0)

#%% POST-PROCESSING
aero_eff = lift_to_drag(:foil, ρ, model)
C_l,C_d = aero_coeffs(:foil, chord, ρ, velocity, model)
yplus,y = y_plus(:foil,ρ,model)
let
    plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
    plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
    plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
    plot!(1:length(Rp), Rp, yscale=:log10, label="p")
end
paraview_vis(paraview_path = "paraview", #Path to paraview
             vtk_path = "/home/tim/Documents/MEng Individual Project/Julia/FVM_1D_TW/vtk_results/iteration_..vtk") #Path to vtk files