using Plots, FVM_1D, Krylov, AerofoilOptimisation

#%% AEROFOIL GEOMETRY DEFINITION
@time foil,ctrl_p = spline_foil(FoilDef(
    chord   = 100, #[mm]
    LE_h    = 0, #[%c, at α=0°]
    TE_h    = 0, #[%c, at α=0°]
    peak    = [25,2.5], #[%c]
    trough  = [80,-7.5], #[%c]
    xover = 50, #[%c]
    α = 0 #[°]
)) #Returns aerofoil MCL & control point vector (spline method)

#%% AEROFOIL MESHING
lines = update_mesh(
    chord = foil.chord, #[mm]
    ctrl_p = ctrl_p, #Control point vector
    vol_size = (16,10), #Total fluid volume size (x,y) in chord multiples [aerofoil located in the vertical centre at the 1/3 position horizontally]
    thickness = 1, #Aerofoil thickness [%c]
    BL_thick = 1, #Boundary layer mesh thickness [%c]
    BL_layers = 35, #Boundary layer mesh layers [-]
    BL_stretch = 1.2, #Boundary layer stretch factor (successive multiplication factor of cell thickness away from wall cell) [-]
    py_lines = (13,44,51,59,36,68,247,284), #SALOME python script relevant lines (notebook path, 3 B-Spline lines,chord line, thickness line, BL line .unv path)
    py_path = "/home/tim/Documents/MEng Individual Project/Julia/AerofoilOptimisation/FoilMesh.py", #Path to SALOME python script
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

# Inlet conditions
chord = 100
velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1.48e-5
Re = (chord*0.001*velocity[1])/nu

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:top, velocity),
    Dirichlet(:bottom, velocity),
    Dirichlet(:foil, noSlip)
)

@assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:foil, 0.0)
)

schemes = (
    U = set_schemes(time=Euler, divergence=Upwind),
    p = set_schemes(time=Euler, divergence=Upwind)
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
        relax       = 0.3,
    )
)

runtime = set_runtime(
    iterations=5000, write_interval=10, time_step=0.001)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)

Rx, Ry, Rp = simple!(model, config) #, pref=0.0)

#%% POST-PROCESSING
aero_eff = foil_obj_func(:foil, model.p, model.U, 1.25, nu, model.turbulence)
let
    plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
    plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
    plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
    plot!(1:length(Rp), Rp, yscale=:log10, label="p")
end
paraview_vis(paraview_path = "paraview", #Path to paraview
             vtk_path = "/home/tim/Documents/MEng Individual Project/Julia/FVM_1D_TW/vtk_results/iteration_..vtk") #Path to vtk files