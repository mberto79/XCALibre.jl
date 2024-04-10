using Plots, FVM_1D, Krylov, AerofoilOptimisation

#%% REYNOLDS & Y+ CALCULATIONS
chord = 500.0
Re = 10000
nu,ρ = 1.48e-5,1.225
yplus_init,BL_layers = 1.0,35
laminar = true
velocity,BL_mesh = BL_calcs(Re,nu,ρ,chord,yplus_init,BL_layers,laminar) #Returns (BL mesh thickness, BL mesh growth rate)

#%% CFD CASE SETUP & SOLVE
iter = 15
aero_eff = Array{Float64,1}(undef,iter)
C_l = Array{Float64,1}(undef,iter)
C_d = Array{Float64,1}(undef,iter)
for i ∈ 15:iter
    α = i-1

    # Aerofoil Mesh
    create_NACA_mesh(
        chord = chord, #[mm]
        α = α, #[°]
        cutoff = 0.5*(chord/100), #Min thickness of TE [mm]. Default = 0.5; reduce for aerofoils with very thin TE
        vol_size = (16,10), #Total fluid volume size (x,y) in chord multiples [aerofoil located in the vertical centre at the 1/3 position horizontally]
        BL_thick = 1, #Boundary layer mesh thickness [%c]
        BL_layers = BL_layers, #Boundary layer mesh layers [-]
        BL_stretch = 1.2, #Boundary layer stretch factor (successive multiplication factor of cell thickness away from wall cell) [-]
        py_lines = (14,37,44,248,358,391,405), #SALOME python script relevant lines (notebook path, chord line, points line, splines line, BL thickness, foil end BL fidelity, .unv path)
        dat_path = "/home/tim/Documents/MEng Individual Project/Julia/AerofoilOptimisation/foil_dats/NACA0012.dat",
        py_path = "/home/tim/Documents/MEng Individual Project/Julia/AerofoilOptimisation/foil_pythons/NACAMesh.py", #Path to SALOME python script
        salome_path = "/home/tim/Downloads/InstallationFiles/SALOME-9.11.0/mesa_salome", #Path to SALOME installation
        unv_path = "/home/tim/Documents/MEng Individual Project/Julia/FVM_1D_TW/unv_sample_meshes/NACAMesh.unv", #Path to .unv destination
        note_path = "/home/tim/Documents/MEng Individual Project/SALOME", #Path to SALOME notebook (.hdf) destination
        GUI = false #SALOME GUI selector
    )
    mesh_file = "unv_sample_meshes/NACAMesh.unv"
    mesh = build_mesh(mesh_file, scale=0.001)
    mesh = update_mesh_format(mesh)

    model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

    # Boundary Conditions
    noSlip = [0.0, 0.0, 0.0]

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
        U = set_schemes(time=Euler,divergence=Upwind,gradient=Midpoint),
        p = set_schemes(time=Euler,divergence=Upwind),
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
        iterations=1000, write_interval=50, time_step=0.005)

    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime)

    GC.gc()

    initialise!(model.U, velocity)
    initialise!(model.p, 0.0)

    Rx, Ry, Rp = simple!(model, config) #, pref=0.0)

    #%% POST-PROCESSING
    aero_eff[i] = lift_to_drag(:foil, ρ, model)
    C_l[i],C_d[i] = aero_coeffs(:foil, chord, ρ, velocity, model)
    yplus,y = y_plus(:foil,ρ,model)
    let
        plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
        plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
        plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
        plot!(1:length(Rp), Rp, yscale=:log10, label="p")
    end
    vtk_files = filter(x->endswith(x,".vtk"), readdir("vtk_results/"))
    for file ∈ vtk_files
        filepath = "vtk_results/"*file
        dest = "vtk_loop/Re=10k (laminar model)/laminar$(i-1)_"*file
        mv(filepath, dest)
    end
end
paraview_vis(paraview_path = "paraview", #Path to paraview
             vtk_path = "/home/tim/Documents/MEng Individual Project/Julia/FVM_1D_TW/vtk_results/iteration_..vtk") #Path to vtk files
vtk_files = filter(x->endswith(x,".vtk"), readdir("vtk_results/"))
for file ∈ vtk_files
    filepath = "vtk_results/"*file
    dest = "vtk_loop/Re=10k (laminar model)/laminar$(6)_"*file
    mv(filepath, dest)
end