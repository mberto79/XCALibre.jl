using Plots, FVM_1D, Krylov, AerofoilOptimisation
using BayesianOptimization, GaussianProcesses, Distributions

function foil_optim(y::Vector{Float64})
    println(y)
    #%% AEROFOIL GEOMETRY DEFINITION
    foil,ctrl_p = spline_foil(FoilDef(
        chord   = 100, #[mm]
        LE_h    = 0, #[%c, at α=0°]
        TE_h    = 0, #[%c, at α=0°]
        peak    = [y[1],y[2]], #[%c]
        trough  = [y[3],-y[4]], #[%c]
        xover = y[5], #[%c]
        α = 5 #[°]
    )) #Returns aerofoil MCL & control point vector (spline method)

    #%% REYNOLDS & Y+ CALCULATIONS
    chord = 100.0
    Re = 80000
    nu,ρ = 1.48e-5,1.225
    yplus_init,BL_layers = 2.0,50
    laminar = false
    velocity,BL_mesh = BL_calcs(Re,nu,ρ,chord,yplus_init,BL_layers,laminar) #Returns (BL mesh thickness, BL mesh growth rate)

    #%% AEROFOIL MESHING
    lines = update_mesh(
        chord = foil.chord, #[mm]
        ctrl_p = ctrl_p, #Control point vector
        vol_size = (16,10), #Total fluid volume size (x,y) in chord multiples [aerofoil located in the vertical centre at the 1/3 position horizontally]
        thickness = 1, #Aerofoil thickness [%c]
        BL_thick = BL_mesh[1], #Boundary layer mesh thickness [mm]
        BL_layers = BL_layers, #Boundary layer mesh layers [-]
        BL_stretch = BL_mesh[2], #Boundary layer stretch factor (successive multiplication factor of cell thickness away from wall cell) [-]
        py_lines = (13,44,51,59,36,68,247,284), #SALOME python script relevant lines (notebook path, 3 B-Spline lines,chord line, thickness line, BL line .unv path)
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
    νR = 10
    Tu = 0.025
    k_inlet = 3/2*(Tu*velocity[1])^2
    ω_inlet = k_inlet/(νR*nu)
    model = RANS{KOmega}(mesh=mesh, viscosity=ConstantScalar(nu))

    # Boundary Conditions
    noSlip = [0.0, 0.0, 0.0]

    @assign! model U ( 
        FVM_1D.FVM_1D.Dirichlet(:inlet, velocity),
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
        iterations=1000, write_interval=1000, time_step=1)

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
    C_l,C_d = aero_coeffs(:foil, chord, ρ, velocity, model)
    aero_eff = lift_to_drag(:foil, ρ, model)

    if isnan(aero_eff)
        aero_eff = 0
    end
    
    let
        p = plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
        plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
        plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
        plot!(1:length(Rp), Rp, yscale=:log10, label="p")
        display(p)
    end
    vtk_files = filter(x->endswith(x,".vtk"), readdir("vtk_results/"))
    for file ∈ vtk_files
        filepath = "vtk_results/"*file
        dest = "vtk_loop/Optimisation (Re=80k,k-o,5 var,5 AoA)/$(aero_eff)"*file
        mv(filepath, dest,force=true)
    end
    return aero_eff
end
model = ElasticGPE(5,                            # 2 input dimensions
                   mean = MeanConst(0.0),         
                   kernel = SEArd([0.0,0.0,0.0,0.0,0.0], 5.0),
                   capacity = 3000)              # the initial capacity of the GP is 3000 samples.
set_priors!(model.mean, [Normal(1, 2)])

modeloptimizer = MAPGPOptimizer(every = 10,       
                                maxeval = 40)

#STORAGE OF COMPUTED VALUES
#2300hrs 07/03 storage
x_vals_2var = [11.08938933397953 15.48478524358638 12.164002984073132 12.665267317724187 21.26068771860812 1.0 15.34740845994921 1.0 9.42885874978036 2.653259572153325 30.0 4.267729125036729 1.0 1.0 15.786762146780873 30.0 2.737493768136838 19.683572162740344 1.0 30.0 30.0 9.820257249439429 11.673746894446964 26.370495531304375 9.506574498660447 19.614447203812354 23.359331007436392 14.526194474816485 7.874144974523727 5.1273253647317425 5.158061558328708 3.777361361168503 14.805483677559673 25.38145327836131 7.27134740125583 3.814643337577621 3.817745360018021 7.386916377159381 3.8177116209126263 5.4689482993908705 3.8177309533844097 6.944983285733449 5.907533032980499 22.698796100917292 10.679562425058732 8.198202442543723 3.8282901913616167 8.27302852625749 12.365596846815482 3.811898651545394 12.830994420292395; 26.06594970426061 21.331303626714156 24.90839448225568 26.537190362714103 14.746335375614578 1.0 24.564036677111638 17.622546497028218 1.0 1.0 25.35950065789937 4.245708413936353 30.0 1.0 27.827264459813676 1.0 23.259636774424752 7.190881749756314 11.199431193728273 19.065928433441623 10.330779652929618 15.038064383513078 11.2039025271804 30.0 18.616053221186974 1.0 21.892195186099517 15.369866610835304 30.0 1.0 14.671291693458903 1.0 2.406499206419488 6.862059691266949 3.2203207577392394 1.0 1.0 22.481362025274223 1.0 26.554820836314928 1.0 11.00964175066907 18.486474202587498 26.93671100363466 21.326905624162503 13.976892280854305 1.0 16.431639017531314 6.790357502460945 1.0 30.0]
y_vals_2var = [5.237414628100506, 3.5703363658222647, 4.970811997633325, 4.289373133638538, 0.787067699901464, 9.55434224325726, 3.17, 3.323, 5.945, 10.80000677639661, 0.622, 4.267, 2.188, 9.554, 2.738175907960623, 0.0747785746782299, 3.8530104100151146, 0.7702909925201151, 2.074455808807779, 0.058414465505498336, -0.09805447703354525, 6.027244507415013, 3.903059865115453, 0.75318843484064, 6.095487481230004, 1.3534367383145045, 0.9585882800209506, 3.0633350932189476, 3.2116405372735892, 10.617440022682384, 5.329122504203392, 11.4909115412491, 2.1746956893396368, 0.16561338917773344, 5.26112380967508, 11.544570483172807, 11.55438194511509, 5.5446278551129415, 11.55415271974972, 4.438819947934602, 11.607213603173452, 4.710621171250421, 5.355218569819767, 0.9179811619637491, 5.653514861410564, 6.154540244727485, 11.526923763536402, 6.278113560560658, 2.736036208435878, 11.55088205030062, 3.823900732054509]
x_vals_5var = [34.27399476038774 27.09399629096962 15.02203717760744 24.440556047153468 32.0425571757026 21.923316707110597; 12.093789095480892 2.221730344248709 29.297892959183134 25.724331874754874 10.57479268908467 25.770020240213874; 71.63711380373994 71.01434587748858 64.50764269875424 66.32639358203309 77.8937163981997 72.25893450824535; 15.494539864786482 7.397608009286941 23.41189698805543 28.907572496153296 15.810945482801325 28.851680124760474; 69.0067752013108 34.78496497415903 31.62581708936928 46.02348064693361 71.54933655329228 60.17862117056916]
y_vals_5var = [6.593788718321878, -1.1094654377098172, -0.6175620177483727, 0.4223174105222193, 7.827494504883097, 1.4009290330364237]
append!(model,x_vals_5var,y_vals_5var)

opt = BOpt(foil_optim,
           model,
           UpperConfidenceBound(),
           modeloptimizer,                        
           [10.0,1.0,60.0,1.0,20.0], [40.0,30.0,90.0,30.0,80.0],       
           repetitions = 1,
           maxiterations = 60,
           sense = Max,
           initializer_iterations = 0,   
            verbosity = Progress)

result = boptimize!(opt)