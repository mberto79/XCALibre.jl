using Plots, FVM_1D, Krylov, AerofoilOptimisation
using BayesianOptimization, GaussianProcesses, Distributions

function foil_optim(y::Vector{Float64})
    println(y)
    #%% AEROFOIL GEOMETRY DEFINITION
    foil,ctrl_p = spline_foil(FoilDef(
        chord   = 100, #[mm]
        LE_h    = 0, #[%c, at α=0°]
        TE_h    = 0, #[%c, at α=0°]
        peak    = [25,y[1]], #[%c]
        trough  = [75,-y[2]], #[%c]
        xover = y[3], #[%c]
        α = 0 #[°]
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
        dest = "vtk_loop/Optimisation (Re=80k,k-o,3 var,0 AoA)/$(aero_eff)_$(y[1])_$(y[2])_$(y[3])"*file
        mv(filepath, dest)
    end
    return aero_eff
end
model = ElasticGPE(3,                            # 2 input dimensions
                   mean = MeanConst(0.0),         
                   kernel = SEArd([0.0,0.0,0.0], 5.0),
                   capacity = 3000)              # the initial capacity of the GP is 3000 samples.
set_priors!(model.mean, [Normal(1, 2)])

modeloptimizer = MAPGPOptimizer(every = 10,       
                                maxeval = 40)

#STORAGE OF COMPUTED VALUES
#2300hrs 07/03 storage
x_vals_2var = [7.90625 27.65625 37.53125 17.78125 12.84375 32.59375 22.71875 2.96875 4.203125 23.953125 0.5 16.112870099295126 33.46937876554953 7.90625 27.65625 37.53125 17.78125 12.84375 32.59375 22.71875 2.96875 4.203125 23.953125 15.010205826656271 15.830122468845381 17.709843847742608 17.693108068662305 12.441712830441778 20.381379421813556 14.83461375903239 12.437768692227849 14.45473529481564 9.711325455944475 20.84864205902731 26.41633713957935 1.0 40.0 31.216423971289505 40.0 40.0 1.0 1.0 8.580312812074714 22.103821598380538 7.8172242166187385 27.716875068501537 29.961586645520985 17.567184170687568 34.78244008040918 31.86142095097858 40.0 13.801489776117927 19.666684919966045 26.59770659610416 10.770083323245435 5.4743563081070175 5.432955437689993 35.403438464631165 35.39378119835325 24.558195298193905 20.308227438232 7.794013730882644 12.299429720520958 36.591437842038125 7.8851200570649365 13.193285635689916 16.699649168182926 3.6239275434276603 37.76088535115332 31.042294707993324 1.0 22.07933248925248 26.727653903641514 9.83243335994195 29.088371578562594 40.0 11.799179344120184 26.419631745886253 9.795552898008784; 12.84375 32.59375 2.96875 22.71875 7.90625 27.65625 17.78125 37.53125 19.015625 38.765625 34.92286502681552 6.989950515116923 33.82142295164507 12.84375 32.59375 2.96875 22.71875 7.90625 27.65625 17.78125 37.53125 19.015625 38.765625 35.74575148959419 38.329659721186225 36.29551115520995 25.009409583021228 37.97906967983588 28.459591432020265 26.22306056195864 23.192744455362153 18.861812783411008 26.565510403614283 24.09225718774762 1.0 1.0 18.180905005266517 13.572321867982039 40.0 29.173555822210368 11.463605763264548 26.28888821445399 1.0 8.256575199172486 34.176118317657604 22.450374936075757 40.0 14.488817433210494 20.29461933585465 3.498981358115585 10.066767164978334 1.0 1.0 11.31531559926058 17.8290050647118 6.501958287224373 27.62082242702505 11.152873854376223 40.0 28.031461590068773 40.0 40.0 31.025554770868858 26.345525746783846 21.262902649508707 13.64174703887929 30.430677570709687 32.02143798087051 34.31128065934597 20.279816643055902 19.49701598215768 33.724047582293394 17.102974595318457 7.102249102902149 27.645423730989965 23.671031660139395 34.48449733887374 30.996401981607946 33.67166990405061]
y_vals_2var = [1.2205801265750773, 0.9852896728958433, -0.9275936057855946, 2.8187764816234226, 2.154853195082591, 0.7607322271213195, 0.8133944237116522, 1.0707233254321726, 1.321805751477013, 1.2255568641349772, 1.5593437741124245, 1.0557608689281692, 0.45943949130426726, 1.2205801265750786, 0.9852896728958433, -0.9275936057855946, 2.8187764816234226, 2.154853195082591, 0.7607322271213195, 0.8133944237116522, 1.0707233254321726, 1.321805751477013, 1.2255568641349772, 2.3803608902434115, 2.3562707977989437, 1.7055569879810102, 3.5415788822143743, 2.2391038287845673, 1.6491635387008177, 4.348900360642778, 4.089583037763437, 4.219225560932352, 2.475931721858065, 1.4440296793558784, -0.8780148980707811, -3.317712256351981, -0.2926040434003621, -0.16676669710224595, 0.9042411651032827, 0.3505789253948884, -1.6982035224533447, 0.9523968353577309, 0.9673348015578179, -0.027514960612225883, 1.7303509961895327, 0.3820929312194527, 0.6615333168470316, 1.4428347766431768, 0.0837285125282743, -1.0278507405759556, -0.7473650148464844, 1.6191894386042227, 0.08006124531332544, -0.25281587586343185, 3.544778113321775, -5.909293937369014, 1.611863337386052, -0.4642985767629187, 0.561079899179818, 1.1162704429218777, 1.4849835672024414, 1.6363402542779004, 2.4699041979866316, 0.49013203175654085, 2.6415520309583465, 2.9974090125344732, 3.3313654247165183, 1.063432546797997, 0.3467354997018567, 0.35444189584271735, 0.8907152773597035, 1.2255058517674906, 0.16347224089915113, -0.9119627705637657, 0.866296677608575, 0.040258163506355374, 2.6499062369910416]
x_vals_3var = [13.794461952638912 37.617537464394985 15.312998425617169 11.032754852262391 26.969375694056968 10.881817023213868 22.223698589970205 37.63131759947868 25.2436404914678 29.267148605903934 36.749781243875006 37.37838154690229 33.46900299682019 28.19455600846492 38.84546044598028 23.866365953976246 16.92323296085368 37.230345968456 39.42349062601564 16.304584315982282 23.372221374365235 5.0132212660994675 20.48864386123906 4.481398047364506 26.03587486478962 5.215473225725318 16.689950414266256 11.274933421551113; 16.749653888918346 2.0115959890357207 7.916624221681229 28.651163311751155 5.292150583924166 32.51112034062899 36.4454853142256 25.643379657768346 30.572419844773187 14.376753596441178 8.55489854890834 10.197009207209131 21.74910060431163 34.39609826644195 15.407805169381746 3.042228287183998 39.43603333079567 14.539157550903742 38.25912942594137 5.632109873235206 12.140411852871178 34.5246951419256 2.55103539566358 8.746379733281405 23.957007576670946 13.57099934303202 17.939416203692126 34.90424230669304; 57.152739781873734 46.60621562565998 39.1723259631826 42.47946247473074 47.49706509162555 41.497906100791226 49.47840067700167 51.81734208194564 44.719694482393336 37.79254568444146 38.499480793068734 35.45790330853696 56.70996623752935 52.99560768572363 41.731326462997224 63.17761069394071 46.09889935691163 43.729240825869866 46.271821689270425 51.188717718254296 49.09236897237527 40.957523626776634 60.36716680914216 35.91294399521214 36.07904708028927 49.09301647202144 47.57450835861244 47.70962405888591]
y_vals_3var = [3.9734665411684764, -0.90749719888753, 1.2487170870747284, 4.005024027096534, -1.0395002714215507, 2.5436107977612603, 1.1190803867223063, 0.45110645530023363, 0.5390481276403276, -0.4601241684049206, -0.9716720827301466, -0.8967472035777866, 0.5040889973616791, 0.8313337041229713, -0.4743077788291949, -0.3141049303601756, 1.5926288651155938, -0.2588630522221985, -0.029111781774500987, 0.6856819404055373, 0.029363797651721927, 1.2260871684638046, 0.1781957096274071, -2.886117972948516, 0.05113562160245366, 0.5895386994764986, 2.246405787476819, 2.49387270768781]

opt = BOpt(foil_optim,
           model,
           UpperConfidenceBound(),
           modeloptimizer,                        
           [1.0,1.0,35], [40.0,40.0,65],       
           repetitions = 1,
           maxiterations = 60,
           sense = Max,
           initializer_iterations = 0,   
            verbosity = Progress)

result = boptimize!(opt)

#2300hrs 07/03 storage
#model.x = 
#model.y = [1.2205801265750773, 0.9852896728958433, -0.9275936057855946, 2.8187764816234226, 2.154853195082591, 0.7607322271213195, 0.8133944237116522, 1.0707233254321726, 1.321805751477013, 1.2255568641349772, 1.5593437741124245, 1.0557608689281692, 0.45943949130426726, 1.2205801265750786, 0.9852896728958433, -0.9275936057855946, 2.8187764816234226, 2.154853195082591, 0.7607322271213195, 0.8133944237116522, 1.0707233254321726, 1.321805751477013, 1.2255568641349772, 2.3803608902434115, 2.3562707977989437, 1.7055569879810102, 3.5415788822143743]