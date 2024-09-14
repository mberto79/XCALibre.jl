export cpiso!

"""
    cpiso!(model, config; resume=true, pref=nothing)

Compressible variant of the PISO algorithm with a sensible enthalpy transport equation for 
the energy. 

### Input
- `model` -- Physics model defiend by user and passed to run!.
- `config`   -- Configuration structure defined by user with solvers, schemes, runtime and 
                hardware structures set.
- `resume`   -- True or false indicating if case is resuming or starting a new simulation.
- `pref`     -- Reference pressure value for cases that do not have a pressure defining BC.

### Output
- `R_ux`  - Vector of x-velocity residuals for each iteration.
- `R_uy`  - Vector of y-velocity residuals for each iteration.
- `R_uz`  - Vector of y-velocity residuals for each iteration.
- `R_p`   - Vector of pressure residuals for each iteration.

"""
function cpiso!(
    model, config; 
    limit_gradient=false, pref=nothing, ncorrectors=0, inner_loops=2) 

    residuals = setup_unsteady_compressible_solvers(
        CPISO, model, config; 
        limit_gradient=limit_gradient, 
        pref=pref,
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )
        
    return residuals
end

# Setup for all compressible algorithms
function setup_unsteady_compressible_solvers(
    solver_variant, model, config; 
    limit_gradient=false, pref=nothing, ncorrectors=0, inner_loops=2
    ) 

    (; solvers, schemes, runtime, hardware) = config

    @info "Extracting configuration and input fields..."

    (; U, p) = model.momentum
    (; rho) = model.fluid
    mesh = model.domain

    @info "Pre-allocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    rhorDf = FaceScalarField(mesh)
    mueff = FaceScalarField(mesh)
    mueffgradUt = VectorField(mesh)
    # initialise!(rDf, 1.0)
    rhorDf.values .= 1.0
    divHv = ScalarField(mesh)
    ddtrho = ScalarField(mesh)
    psidpdt = ScalarField(mesh)
    divmdotf = ScalarField(mesh)
    psi = ScalarField(mesh)


    @info "Defining models..."

    # rho eqn doesn't work at the moment.
    # rho_eqn = (
    #     Time{schemes.rho.time}(rho) 
    #     == 
    #     -Source(divmdotf)
    # ) → ScalarEquation(mesh)

    U_eqn = (
        Time{schemes.U.time}(rho, U)
        + Divergence{schemes.U.divergence}(mdotf, U) 
        - Laplacian{schemes.U.laplacian}(mueff, U) 
        == 
        -Source(∇p.result)
        +Source(mueffgradUt)
    ) → VectorEquation(mesh)

    if typeof(model.fluid) <: WeaklyCompressible
        # p_eqn = (
        #     Time{schemes.p.time}(psi, p)  # correction(fvm::ddt(p)) means d(p)/d(t) - d(pold)/d(t)
        #     - Laplacian{schemes.p.laplacian}(rhorDf, p)
        #     ==
        #     -Source(divHv)
        #     -Source(ddtrho)
        #     +Source(psidpdt)
        # ) → ScalarEquation(mesh)
        p_eqn = (
            Time{schemes.p.time}(psi, p)  # correction(fvm::ddt(p)) means d(p)/d(t) - d(pold)/d(t)
            - Laplacian{schemes.p.laplacian}(rhorDf, p)
            ==
            -Source(divHv)
        ) → ScalarEquation(mesh)
    elseif typeof(model.fluid) <: Compressible
        pconv = FaceScalarField(mesh)
        p_eqn = (
            Time{schemes.p.time}(psi, p)
            - Laplacian{schemes.p.laplacian}(rhorDf, p) 
            + Divergence{schemes.p.divergence}(pconv, p)
            ==
            -Source(divHv)
            -Source(ddtrho) # Needs to capture the correction part of dPdT and the explicit drhodt
        ) → ScalarEquation(mesh)
    end

    @info "Initialising preconditioners..."

    @reset U_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, U_eqn, U.BCs, config)
    @reset p_eqn.preconditioner = set_preconditioner(
                    solvers.p.preconditioner, p_eqn, p.BCs, config)
    # @reset rho_eqn.preconditioner = set_preconditioner(
    #                 solvers.rho.preconditioner, rho_eqn, p.BCs, config)

    @info "Pre-allocating solvers..."
     
    @reset U_eqn.solver = solvers.U.solver(_A(U_eqn), _b(U_eqn, XDir()))
    @reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))
    # @reset rho_eqn.solver = solvers.rho.solver(_A(rho_eqn), _b(rho_eqn))
  
    @info "Initialising energy model..."
    energyModel = initialise(model.energy, model, mdotf, rho, p_eqn, config)

    @info "Initialising turbulence model..."
    turbulenceModel = initialise(model.turbulence, model, mdotf, p_eqn, config)

    residuals  = solver_variant(
        model, turbulenceModel, energyModel, ∇p, U_eqn, p_eqn, config; limit_gradient=limit_gradient, 
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops)

    return residuals    
end # end function

function CPISO(
    model, turbulenceModel, energyModel, ∇p, U_eqn, p_eqn, config; 
    limit_gradient=false, pref=nothing, ncorrectors=0, inner_loops=2)
    
    # Extract model variables and configuration
    (; U, p) = model.momentum
    (; rho, rhof, nu) = model.fluid
    (; dpdt) = model.energy
    mesh = model.domain
    p_model = p_eqn.model
    (; solvers, schemes, runtime, hardware) = config
    (; iterations, write_interval, dt) = runtime
    (; backend) = hardware
    
    # divmdotf = get_source(rho_eqn, 1)
    mdotf = get_flux(U_eqn, 2)
    mueff = get_flux(U_eqn, 3)
    mueffgradUt = get_source(U_eqn, 2)
    rhorDf = get_flux(p_eqn, 2)
    divHv = get_source(p_eqn, 1)
    # ddtrho = get_source(p_eqn, 2)
    # psidpdt = get_source(p_eqn, 3)

    @info "Initialise VTKWriter (Store mesh in host memory)"

    VTKMeshData = initialise_writer(model.domain)
    
    @info "Allocating working memory..."

    # Define aux fields 
    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT)
    S2 = ScalarField(mesh)

    n_cells = length(mesh.cells)
    n_faces = length(mesh.faces)
    Uf = FaceVectorField(mesh)
    pf = FaceScalarField(mesh)
    nueff = FaceScalarField(mesh)
    gradpf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)
    Psi = ScalarField(mesh)
    Psif = FaceScalarField(mesh)

    divmdotf = ScalarField(mesh)

    mugradUTx = FaceScalarField(mesh)
    mugradUTy = FaceScalarField(mesh)
    mugradUTz = FaceScalarField(mesh)

    divmugradUTx = ScalarField(mesh)
    divmugradUTy = ScalarField(mesh)
    divmugradUTz = ScalarField(mesh)

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)
    prev = _convert_array!(prev, backend) 

    corr = zeros(TF, n_faces)
    corr = _convert_array!(corr, backend) 

    # Pre-allocate vectors to hold residuals 
    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_uz = ones(TF, iterations)
    R_p = ones(TF, iterations)
    cellsCourant =adapt(backend, zeros(TF, length(mesh.cells)))

    
    # Initial calculations
    time = zero(TF) # assuming time=0
    interpolate!(Uf, U, config)   
    correct_boundaries!(Uf, U, U.BCs, time, config)
    flux!(mdotf, Uf, config)
    grad!(∇p, pf, p, p.BCs, time, config)
    thermo_Psi!(model, Psi); thermo_Psi!(model, Psif, config);
    @. rho.values = Psi.values * p.values
    @. rhof.values = Psif.values * pf.values
    flux!(mdotf, Uf, rhof, config)

    # grad limiter test
    limit_gradient!(∇p, p, config)

    update_nueff!(nueff, nu, model.turbulence, config)
    @. mueff.values = rhof.values*nueff.values

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    @info "Starting PISO loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    # volumes = getproperty.(mesh.cells, :volume)

    @time for iteration ∈ 1:iterations
        time = (iteration - 1)*dt

        # println("Max. CFL : ", maximum((U.x.values.^2+U.y.values.^2).^0.5*runtime.dt./volumes.^(1/3)))

        ## CHECK GRADU AND EXPLICIT STRESSES
        grad!(gradU, Uf, U, U.BCs, time, config)
        
        # Set up and solve momentum equations
        explicit_shear_stress!(mugradUTx, mugradUTy, mugradUTz, mueff, gradU)
        div!(divmugradUTx, mugradUTx, config)
        div!(divmugradUTy, mugradUTy, config)
        div!(divmugradUTz, mugradUTz, config)

        @. mueffgradUt.x.values = divmugradUTx.values
        @. mueffgradUt.y.values = divmugradUTy.values
        @. mueffgradUt.z.values = divmugradUTz.values

        solve_equation!(U_eqn, U, solvers.U, xdir, ydir, zdir, config)

        energy!(energyModel, model, prev, mdotf, rho, mueff, time, config)
        thermo_Psi!(model, Psi); thermo_Psi!(model, Psif, config);

        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rhorDf, rD, config)
        @. rhorDf.values *= rhof.values

        remove_pressure_source!(U_eqn, ∇p, config)
        
        for i ∈ 1:inner_loops
            H!(Hv, U, U_eqn, config)
            
            # Interpolate faces
            interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
            correct_boundaries!(Uf, Hv, U.BCs, time, config)

            if typeof(model.fluid) <: Compressible
                flux!(pconv, Uf, config)
                @. pconv.values *= Psif.values
                corr = 0.0#fvc::ddtCorr(rho, U, phi)
                flux!(mdotf, Uf, config)
                @. mdotf.values *= rhof.values
                @. mdotf.values += rhorDf.values*corr
                interpolate!(pf, p, config)
                correct_boundaries!(pf, p, p.BCs, time, config)
                @. mdotf.values -= mdotf.values*Psif.values*pf.values/rhof.values
                div!(divHv, mdotf, config)

            elseif typeof(model.fluid) <: WeaklyCompressible
                @. corr = mdotf.values
                flux!(mdotf, Uf, config)
                @. mdotf.values *= rhof.values
                @. corr -= mdotf.values
                @. corr *= 0.0/runtime.dt
                @. mdotf.values += rhorDf.values*corr/rhof.values
                div!(divHv, mdotf, config)
            end
            
            # Pressure calculations
            @. prev = p.values
            solve_equation!(p_eqn, p, solvers.p, config; ref=nothing)

            # Gradient
            grad!(∇p, pf, p, p.BCs, time, config) 

            # grad limiter test
            limit_gradient!(∇p, p, config)

            correct = false
            if correct
                ncorrectors = 1
                for i ∈ 1:ncorrectors
                    discretise!(p_eqn)
                    apply_boundary_conditions!(p_eqn, p.BCs)
                    setReference!(p_eqn.equation, pref, 1)
                    interpolate!(gradpf, ∇p, p)
                    nonorthogonal_flux!(pf, gradpf) # careful: using pf for flux (not interpolation)
                    correct!(p_eqn.equation, p_model.terms.term1, pf)
                    solve_equation!(p_eqn, p, solvers.p, config; ref=nothing)
                    grad!(∇p, pf, p, pBCs, time, config) 
                end
            end

            explicit_relaxation!(p, prev, solvers.p.relax, config)

            if ~isempty(solvers.p.limit)
                pmin = solvers.p.limit[1]; pmax = solvers.p.limit[2]
                clamp!(p.values, pmin, pmax)
            end

            pgrad = face_normal_gradient(p, pf)
        
            if typeof(model.fluid) <: Compressible
                @. mdotf.values += (pconv.values*(pf.values) - pgrad.values*rhorDf.values)  
            elseif typeof(model.fluid) <: WeaklyCompressible
                @. mdotf.values -= pgrad.values*rhorDf.values
            end
   
            @. rho.values = max.(Psi.values * p.values, 0.001)
            @. rhof.values = max.(Psif.values * pf.values, 0.001)

            # Velocity and boundaries correction
            correct_velocity!(U, Hv, ∇p, rD, config)
            interpolate!(Uf, U, config)
            correct_boundaries!(Uf, U, U.BCs, time, config)
            
            @. dpdt.values = (p.values-prev)/runtime.dt

            grad!(gradU, Uf, U, U.BCs, time, config)
            turbulence!(turbulenceModel, model, S, S2, prev, time, config) 
            update_nueff!(nueff, nu, model.turbulence, config)
        end # corrector loop end

    residual!(R_ux, U_eqn, U.x, iteration, xdir, config)
    residual!(R_uy, U_eqn, U.y, iteration, ydir, config)
    if typeof(mesh) <: Mesh3
        residual!(R_uz, U_eqn, U.z, iteration, zdir, config)
    end
    residual!(R_p, p_eqn, p, iteration, nothing, config)
    maxCourant = max_courant_number!(cellsCourant, model, config)

    ProgressMeter.next!(
        progress, showvalues = [
            (:time, iteration*runtime.dt),
            (:Courant, maxCourant),
            (:Ux, R_ux[iteration]),
            (:Uy, R_uy[iteration]),
            (:Uz, R_uz[iteration]),
            (:p, R_p[iteration]),
            ]
        )

    if iteration%write_interval + signbit(write_interval) == 0
        model2vtk(model, VTKMeshData, @sprintf "timestep_%.6d" iteration)
    end

    end # end for loop

    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p)
end