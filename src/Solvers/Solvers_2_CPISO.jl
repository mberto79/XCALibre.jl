export cpiso!

"""
    cpiso!(model, config; 
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0)

Compressible and transient variant of the PISO algorithm with a sensible enthalpy transport equation for the energy. 

# Input arguments

- `model` reference to a `Physics` model defined by the user.
- `config` Configuration structure defined by the user with solvers, schemes, runtime and hardware structures configuration details.
- `output` select the format used for simulation results from `VTK()` or `OpenFOAM` (default = `VTK()`)
- `pref` Reference pressure value for cases that do not have a pressure defining BC. Incompressible solvers only (default = `nothing`)
- `ncorrectors` number of non-orthogonality correction loops (default = `0`)
- `inner_loops` number to inner loops used in transient solver based on PISO algorithm (default = `0`)

# Output

- `Ux` Vector of x-velocity residuals for each iteration.
- `Uy` Vector of y-velocity residuals for each iteration.
- `Uz` Vector of y-velocity residuals for each iteration.
- `p` Vector of pressure residuals for each iteration.
"""
function cpiso!(
    model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=2) 

    residuals = setup_unsteady_compressible_solvers(
        CPISO, model, config; 
        output=output,
        pref=pref,
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )
        
    return residuals
end

# Setup for all compressible algorithms
function setup_unsteady_compressible_solvers(
    solver_variant, model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=2
    ) 

    (; solvers, schemes, runtime, hardware) = config

    @info "Extracting configuration and input fields..."

    (; U, p, Uf, pf) = model.momentum
    (; rho) = model.fluid
    mesh = model.domain

    @info "Pre-allocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    rhorDf = FaceScalarField(mesh)
    initialise!(rhorDf, 1.0)
    mueff = FaceScalarField(mesh)
    mueffgradUt = VectorField(mesh)
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
        - Source(∇p.result)
        + Source(mueffgradUt)
    ) → VectorEquation(U)

    if typeof(model.fluid) <: WeaklyCompressible
        
        p_eqn = (
            Time{schemes.p.time}(psi, p)  # correction(fvm::ddt(p)) means d(p)/d(t) - d(pold)/d(t)
            - Laplacian{schemes.p.laplacian}(rhorDf, p)
            ==
            - Source(divHv)
        ) → ScalarEquation(p)

    elseif typeof(model.fluid) <: Compressible

        pconv = FaceScalarField(mesh)

        p_eqn = (
            Time{schemes.p.time}(psi, p)
            - Laplacian{schemes.p.laplacian}(rhorDf, p) 
            + Divergence{schemes.p.divergence}(pconv, p)
            ==
            -Source(divHv)
            -Source(ddtrho) # capture correction part of dPdT and explicit drhodt
        ) → ScalarEquation(p)
    end

    @info "Initialising preconditioners..."

    @reset U_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, U_eqn, U.BCs, config)
    @reset p_eqn.preconditioner = set_preconditioner(
                    solvers.p.preconditioner, p_eqn, p.BCs, config)


    @info "Pre-allocating solvers..."
     
    @reset U_eqn.solver = solvers.U.solver(_A(U_eqn), _b(U_eqn, XDir()))
    @reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))

  
    @info "Initialising energy model..."
    energyModel = initialise(model.energy, model, mdotf, rho, p_eqn, config)

    @info "Initialising turbulence model..."
    turbulenceModel = initialise(model.turbulence, model, mdotf, p_eqn, config)

    residuals  = solver_variant(
        model, turbulenceModel, energyModel, ∇p, U_eqn, p_eqn, config;
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops)

    return residuals    
end # end function

function CPISO(
    model, turbulenceModel, energyModel, ∇p, U_eqn, p_eqn, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=2)
    
    # Extract model variables and configuration
    (; U, p, Uf, pf) = model.momentum
    (; rho, rhof, nu) = model.fluid
    (; dpdt) = model.energy
    mesh = model.domain
    p_model = p_eqn.model
    (; solvers, schemes, runtime, hardware) = config
    (; iterations, write_interval, dt) = runtime
    (; backend) = hardware
    
    mdotf = get_flux(U_eqn, 2)
    mueff = get_flux(U_eqn, 3)
    mueffgradUt = get_source(U_eqn, 2)
    rhorDf = get_flux(p_eqn, 2)
    divHv = get_source(p_eqn, 1)

    @info "Initialise writer (Store mesh in host memory)"

    outputWriter = initialise_writer(output, model.domain)
    
    @info "Allocating working memory..."

    # Define aux fields 
    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    Uf = FaceVectorField(mesh)
    S = StrainRate(gradU, gradUT, U, Uf)

    n_cells = length(mesh.cells)
    n_faces = length(mesh.faces)
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

    limit_gradient!(schemes.p.limiter, ∇p, p, config)

    update_nueff!(nueff, nu, model.turbulence, config)
    @. mueff.values = rhof.values*nueff.values

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    @info "Starting CPISO loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    for iteration ∈ 1:iterations
        time = iteration *dt

        ## CHECK GRADU AND EXPLICIT STRESSES
        # grad!(gradU, Uf, U, U.BCs, time, config) # calculated in `turbulence!`
        
        explicit_shear_stress!(mugradUTx, mugradUTy, mugradUTz, mueff, gradU, config)
        div!(divmugradUTx, mugradUTx, config)
        div!(divmugradUTy, mugradUTy, config)
        div!(divmugradUTz, mugradUTz, config)
        
        @. mueffgradUt.x.values = divmugradUTx.values
        @. mueffgradUt.y.values = divmugradUTy.values
        @. mueffgradUt.z.values = divmugradUTz.values
        
        # Set up and solve momentum equations

        rx, ry, rz = solve_equation!(U_eqn, U, solvers.U, xdir, ydir, zdir, config)

        energy!(energyModel, model, prev, mdotf, rho, mueff, time, config)
        thermo_Psi!(model, Psi); thermo_Psi!(model, Psif, config);

        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rhorDf, rD, config)
        @. rhorDf.values *= rhof.values

        remove_pressure_source!(U_eqn, ∇p, config)
        
        rp = 0.0
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
            rp = solve_equation!(p_eqn, p, solvers.p, config; ref=nothing)
            explicit_relaxation!(p, prev, solvers.p.relax, config)

            # Gradient
            grad!(∇p, pf, p, p.BCs, time, config) 
            limit_gradient!(schemes.p.limiter, ∇p, p, config)

            # non-orthogonal correction
            for i ∈ 1:ncorrectors
                discretise!(p_eqn, p, config)       
                apply_boundary_conditions!(p_eqn, p.BCs, nothing, time, config)
                setReference!(p_eqn, pref, 1, config)
                nonorthogonal_face_correction(p_eqn, ∇p, rhorDf, config)
                update_preconditioner!(p_eqn.preconditioner, p.mesh, config)
                rp = solve_system!(p_eqn, solvers.p, p, nothing, config)
                explicit_relaxation!(p, prev, solvers.p.relax, config)
                grad!(∇p, pf, p, p.BCs, time, config) 
                limit_gradient!(schemes.p.limiter, ∇p, p, config)
            end

            if ~isempty(solvers.p.limit)
                pmin = solvers.p.limit[1]; pmax = solvers.p.limit[2]
                clamp!(p.values, pmin, pmax)
            end

            # pgrad = face_normal_gradient(p, pf)
        
            if typeof(model.fluid) <: Compressible
                # @. mdotf.values += (pconv.values*(pf.values) - pgrad.values*rhorDf.values)  
                correct_mass_flux(mdotf, p, rhorDf, config)
                @. mdotf.values += pconv.values*(pf.values)
            elseif typeof(model.fluid) <: WeaklyCompressible
                # @. mdotf.values -= pgrad.values*rhorDf.values
                correct_mass_flux(mdotf, p, rhorDf, config)
            end
   
            # TO-DO: this needs to be exposed to users eventually
            @. rho.values = max.(Psi.values * p.values, 0.001)
            @. rhof.values = max.(Psif.values * pf.values, 0.001)

            # Velocity and boundaries correction
            correct_velocity!(U, Hv, ∇p, rD, config)
            interpolate!(Uf, U, config)
            correct_boundaries!(Uf, U, U.BCs, time, config)
            
            @. dpdt.values = (p.values-prev)/runtime.dt

            turbulence!(turbulenceModel, model, S, prev, time, config) 
            update_nueff!(nueff, nu, model.turbulence, config)
        end # corrector loop end

    maxCourant = max_courant_number!(cellsCourant, model, config)

    R_ux[iteration] = rx
    R_uy[iteration] = ry
    R_uz[iteration] = rz
    R_p[iteration] = rp

    ProgressMeter.next!(
        progress, showvalues = [
            (:time, iteration*runtime.dt),
            (:Courant, maxCourant),
            (:Ux, R_ux[iteration]),
            (:Uy, R_uy[iteration]),
            (:Uz, R_uz[iteration]),
            (:p, R_p[iteration]),
            turbulenceModel.state.residuals...,
            energyModel.state.residuals
            ]
        )

    if iteration%write_interval + signbit(write_interval) == 0
        save_output(model, outputWriter, time)
    end

    end # end for loop

    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p)
end