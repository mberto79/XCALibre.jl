export piso!

"""
    cpiso!(model, config; 
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0)

Incompressible and transient variant of the SIMPLE algorithm to solving coupled momentum and mass conservation equations. 

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
function piso!(
    model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=2)

    residuals = setup_incompressible_solvers(
        PISO, model, config; 
        output=output,
        pref=pref,
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )
        
    return residuals
end

function PISO(
    model, turbulenceModel, ∇p, U_eqn, p_eqn, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=2
    )
    
    # Extract model variables and configuration
    (; U, p, Uf, pf) = model.momentum
    (; nu) = model.fluid
    mesh = model.domain
    (; solvers, schemes, runtime, hardware) = config
    (; iterations, write_interval, dt) = runtime
    (; backend) = hardware
    
    mdotf = get_flux(U_eqn, 2)
    nueff = get_flux(U_eqn, 3)
    rDf = get_flux(p_eqn, 1)
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
    pf = FaceScalarField(mesh)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    TI = _get_int(mesh)
    prev = zeros(TF, n_cells)
    prev = _convert_array!(prev, backend) 

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
    limit_gradient!(schemes.p.limiter, ∇p, p, config)

    update_nueff!(nueff, nu, model.turbulence, config)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    @info "Starting PISO loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations
        time = iteration *dt

        rx, ry, rz = solve_equation!(
            U_eqn, U, solvers.U, xdir, ydir, zdir, config; time=time)
          
        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        remove_pressure_source!(U_eqn, ∇p, config)
        
        rp = 0.0
        for i ∈ 1:inner_loops
            H!(Hv, U, U_eqn, config)
            
            # Interpolate faces
            interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
            correct_boundaries!(Uf, Hv, U.BCs, time, config)
            # div!(divHv, Uf, config)

            # new approach
            flux!(mdotf, Uf, config)
            div!(divHv, mdotf, config)
            
            # Pressure calculations (previous implementation)
            @. prev = p.values
            rp = solve_equation!(p_eqn, p, solvers.p, config; ref=pref, time=time)
            if i == inner_loops
                explicit_relaxation!(p, prev, 1.0, config)
            else
                explicit_relaxation!(p, prev, solvers.p.relax, config)
            end

            grad!(∇p, pf, p, p.BCs, time, config) 
            limit_gradient!(schemes.p.limiter, ∇p, p, config)

            # nonorthogonal correction (experimental)
            for i ∈ 1:ncorrectors
                discretise!(p_eqn, p, config)       
                apply_boundary_conditions!(p_eqn, p.BCs, nothing, time, config)
                setReference!(p_eqn, pref, 1, config)
                nonorthogonal_face_correction(p_eqn, ∇p, rDf, config)
                update_preconditioner!(p_eqn.preconditioner, p.mesh, config)
                rp = solve_system!(p_eqn, solvers.p, p, nothing, config)

                if i == ncorrectors
                    explicit_relaxation!(p, prev, 1.0, config)
                else
                    explicit_relaxation!(p, prev, solvers.p.relax, config)
                end
                grad!(∇p, pf, p, p.BCs, time, config) 
                limit_gradient!(schemes.p.limiter, ∇p, p, config)
            end

            # old approach - keep for now!
            # correct_velocity!(U, Hv, ∇p, rD, config)
            # interpolate!(Uf, U, config)
            # correct_boundaries!(Uf, U, U.BCs, time, config)
            # flux!(mdotf, Uf, config) # old approach

            # new approach
            interpolate!(Uf, U, config) # velocity from momentum equation
            correct_boundaries!(Uf, U, U.BCs, time, config)
            flux!(mdotf, Uf, config)
            correct_mass_flux(mdotf, p, rDf, config)
            correct_velocity!(U, Hv, ∇p, rD, config)

        end # corrector loop end
        
        # correct_mass_flux(mdotf, p, rDf, config) # new approach

    turbulence!(turbulenceModel, model, S, prev, time, config) 
    update_nueff!(nueff, nu, model.turbulence, config)

    # if typeof(mesh) <: Mesh3
    #     residual!(R_uz, U_eqn, U.z, iteration, zdir, config)
    # end
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
            turbulenceModel.state.residuals...
            ]
        )

    if iteration%write_interval + signbit(write_interval) == 0
        save_output(model, outputWriter, time)
    end

    end # end for loop

    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p)
end