export piso!

"""
    cpiso!(model; 
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0)

Incompressible and transient variant of the SIMPLE algorithm to solving coupled momentum and mass conservation equations. 

# Input arguments

- `model` reference to a `Physics` model defined by the user.
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
    model; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=2)

    residuals = setup_incompressible_solvers(
        PISO, model; 
        output=output,
        pref=pref,
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )
        
    return residuals
end

function PISO(
    model, turbulenceModel, ∇p, U_eqn, p_eqn; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=2
    )
    
    # Extract model variables and configuration
    (; U, p, Uf, pf) = model.momentum
    (; nu) = model.fluid
    mesh = model.domain
    (; solvers, schemes, runtime, hardware, boundaries) = get_configuration(CONFIG)
    (; iterations, write_interval, dt) = runtime
    (; backend) = hardware
    
    mdotf = get_flux(U_eqn, 2)
    nueff = get_flux(U_eqn, 3)
    rDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)

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
    # prev = zeros(TF, n_cells)
    # prev = _convert_array!(prev, backend) 
    prev = KernelAbstractions.zeros(backend, TF, n_cells)

    # Pre-allocate vectors to hold residuals 
    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_uz = ones(TF, iterations)
    R_p = ones(TF, iterations)
    cellsCourant =adapt(backend, zeros(TF, length(mesh.cells)))
    
    # Initial calculations
    time = zero(TF) # assuming time=0
    interpolate!(Uf, U)   
    correct_boundaries!(Uf, U, boundaries.U, time)
    flux!(mdotf, Uf)
    grad!(∇p, pf, p, boundaries.p, time)
    limit_gradient!(schemes.p.limiter, ∇p, p)

    update_nueff!(nueff, nu, model.turbulence)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    @info "Starting PISO loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations
        time = iteration *dt

        rx, ry, rz = solve_equation!(
            U_eqn, U, boundaries.U, solvers.U, xdir, ydir, zdir; time=time)
          
        # Pressure correction
        inverse_diagonal!(rD, U_eqn)
        interpolate!(rDf, rD)
        remove_pressure_source!(U_eqn, ∇p)
        
        rp = 0.0
        for i ∈ 1:inner_loops
            H!(Hv, U, U_eqn)
            
            # Interpolate faces
            interpolate!(Uf, Hv) # Careful: reusing Uf for interpolation
            correct_boundaries!(Uf, Hv, boundaries.U, time)
            # div!(divHv, Uf)

            # new approach
            flux!(mdotf, Uf)
            div!(divHv, mdotf)
            
            # Pressure calculations (previous implementation)
            @. prev = p.values
            rp = solve_equation!(p_eqn, p, boundaries.p, solvers.p; ref=pref, time=time)
            if i == inner_loops
                explicit_relaxation!(p, prev, 1.0)
            else
                explicit_relaxation!(p, prev, solvers.p.relax)
            end

            grad!(∇p, pf, p, boundaries.p, time) 
            limit_gradient!(schemes.p.limiter, ∇p, p)

            # nonorthogonal correction (experimental)
            for i ∈ 1:ncorrectors
                discretise!(p_eqn, p)       
                apply_boundary_conditions!(p_eqn, boundaries.p, nothing, time)
                setReference!(p_eqn, pref, 1)
                nonorthogonal_face_correction(p_eqn, ∇p, rDf)
                update_preconditioner!(p_eqn.preconditioner, p.mesh)
                rp = solve_system!(p_eqn, solvers.p, p, nothing)

                if i == ncorrectors
                    explicit_relaxation!(p, prev, 1.0)
                else
                    explicit_relaxation!(p, prev, solvers.p.relax)
                end
                grad!(∇p, pf, p, boundaries.p, time) 
                limit_gradient!(schemes.p.limiter, ∇p, p)
            end

            # old approach - keep for now!
            # correct_velocity!(U, Hv, ∇p, rD)
            # interpolate!(Uf, U)
            # correct_boundaries!(Uf, U, boundaries.U, time)
            # flux!(mdotf, Uf) # old approach

            # new approach
            interpolate!(Uf, U) # velocity from momentum equation
            correct_boundaries!(Uf, U, boundaries.U, time)
            flux!(mdotf, Uf)
            correct_mass_flux(mdotf, p, rDf)
            correct_velocity!(U, Hv, ∇p, rD)

        end # corrector loop end
        
        # correct_mass_flux(mdotf, p, rDf) # new approach

    turbulence!(turbulenceModel, model, S, prev, time) 
    update_nueff!(nueff, nu, model.turbulence)

    # if typeof(mesh) <: Mesh3
    #     residual!(R_uz, U_eqn, U.z, iteration, zdir)
    # end
    maxCourant = max_courant_number!(cellsCourant, model)

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
        save_output(model, outputWriter, iteration, time)
    end

    end # end for loop

    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p)
end