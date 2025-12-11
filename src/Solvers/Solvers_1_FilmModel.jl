export filmModel!

function filmModel!(
    model, config;
    output=VTK()#, pref=nothing, ncorrectors=0, inner_loops=0
)
    print("Using film model\n")
    residuals = setup_FilmModel_Solver(
        FilmModel, model, config,
        output=output
    )
    
    return residuals
end

function setup_FilmModel_Solver(solver_variant, model, config;
    output=VTK())

    (; solvers, schemes, runtime, hardware, boundaries) = config

    @info "Extracting configuration and input fields..."

    (; U, h, Uf, hf) = model.momentum
    mesh = model.domain
    (; rho) = model.fluid
    rho_l = rho

    @info "Pre-allocating fields..."
    ∇h = Grad{schemes.h.gradient}(h)
    mdotf = FaceScalarField(mesh)
    Sm = ScalarField(mesh)
    h_prev = ScalarField(mesh)
    Si_mom = ScalarField(mesh)
    nueff = FaceScalarField(mesh)

    @info "Defining models.."

    # Edit
    @info "U equation still need updating"
    U_eqn = (
        Time{schemes.U.time}(h, U)
        + Divergence{schemes.U.divergence}(mdotf,U)
        #+ Grd{schemes.h.gradient}(h, )
        #+ Si(nueff, U)
        ==
        Source()
    ) → VectorEquation(U, boundaries.U)

    h_eqn = (
        Time{schemes.h.time}(h)
        + Divergence{schemes.h.divergence}(mdotf, h)
        ==
        Source(Sm)#/rho_l)
    ) → ScalarEquation(h, boundaries.h)

    @info "Initialising preconditioners"

    @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    @reset h_eqn.preconditioner = set_preconditioner(solvers.h.preconditioner, h_eqn)

    @info "Pre-allocating solvers"

    @reset U_eqn.solver = _workspace(solvers.U.solver, _b(U_eqn, XDir()))
    @reset h_eqn.solver = _workspace(solvers.h.solver, _b(h_eqn))

    @info "No turbulence model for now"
    p_eqn = (Time{schemes.h.time}(rho_l,h)==Source(Sm)) → ScalarEquation(h, boundaries.h)
    turbulenceModel, config = initialise(model.turbulence, model, mdotf, p_eqn, config)

    residuals = solver_variant(
        model, turbulenceModel, ∇h, U_eqn, h_eqn, config
    )
end

function FilmModel(
    model, turbulenceModel, ∇h, U_eqn, h_eqn, config;
    output=VTK(), ncorrectors=0
)

    (; U, h, Uf, hf) = model.momentum
    (; nu) = model.fluid
    mesh = model.domain
    (; solvers, schemes, runtime, hardware, boundaries, postprocess) = config
    (; iterations, write_interval, dt) = runtime
    (; backend) = hardware

    Postprocess = convert_time_to_iterations(postprocess, model, dt, iterations)
    mdotf = get_flux(U_eqn, 2)
    #nueff = get_flux(U_eqn, 3)
    divHv = get_source(h_eqn, 1)
    
    outputWriter = initialise_writer(output, model.domain)

    @info "Allocating working memory"

    # Define aux fields
    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT, U, Uf)

    n_cells = length(mesh.cells)
    Hv = VectorField(mesh) #unsure on these 2
    rD = ScalarField(mesh)

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    prev = KernelAbstractions.zeros(backend, TF, n_cells)

    # Pre-allocate vectors to hold residuals
    R_ux = zeros(TF, iterations)
    R_uy = zeros(TF, iterations)
    R_uz = zeros(TF, iterations)
    R_h = zeros(TF, iterations)

    # Initial calculations
    time = zero(TF) # assuming time = 0
    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, boundaries.U, time, config)
    flux!(mdotf, Uf, config)
    grad!(∇h, hf, h, boundaries.h, time, config)
    limit_gradient!(schemes.h.limiter, ∇h, h, config)

    #update_nueff!(nueff, nu, turbulenceModel, config)

    @info "Starting loops"
    
    progress = Progress(iterations; dt=1.0, showspeed=true)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    for iteration ∈ 1:iterations
        time = iteration

        rx, ry, rz = solve_equation!(U_eqn, U, boundaries.U, solvers.U , xdir, ydir, zdir, config)

        # h correction - not sure if this is necessary but using this anyway
        #inverse_diagonal!(rD, U_eqn, config)
        #interpolate!(rDf, rD, config)
        #remove_pressure_source!(U_eqn, ∇h, config)

        H!(Hv, U, U_eqn, config)

        # Interpolate faces
        interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, boundaries.U, time, config)

        flux!(mdotf, Uf, config)
        div!(divHv, mdotf, config)

        @. prev = h.values
        rh = solve_equation!(h_eqn, h, boundaries.h, solvers.h, config)
        explicit_relaxation!(h, prev, solvers.h.relax, config)

        for i ∈ 1:ncorrectors

            discretise!(h_eqn, h, config)
            apply_boundary_conditions!(h_eqn, bouundaries.h, nothing, time, config)

            rp = solve_system!(h_eqn, solvers.h, h, nothing, config)
            explicit_relaxation!(h, prev, solvers.h.relax, config)
        end

        #correct_mass_flux
        #correct_velocity!()

        R_ux[iteration] = rx
        R_uy[iteration] = ry
        R_uz[iteration] = rz
        R_h[iteration] = rh

        Uz_convergence = true
        #if typeof(mesh)

        if (R_ux[iteration] <= solvers.U.convergence &&
            R_uy[iteration] <= solvers.U.convergence &&
            Uz_convergence &&
            R_h[iteration] <= solvers.h.convergence)

            progress.n = iterations
            finish!(progress)
            @info "Simulation converged in $iteration iterations"
            if !signbit(write_interval)
                save_output(model, outputWriter, iteration, time, config)
                save_postprocessing(postprocess, iteration, time, mesh, outputWriter, config.boundaries)
            end
            break
        end

        
        ProgressMeter.next!(
            progress, showvalues = [
                (:iter, iteration),
                (:Ux, R_ux[iteration]),
                (:Uy, R_uy[iteration]),
                (:Uz, R_uz[iteration]),
                (:h, R_h[iteration]),
                turbulenceModel.state.residuals...
            ]
        )

        runtime_postprocessing!(postprocess, iteration, iterations)

        if iteration % write_interval + signbit(write_interval) == 0
            save_output_film(model, outputWriter, iteration, time, config)
            save_postprocessing(postprocess, iteration, time, mesh, outputWriter, config.boundaries)
        end

    end # end for loop

    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, h=R_h)
end

function correct_mass_flux()
end

# Reworked save_output for film model
function save_output_film(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config
    ) where {T,F,SO,M,Tu,E,D,BI}
    args = (
            ("U", model.momentum.U), 
            ("h", model.momentum.h)
        )
    
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end

#function correct_mass_flux(mdotf, h, rDf, config)
#    (; faces, cells, boundary_cellsID) = mdotf.mesh
#    (; hardware) = config
#    (; backend, workgroup) = hardware
#
#    n_faces = length(faces)
#    n_bfaces = lenght(boundary_cellsID)
#    n_ifaces = n_faces - n_bfaces
#
#    ndrange = n_ifaces
#    kernel! = _correct_mass_flux(_setup(backend, workgroup, ndrange)...)
#    kernel!(mdotf, p, rDf, faces, cells, n_bfaces)
#end
#
#@kernel function _correct_mass_flux(mdotf, h, rDf, faces, cells, n_bfaces)
