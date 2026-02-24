export filmModel!

function filmModel!(
    model, config;
    output=VTK()#, pref=nothing, ncorrectors=0, inner_loops=0
)
    #print("Using film model\n")
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
    

    @info "Pre-allocating fields..."
    rho_mdotf = FaceScalarField(mesh)
    hmdotf = FaceScalarField(mesh)
    rhohf = FaceScalarField(mesh)
    Sm = ScalarField(mesh)
    initialise!(Sm, 0)
    rho_l = ScalarField(mesh)
    initialise!(rho_l, rho.values)
    h∇PL = VectorField(mesh)
    Ph = VectorField(mesh)
    τw = VectorField(mesh)
    τθw = VectorField(mesh)
    

    @info "Defining models.."

    # Edit
    @info "U equation still need updating"
    U_eqn = (
        Time{schemes.U.time}(rhohf, U)
        + Divergence{schemes.U.divergence}(hmdotf,U)
        ==
          Source(h∇PL)
        + Source(Ph)
        - Source(τw)
        + Source(τθw)
    ) → VectorEquation(U, boundaries.U)

    h_eqn = (
        Time{schemes.h.time}(rho_l, h)
        + Divergence{schemes.h.divergence}(rho_mdotf, h)
        ==
        Source(Sm)
    ) → ScalarEquation(h, boundaries.h)

    @info "Initialising preconditioners"

    @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    @reset h_eqn.preconditioner = set_preconditioner(solvers.h.preconditioner, h_eqn)

    @info "Pre-allocating solvers"

    @reset U_eqn.solver = _workspace(solvers.U.solver, _b(U_eqn, XDir()))
    @reset h_eqn.solver = _workspace(solvers.h.solver, _b(h_eqn))

    @info "No turbulence model for now"
    #p_eqn = (Time{schemes.h.time}(rho_l,h)==Source(Sm)) → ScalarEquation(h, boundaries.h)
    #turbulenceModel, config = initialise(model.turbulence, model, mdotf, p_eqn, config)

    residuals = solver_variant(
        model, #turbulenceModel,
         U_eqn, h_eqn, config
    )
end

function FilmModel(
    model, #turbulenceModel,
     U_eqn, h_eqn, config;
    output=VTK(), ncorrectors=0
)

    
    (; U, h, Uf, hf, coeffs) = model.momentum
    (; rho, nu) = model.fluid
    mesh = model.domain
    (; solvers, schemes, runtime, hardware, boundaries, postprocess) = config
    (; workgroup, backend) = hardware
    (; iterations, write_interval, dt) = runtime
    (; backend) = hardware

    Postprocess = convert_time_to_iterations(postprocess, model, dt, iterations)
    rhohf = get_flux(U_eqn, 1)
    hmdotf = get_flux(U_eqn, 2)

    h∇PL = get_source(U_eqn, 1)
    Ph = get_source(U_eqn,2)
    τw = get_source(U_eqn,3)
    τθw = get_source(U_eqn,4)
    
    rho_mdotf = get_flux(h_eqn,2)
    Sm = get_source(h_eqn,1)
    mu = nu.values*rho.values

    outputWriter = initialise_writer(output, model.domain)

    @info "Allocating working memory"

    n = [sind(coeffs.ϕ),0,cosd(coeffs.ϕ)]
    g = 9.8
    G = g*[0,0,-1]

    # Define aux fields
    mdotf = FaceScalarField(mesh)

    PLf = FaceScalarField(mesh)
    ∇PL = Grad{Gauss}(PLf)

    ∇h = Grad{schemes.h.gradient}(h)
    ∇hf = FaceVectorField(mesh)
    Δh = ScalarField(mesh)
    Δhf = FaceScalarField(mesh)

    w = ScalarField(mesh)
    wf = FaceScalarField(mesh)
    ∇w = Grad{schemes.h.gradient}(w)
    
    plate_tangent_vector = [1,0,0] # temporary,  should be worked out later

    #Hv = VectorField(mesh)
    #rD = ScalarField(mesh)
    #rDf = FaceScalarField(mesh)


    h_inlet = h.values[1]
    h_min = 1e-4
    factor = 1.5
    for i ∈ eachindex(h.values)
    #    #h.values[i] = -(h_inlet-h_min)/(0.01*2)*mesh.cells[i].centre[1]-(h_inlet-h_min)/(0.01*2)*2*abs(mesh.cells[i].centre[2]-0.005)+h_inlet
    #    a = h_inlet/2
    #    c = factor
    #    b = (log(h_min/a))/(0.01^c)
    #    h.values[i] = a*exp(b*mesh.cells[i].centre[1]^c)#+a*exp(b*abs(mesh.cells[i].centre[2]-0.005)^c)
        #println(sqrt((0.005-mesh.cells[i].centre[1])^2+(0.005-mesh.cells[i].centre[2])^2+(0.005-mesh.cells[i].centre[3])^2))
        if sqrt((0.05-mesh.cells[i].centre[1])^2+(0.05-mesh.cells[i].centre[2])^2+(0.005-mesh.cells[i].centre[3])^2)<0.1^2
            h.values[i] = 0.001
        end
    end

    w_bc = [
        #Dirichlet(:inlet, 1),
        Zerogradient(:inlet),
        Zerogradient(:outlet),
        #Dirichlet(:outlet, 0),
        Zerogradient(:top),
        Zerogradient(:bottom)
    ]
    Δh_bc = [
        Zerogradient(:inlet),
        Zerogradient(:outlet),
        Zerogradient(:top),
        Zerogradient(:bottom)
    ]

    n_cells = length(mesh.cells)

    #(;cells) = U.mesh
    #ndrange_VF = length(cells)

    #(;faces) = U.mesh
    #ndrange_FSF = length(faces)
    
    #PLf_func! = _calculate_PLf!(_setup(backend, workgroup, ndrange_FSF)...)
    #h∇PL_func! = _calculate_h∇PL!(_setup(backend, workgroup, ndrange_VF)...)
    #Ph_func! = _calculate_Ph!(_setup(backend, workgroup, ndrange_VF)...)
    #τw_func! = _calculate_τw!(_setup(backend, workgroup, ndrange_VF)...)
    #τθw_func! = _calculate_τθw!(_setup(backend, workgroup, ndrange_VF)...)
    
    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    
    prev_u = KernelAbstractions.zeros(backend, TF, n_cells)
    prev_v = KernelAbstractions.zeros(backend, TF, n_cells)
    prev_w = KernelAbstractions.zeros(backend, TF, n_cells)
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

    rho_mdotf = mdotf.values .* rho.values
    
    # Getting the laplacian of h for first U calculation
    grad!(∇h, hf, h, boundaries.h, time, config)
    limit_gradient!(schemes.h.limiter, ∇h, h, config)
    div!(Δh, ∇hf, config)
    interpolate!(Δhf, Δh, config)
    correct_boundaries!(Δhf, Δh, Δh_bc, time, config)

    # Getting h * mdotf and rho * h for U calculation
    # hf is calculated within grad!()
    @. rhohf.values = hf.values *  rho.values
    @. hmdotf.values = mdotf.values * hf.values * rho.values

    @info "need to readd Pg term - Coupling term for other phase"
    Pg = 0# Test Pg term set to zero, as the gradient is found this value doesn't matter
    @info "need to fix surface tension term"
    for i ∈ 1:length(Δhf.values)
        PLf[i] = Pg - hf.values[i]*dot(n,G) - coeffs.σ*Δhf[i]
    end

    grad!(∇PL, PLf, config)
    #limit_gradient!(schemes.PL.limiter, ∇PL, PL, config)

    for i ∈ eachindex(h)
        w[i] = (h.values[i] > coeffs.h_crit)
    end

    grad!(∇w, wf, w, w_bc, time, config)

    for i ∈ eachindex(h.values)
        multiplier =0; 3*(mu/h.values[i])
        τw.x.values[i] = multiplier * U.x.values[i]
        τw.y.values[i] = multiplier * U.y.values[i]
        τw.z.values[i] = multiplier * U.z.values[i]

        Ph_local = (rho.values*g*sind(coeffs.ϕ)*h[i]) .*plate_tangent_vector
        Ph.x.values[i] = Ph_local[1]
        Ph.y.values[i] = Ph_local[2]
        Ph.z.values[i] = Ph_local[3]

        h∇PL_local = h[i]*∇PL[i]
        h∇PL.x.values[i] = h∇PL_local[1]
        h∇PL.y.values[i] = h∇PL_local[2]
        h∇PL.z.values[i] = h∇PL_local[3]

        τθw[i] = coeffs.β*coeffs.σ * (1-cosd(coeffs.θm)) .* ∇w[i]
    end
    @info "Starting loops"
    
    progress = Progress(iterations; dt=1.0, showspeed=true)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()
    rh = 0
    #rx = ry = rz = 1

    for iteration ∈ 1:iterations
        time = iteration
        
        @. prev_u = U.x.values
        @. prev_v = U.y.values
        @. prev_w = U.z.values
        rx, ry, rz = solve_equation!(U_eqn, U, boundaries.U, solvers.U , xdir, ydir, zdir, config)
        explicit_relaxation!(U.x.values, prev_u, solvers.U.relax, config)
        explicit_relaxation!(U.y.values, prev_v, solvers.U.relax, config)
        explicit_relaxation!(U.z.values, prev_w, solvers.U.relax, config)

        interpolate!(Uf, U, config)
        correct_boundaries!(Uf, U, boundaries.U, time, config)

        # h calculations
        flux!(mdotf, Uf, config)

        @. rho_mdotf =  mdotf.values .* rho.values
        

        @. prev = h.values

        #rh = solve_equation!(h_eqn, h, boundaries.h, solvers.h, config)
        #explicit_relaxation!(h, prev, solvers.h.relax, config)

        
        if (iteration == 1)
            
        end
        #for i ∈ 1:ncorrectors
        #    discretise!(h_eqn, h, config)
        #    apply_boundary_conditions!(h_eqn, boundaries.h, nothing, time, config)
#
        #    rh = solve_system!(h_eqn, solvers.h, h, nothing, config)
        #    explicit_relaxation!(h, prev, solvers.h.relax, config)
        #end
        
        #correct_mass_flux(mdotf, PL, rDf, config)

        for i ∈ eachindex(h.values)
            if (h.values[i]<=0) h.values[i] = 1e-18 end
        end

        grad!(∇h, hf, h, boundaries.h, time, config)
        @. hmdotf.values = mdotf.values * hf.values * rho.values
        @. rhohf.values =  hf.values * rho.values

        limit_gradient!(schemes.h.limiter, ∇h, h, config)
        div!(Δh, ∇hf, config)
        interpolate!(Δhf, Δh, config)
        correct_boundaries!(Δhf, Δh, Δh_bc, time, config)

        for i ∈ 1:length(Δhf.values)
            PLf[i] = Pg - hf.values[i]*dot(n,G) - coeffs.σ*Δhf[i]
        end

        grad!(∇PL, PLf, config)        
        

        for i ∈ eachindex(h)
            w[i] = (h.values[i] > coeffs.h_crit)
        end

        grad!(∇w, wf, w, w_bc, time, config)

        #PLf_func!(PLf, hf, Δhf, model)
        #Ph_func!(Ph, h, model)
        #h∇PL_func!(h∇PL, ∇PL, h)
        #τw_func!(τw, U, h, mu)
        #τθw_func!(τθw, ∇w, model)
        for i ∈ eachindex(h)
            multiplier =0; 3*(mu/h.values[i])
            τw.x.values[i] = multiplier * U.x.values[i]
            τw.y.values[i] = multiplier * U.y.values[i]
            τw.z.values[i] = multiplier * U.z.values[i]

            Ph_local = (rho.values*g*sind(coeffs.ϕ)*h[i]) .*plate_tangent_vector
            Ph.x.values[i] = Ph_local[1]
            Ph.y.values[i] = Ph_local[2]
            Ph.z.values[i] = Ph_local[3]

            h∇PL_local = h[i]*∇PL[i]
            h∇PL.x.values[i] = h∇PL_local[1]
            h∇PL.y.values[i] = h∇PL_local[2]
            h∇PL.z.values[i] = h∇PL_local[3]
            
            τθw[i] = coeffs.β*coeffs.σ * (1-cosd(coeffs.θm)) .* ∇w[i]
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
            #R_uy[iteration] <= solvers.U.convergence &&
            Uz_convergence &&
            R_h[iteration] <= solvers.h.convergence)

            progress.n = iterations
            finish!(progress)
            @info "Simulation converged in $iteration iterations"
            if !signbit(write_interval)
                save_output_film(model, outputWriter, iteration, time, config, w)
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
                #turbulenceModel.state.residuals...
            ]
        )

        runtime_postprocessing!(postprocess, iteration, iterations)

        if iteration % write_interval + signbit(write_interval) == 0
            save_output_film(model, outputWriter, iteration, time, config, w)
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
            ("h", model.momentum.h),
        )
    
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end

function save_output_film(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config, w
    ) where {T,F,SO,M,Tu,E,D,BI}
    args = (
            ("U", model.momentum.U), 
            ("h", model.momentum.h),
            ("w", w)
        )
    
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end


#@kernel function _calculate_τw!(τw, U, h, mu)
#    i = @index(Global)
#
#    multiplier = 3*(mu/h.values[i])
#    τw.x.values[i] = multiplier * U.x.values[i]
#    τw.y.values[i] = multiplier * U.y.values[i]
#    τw.z.values[i] = multiplier * U.z.values[i]
#end
#
#@kernel function _calculate_Ph!(Ph, h, model)
#    i = @index(Global)
#    (; fluid, momentum) = model
#    (; rho) = fluid
#    (; coeffs) = momentum
#    plate_tangent_vector = [1,0,0] # Temporary
#    g = 9.81# Temporary
#    Ph_local = (rho.values*g*sind(coeffs.ϕ)*h[i]) .*plate_tangent_vector
#    Ph.x.values[i] = Ph_local[1]
#    Ph.y.values[i] = Ph_local[2]
#    Ph.z.values[i] = Ph_local[3]
#end
#
#
#@kernel function _calculate_PLf!(PLf, hf, Δhf, model)
#    i = @index(Global)
#    
#    Pg = 0
#    (; coeffs) = model.momentum
#    n = [sind(coeffs.ϕ),0,cosd(coeffs.ϕ)]
#    g = 9.8
#    G = g*[0,0,-1] 
#    #println(i)
#    PLf[i] = Pg -hf.values[i]*dot(n,G) - coeffs.σ*Δhf[i];#Pg - (coeffs.σ * model.momentum.h[i] * (dot(n,G)))
#end
#
#@kernel function _calculate_h∇PL!(h∇PL, ∇PL, h)
#    i = @index(Global)
#
#    h∇PL_local = h[i]*∇PL[i]
#    h∇PL.x.values[i] = h∇PL_local[1]
#    h∇PL.y.values[i] = h∇PL_local[2]
#    h∇PL.z.values[i] = h∇PL_local[3]
#end
#
#@kernel function _calculate_τθw!(τθw, ∇w, model)
#    i = @index(Global)
#    (; coeffs) = model.momentum
#    τθw[i] = coeffs.β*coeffs.σ * (1-cosd(coeffs.θm)) .* ∇w[i]
#end

#function calculate_Ph!(Ph, h, model, config)
#    (; hardware) = config
#    (; backend, workgroup) = hardware
#
#    (; cells) = Ph.mesh
#    ndrange = length(cells)
#    kernel! = _calculate_Ph!(_setup(backend, workgroup, ndrange)...)
#    kernel!(Ph, h, model)
#end

#function calculate_h∇PL!(h∇PL, ∇PL, PLf, hf, h, Δhf, model, config)
#    (; hardware) = config
#    (; backend, workgroup) = hardware
#
#    (;cells) = PLf.mesh
#    ndrange = length(cells)
#    kernel! = _calculate_PLf!(_setup(backend, workgroup, ndrange)...)
#    kernel!(PLf, hf, Δhf, model)
#
#    grad!(∇PL, PLf, config)
#
#    (; cells) = h∇PL.mesh
#    ndrange_h∇PL = length(cells)
#    kernel! = _calculate_h∇PL!(_setup(backend, workgroup, ndrange_h∇PL)...)
#    kernel!(h∇PL, ∇PL, h)
#end

#function calculate_τw!(τw, U, h, mu, config)
#    (; hardware) = config
#    (; backend, workgroup) = hardware
#
#    (; cells) = τw.mesh
#    ndrange = length(cells)
#    kernel! = _calculate_τw!(_setup(backend, workgroup, ndrange)...)
#    kernel!(τw, U, h, mu)
#end

#function get_surface_tension!(model, surface_tension, Δh ,∇hf, config)
#    (; hardware) = config
#    (; backend, workgroup) = hardware
#    
#    div!(Δh, ∇hf, config)
#
#    (; cells) = surface_tension.mesh
#    ndrange = length(cells)
#    kernel! = _get_surface_tension!(_setup(backend, workgroup, ndrange)...)
#    kernel!(model, surface_tension, Δh)
#end

#@kernel function _get_surface_tension!(model, surface_tension, Δh)
#    i = @index(Global)
#
#    surface_tension.values[i] = model.momentum.coeffs * Δh.values[i]
#end
#
#function get_PL!(model, PL, surface_tension, config)
#    (; hardware) = config
#    (; backend, workgroup) = hardware
#
#    (; cells) = surface_tension.mesh
#    ndrange = length(cells)
#    kernel! = _get_PL!(_setup(backend, workgroup, ndrange)...)
#    kernel!(model, PL, surface_tension)
#end
#
#@kernel function _get_PL!(model, PL, surface_tension)
#    i = @index(Global)
#
#    
#
#    PL[i] =  - model.momentum.coeffs*model.momentum.h[i]* (dot(n,g)) - surface_tension[i]
#end
#
#function correct_mass_flux(mdotf, h, rDf, config)
#   (; faces, cells, boundary_cellsID) = mdotf.mesh
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
