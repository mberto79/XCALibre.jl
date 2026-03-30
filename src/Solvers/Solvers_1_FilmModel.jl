export filmModel!

function filmModel!(
    model, config;
    output=VTK(),#, pref=nothing, ncorrectors=
    inner_loops=2
)
    #print("Using film model\n")
    residuals = setup_FilmModel_Solver(
        FilmModel, model, config,
        output=output,
        inner_loops=inner_loops
    )
    
    return residuals
end

function setup_FilmModel_Solver(solver_variant, model, config;
    output=VTK(), inner_loops=2)

    (; solvers, schemes, runtime, hardware, boundaries) = config

    @info "Extracting configuration and input fields..."

    (; U, h, Uf, hf) = model.momentum
    mesh = model.domain
    (; rho) = model.fluid
    

    @info "Pre-allocating fields..."
    rho_mdotf = FaceScalarField(mesh)
    initialise!(rho_mdotf, 0);
    phif = FaceScalarField(mesh)
    rhohf = FaceScalarField(mesh)
    mu_h = ScalarField(mesh)
    Sm = ScalarField(mesh)
    divPhi = ScalarField(mesh)
    initialise!(Sm, 0)
    rho_l = FaceScalarField(mesh)
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
        + Divergence{schemes.U.divergence}(phif,U)
        + Si(mu_h, U)
        ==
        - Source(h∇PL)
        + Source(Ph)
        + Source(τθw)
        
    ) → VectorEquation(U, boundaries.U)

    h_eqn = (
        Time{schemes.h.time}(h)
        ==
        - Source(divPhi)
        + Source(Sm)
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
         U_eqn, h_eqn, config; inner_loops=inner_loops
    )
end

function FilmModel(
    model, #turbulenceModel,
     U_eqn, h_eqn, config;
    output=VTK(), ncorrectors=0, inner_loops=2
)

    
    (; U, h, Uf, hf, coeffs) = model.momentum
    (; rho, nu) = model.fluid
    mesh = model.domain
    (; solvers, schemes, runtime, hardware, boundaries, postprocess) = config
    (; workgroup, backend) = hardware
    (; iterations, write_interval, dt) = runtime
    (; backend) = hardware
    
    dt_cpu = zeros(_get_float(mesh), 1)
    copyto!(dt_cpu, config.runtime.dt)

    postprocess = convert_time_to_iterations(postprocess, model, dt, iterations)
    rhohf = get_flux(U_eqn, 1)
    phif = get_flux(U_eqn, 2)
    mu_h = get_flux(U_eqn, 3)

    h∇PL = get_source(U_eqn, 1)
    Ph = get_source(U_eqn,2)
    τθw = get_source(U_eqn,3)

    #rho_mdotf = get_flux(h_eqn,2)
    divPhi = get_source(h_eqn,1)
    Sm = get_source(h_eqn, 2)
    mu = nu.values*rho.values

    outputWriter = initialise_writer(output, model.domain)

    @info "Allocating working memory"

    n = [sind(coeffs.ϕ),0,cosd(coeffs.ϕ)]
    g = 9.8
    G = g*[0,0,-1]

    # Define aux fields
    mdotf = FaceScalarField(mesh)
    rephif = FaceVectorField(mesh)

    PLf = FaceScalarField(mesh)
    ∇PL = Grad{Gauss}(PLf)

    ∇h = Grad{schemes.h.gradient}(h)
    ∇hf = FaceVectorField(mesh)
    Δh = ScalarField(mesh)
    Δhf = FaceScalarField(mesh)
    hmdotf = FaceScalarField(mesh)

    w = ScalarField(mesh)
    wf = FaceScalarField(mesh)
    ∇w = Grad{schemes.h.gradient}(w)

    drhoHdt = ScalarField(mesh)  # this is the one you need to explicitly update after solving h equation

    Hv = VectorField(mesh)
    rD = ScalarField(mesh)
    
    #plate_tangent_vector = [1,0,0] # temporary,  should be worked out later
    plate_tangent_vector = Vector{}([1,0,0])

    w_bc = [
        Dirichlet(:inlet, 1),
        Extrapolated(:outlet),
        Extrapolated(:inlet_sides),
        Extrapolated(:top_of_plate),
        Extrapolated(:side_1),
        Extrapolated(:side_2)
    ]
    
    Δh_bc = [
        #Extrapolated(:inlet),
        #Extrapolated(:outlet),
        #Extrapolated(:inlet_sides),
        #Extrapolated(:top_of_plate),
        #Extrapolated(:side_1),
        #Extrapolated(:side_2)
        Dirichlet(:inlet, 0),
        Dirichlet(:outlet, 0),
        Dirichlet(:inlet_sides, 0),
        Dirichlet(:top_of_plate, 0),
        Dirichlet(:side_1, 0),
        Dirichlet(:side_2, 0)
    ]
    #w_bc = [
    #    Dirichlet(:inlet, 1),
    #    Extrapolated(:outlet),
    #    Extrapolated(:top),
    #    Extrapolated(:bottom)
    #]
    #Δh_bc = [
    #    Extrapolated(:inlet),
    #    Extrapolated(:outlet),
    #    Extrapolated(:top),
    #    Extrapolated(:bottom)
    #]


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
    
    prev = KernelAbstractions.zeros(backend, TF, n_cells)

    # Pre-allocate vectors to hold residuals
    R_ux = zeros(TF, iterations)
    R_uy = zeros(TF, iterations)
    R_uz = zeros(TF, iterations)
    R_h = zeros(TF, iterations)
    cellsCourant = KernelAbstractions.zeros(backend, TF, n_cells)

    # Initial calculations
    time = zero(TF) # assuming time = 0

    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, boundaries.U, time, config)
    flux!(mdotf, Uf, config)
    
    # Getting the laplacian of h for first U calculation
    laplacian!(Δh, hf, h, boundaries.h, time, config)
    #grad!(∇h, hf, h, boundaries.h, time, config)
    #limit_gradient!(schemes.h.limiter, ∇h, h, config)
    #interpolate!(∇hf, ∇h.result, config)
    #correct_boundaries!(∇hf, ∇h.result, ∇h_bc, time, config)
    #div!(Δh, ∇hf, config)
    interpolate!(Δhf, Δh, config)
    correct_boundaries!(Δhf, Δh, Δh_bc, time, config)

    
        
    @. rhohf.values = hf.values *  rho.values[1]
    @. phif.values = mdotf.values * rhohf.values
    @. hmdotf.values = mdotf.values * hf.values

    #@info "need to readd Pg term - Coupling term for other phase"
    Pg = 0# Test Pg term set to zero, as the gradient is found this value doesn't matter

    @. mu_h.values = 3*mu/h.values
    for i ∈ eachindex(Δhf.values)
        PLf[i] = Pg - rho.values[1]*hf.values[i]*dot(n,G) - coeffs.σ*Δhf[i]
    end

    grad!(∇PL, PLf, config)

    for i ∈ eachindex(h.values) 
        w[i] = (h.values[i] > coeffs.h_crit)
    end

    grad!(∇w, wf, w, w_bc, time, config)

    for i ∈ eachindex(h.values)
        Ph_local = (rho.values[1]*g*sind(coeffs.ϕ)*h[i]) .*plate_tangent_vector
        Ph.x.values[i] = Ph_local[1]
        Ph.y.values[i] = Ph_local[2]
        Ph.z.values[i] = Ph_local[3]

        h∇PL[i] = h[i].*∇PL[i]

        τθw[i] = coeffs.β*coeffs.σ * (1-cosd(coeffs.θm)) .* ∇w.result[i]
    end

    @info "Starting loops"
    
    progress = Progress(iterations; dt=1.0, showspeed=true)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()
    #rh = 0
    rx = ry = rz = 0
    @time for iteration ∈ 1:iterations
        copyto!(dt_cpu, config.runtime.dt)
        time += dt_cpu[1]

        rx, ry, rz = solve_equation!(U_eqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config, time=time)
        
        inverse_diagonal!(rD, U_eqn, config)
        remove_source!(U_eqn, h∇PL, 1, config)
        remove_source!(U_eqn, Ph, 2, config)
        remove_source!(U_eqn, τθw, 3, config)

        rh = 0
        for i ∈ 1:inner_loops
            H!(Hv, U, U_eqn, config)

            interpolate!(Uf, Hv, config)
            correct_boundaries!(Uf, Hv, boundaries.U, time, config)

            flux!(mdotf, Uf, config)
            
            
            #@. rhohf.values = hf.values *  rho.values[1]
            @. phif.values = mdotf.values * rhohf.values
            @. hmdotf.values = mdotf.values * hf.values

            div!(divPhi, hmdotf, config)
            
            @. prev = h.values
            rh = solve_equation!(h_eqn, h, boundaries.h, solvers.h, config, time=time)

            if i == inner_loops
                explicit_relaxation!(h, prev, 1.0, config)
            else
                explicit_relaxation!(h, prev, solvers.h.relax, config)
            end
            
            limit_h!(h, coeffs.h_floor, config)
            
            laplacian!(Δh, hf, h, boundaries.h, time, config, disp_warn=false)
            #grad!(∇h, hf, h, boundaries.h, time, config)
            #limit_gradient!(schemes.h.limiter, ∇h, h, config)
            #interpolate!(∇hf, ∇h.result, config)
            #correct_boundaries!(∇hf, ∇h.result, ∇h_bc, time, config)
            #div!(Δh, ∇hf, config)
            
            interpolate!(Δhf, Δh, config)
            correct_boundaries!(Δhf, Δh, Δh_bc, time, config)
            
            
            @. rhohf.values = hf.values *  rho.values[1]
            @. phif.values = mdotf.values * rhohf.values

            


            @. mu_h.values = 3*mu/h.values
            for i ∈ eachindex(Δhf.values)
                PLf[i] = Pg - rhohf[i]*dot(n,G) - coeffs.σ*Δhf[i]
            end

            grad!(∇PL, PLf, config)

            for i ∈ eachindex(h.values) 
                w[i] = (h.values[i] > coeffs.h_crit)
            end

            grad!(∇w, wf, w, w_bc, time, config)

            for i ∈ eachindex(h.values)
                Ph_local = (rho.values[1]*g*sind(coeffs.ϕ)*h[i]) .*plate_tangent_vector
                Ph.x.values[i] = Ph_local[1]
                Ph.y.values[i] = Ph_local[2]
                Ph.z.values[i] = Ph_local[3]

                h∇PL[i] = h[i].*∇PL[i]

                τθw[i] = coeffs.β*coeffs.σ * (1-cosd(coeffs.θm)) .* ∇w.result[i]
            end

            correct_mass_flux2!(phif, h_eqn, config)
            correct_velocity!(U, Hv, h∇PL, Ph, τθw, rD, config)
        end
    
        #for i ∈ 1:ncorrectors
        #    discretise!(h_eqn, h, config)
        #    apply_boundary_conditions!(h_eqn, boundaries.h, nothing, time, config)
  
        #    rh = solve_system!(h_eqn, solvers.h, h, nothing, config)
        #    explicit_relaxation!(h, prev, solvers.h.relax, config)
        #end
        
        
        R_ux[iteration] = rx
        R_uy[iteration] = ry
        R_uz[iteration] = rz
        R_h[iteration] = rh



        maxCourant = max_courant_number!(cellsCourant, model, config)
        update_dt!(config.runtime, maxCourant)

        ProgressMeter.next!(
            progress, showvalues = [
                (:dt, dt_cpu[1]),
                (:time, time),
                (:Courant, maxCourant),
                (:Ux, R_ux[iteration]),
                (:Uy, R_uy[iteration]),
                (:Uz, R_uz[iteration]),
                (:h, R_h[iteration]),
                #turbulenceModel.state.residuals...
            ]
        )

        runtime_postprocessing!(postprocess, iteration, iterations)

        if iteration % write_interval + signbit(write_interval) == 0
            #save_output_film(model, outputWriter, iteration, time, config, w)
            #save_output_film(model, outputWriter, iteration, time, config, w, Δh)
            save_output_film(model, outputWriter, iteration, time, config, w, Δh, h∇PL, mu_h, Ph, τθw, divPhi)
            save_postprocessing(postprocess, iteration, time, mesh, outputWriter, config.boundaries)
        end

    end # end for loop

    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, h=R_h)
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

function save_output_film(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config, w, Δh, h∇PL, mu_h, Ph, τθw, divPhi
    ) where {T,F,SO,M,Tu,E,D,BI}
    args = (
            ("U", model.momentum.U), 
            ("h", model.momentum.h),
            ("w", w),
            ("Δh", Δh),
            ("h∇PL", h∇PL),
            ("mu_h", mu_h),
            ("Ph", Ph),
            ("τθw", τθw),
            ("divPhi", divPhi)
        )
    
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end

function save_output_film(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config, w, Δh
    ) where {T,F,SO,M,Tu,E,D,BI}
    args = (
            ("U", model.momentum.U), 
            ("h", model.momentum.h),
            ("w", w),
            ("Δh", Δh)
        )
    
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end

function limit_h!(h, h_floor, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; cells) = h.mesh
    ndrange = length(cells)

    kernel! = _limit_h!(_setup(backend, workgroup, ndrange)...)
    kernel!(h, h_floor)
end

@kernel function _limit_h!(h, h_floor)
    i = @index(Global)

    @inbounds begin
        if (h[i] <= h_floor) h[i] = h_floor end
    end
end

remove_source!(U_eqn::ME, S, Sindex, config) where {ME} = begin # Extend to 3D
    # backend = _get_backend(get_phi(ux_eqn).mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    cells = get_phi(U_eqn).mesh.cells
    source_sign = get_source_sign(U_eqn, Sindex)
    (; bx, by, bz) = U_eqn.equation

    ndrange = length(bx)
    kernel! = _remove_source!(_setup(backend, workgroup, ndrange)...)
    kernel!(cells, source_sign, S, bx, by, bz)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _remove_source!(cells, source_sign, S, bx, by, bz) #Extend to 3D
    i = @index(Global)


    @inbounds begin
        (; volume) = cells[i]
        calc = source_sign*S[i]*volume
        bx[i] -= calc[1]
        by[i] -= calc[2]
        bz[i] -= calc[3]
    end
end

function correct_velocity!(U, Hv, h∇PL, Ph, τθw, rD, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(U)
    kernel! = _correct_velocity_film!(_setup(backend, workgroup, ndrange)...)
    kernel!(U, Hv, h∇PL, Ph, τθw, rD)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _correct_velocity_film!(U, Hv, h∇PL, Ph, τθw, rD)
    i = @index(Global)

    @uniform begin
        Ux, Uy, Uz = U.x, U.y, U.z
        Hvx, Hvy, Hvz = Hv.x, Hv.y, Hv.z
        valsx = -h∇PL.x.values + Ph.x.values + τθw.x.values
        valsy = -h∇PL.y.values + Ph.y.values + τθw.y.values
        valsz = -h∇PL.z.values + Ph.z.values + τθw.z.values
        rDvalues = rD.values
    end

    @inbounds begin
        rDvalues_i = rDvalues[i]
        Ux[i] = Hvx[i] + valsx[i] * rDvalues_i
        Uy[i] = Hvy[i] + valsy[i] * rDvalues_i
        Uz[i] = Hvz[i] + valsz[i] * rDvalues_i
    end
end

function correct_mass_flux2!(phif, h_eqn, config)
    # sngrad = FaceScalarField(mesh)
    (; faces, cells, boundary_cellsID) = phif.mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    h = h_eqn.model.terms[1].phi
    A = _A(h_eqn)
    nzval = _nzval(A)
    colval = _colval(A)
    rowptr = _rowptr(A)

    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    ndrange = n_ifaces # length(n_ifaces) was a BUG! should be n_ifaces only!!!!
    kernel! = _correct_mass_flux2!(_setup(backend, workgroup, ndrange)...)
    kernel!(phif, h, nzval, colval, rowptr, faces, cells, n_bfaces)
    KernelAbstractions.synchronize(backend)

    BCs = config.boundaries[1] # assume periodics always defined by user (extract first)
    for BC ∈ BCs
        correct_mass_periodic(
            BC, phif, h, nzval, colval, rowptr, cells, faces, backend, workgroup)
        KernelAbstractions.synchronize(backend)
    end
end

@kernel function _correct_mass_flux2!(
    phif, h, nzval, colval, rowptr, faces, cells, n_bfaces)
    i = @index(Global)
    fID = i + n_bfaces

    @inbounds begin 
        face = faces[fID]
        (; area, normal, ownerCells, delta) = face 
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        h1 = h[cID1]
        h2 = h[cID2]
        # need to get aN from sparse system
        zID = spindex(rowptr, colval, cID1, cID2)
        aN = nzval[zID]
        phif[fID] -= aN*(h2 - h1)
    end
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
#end