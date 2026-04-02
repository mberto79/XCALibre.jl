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
    τθw = VectorField(mesh)
    Df = FaceScalarField(mesh)
    

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
        - Laplacian{schemes.h.laplacian}(Df,h)
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

    Df = get_flux(h_eqn, 2)
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

    PLf = FaceScalarField(mesh)
    ∇PL = Grad{Gauss}(PLf)
    P_capf = FaceScalarField(mesh)
    P_hydrf = FaceScalarField(mesh)
    Surf_tensionf = FaceScalarField(mesh)
    Pf = FaceScalarField(mesh)
    ∇P = Grad{Gauss}(Pf)
    h∇P = VectorField(mesh)


    Δh = ScalarField(mesh)
    Δhf = FaceScalarField(mesh)
    hmdotf = FaceScalarField(mesh)

    w = ScalarField(mesh)
    wf = FaceScalarField(mesh)
    ∇w = Grad{schemes.h.gradient}(w)

    Hv = VectorField(mesh)
    HbyA = VectorField(mesh)
    rD = ScalarField(mesh)
    rDf = FaceScalarField(mesh)
    
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
        Dirichlet(:inlet, 0),
        Dirichlet(:outlet, 0),
        Dirichlet(:inlet_sides, 0),
        Dirichlet(:top_of_plate, 0),
        Dirichlet(:side_1, 0),
        Dirichlet(:side_2, 0)
    ]

    n_cells = length(mesh.cells)

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
    interpolate!(Δhf, Δh, config)
    correct_boundaries!(Δhf, Δh, Δh_bc, time, config)

    @. rhohf.values = hf.values *  rho.values[1]
    @. phif.values = mdotf.values * rhohf.values
    @. hmdotf.values = mdotf.values * hf.values

    #@info "need to readd Pg term - Coupling term for other phase"
    Pg = 0# Test Pg term set to zero, as the gradient is found this value doesn't matter

    @. mu_h.values = 3*mu/h.values
    for i ∈ eachindex(Δhf.values)
        P_hydrf.values[i] = rho.values[1]*hf.values[i]*dot(n,G)
        P_capf.values[i] = Pg
        Surf_tensionf[i] = coeffs.σ*Δhf[i]
        PLf[i] = P_capf[i] - P_hydrf[i] - Surf_tensionf[i]
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
        h∇P[i] =  h[i].*∇P[i]

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
        interpolate!(rDf, rD, config)
        remove_film_pressure_source!(U_eqn, P_hydrf, Surf_tensionf, rho.values[1], config)

        rh = 0
        for i ∈ 1:inner_loops
            HbyA!(HbyA, Hv, U, U_eqn, rD, config)

            interpolate!(Uf, HbyA, config)
            correct_boundaries!(Uf, HbyA, boundaries.U, time, config)

            flux!(mdotf, Uf, config)
            
            @. hmdotf.values = mdotf.values * hf.values

            for i ∈ eachindex(Df.values)
                Df[i] = rho.values[1] * dot(n,G) * hf[i] * rDf[i]
            end
            
            div!(divPhi, hmdotf, config)
            
            @. prev = h.values
            rh = solve_equation!(h_eqn, h, boundaries.h, solvers.h, config, time=time)

            if i == inner_loops
                explicit_relaxation!(h, prev, 1.0, config)
            else
                explicit_relaxation!(h, prev, solvers.h.relax, config)
            end
            
            limit_h!(h, coeffs.h_floor, config)
            
            interpolate!(hf, h, config)
            correct_boundaries!(hf, h, boundaries.h, time, config)

            correct_velocity2!(U, HbyA, h, P_hydrf, Surf_tensionf, rho.values[1], rD, config)
            correct_mass_flux2!(mdotf, Df, h_eqn, config)
        end
    

        laplacian!(Δh, hf, h, boundaries.h, time, config, disp_warn=false)
        interpolate!(Δhf, Δh, config)
        correct_boundaries!(Δhf, Δh, Δh_bc, time, config)
            
        @. rhohf.values = hf.values *  rho.values[1]
        @. phif.values = mdotf.values * rhohf.values

        @. mu_h.values = 3*mu/h.values
        for i ∈ eachindex(Δhf.values)
            P_hydrf.values[i] = rho.values[1]*hf.values[i]*dot(n,G)
            P_capf.values[i] = Pg
            Surf_tensionf.values[i] = coeffs.σ*Δhf[i]

            PLf[i] = P_capf[i] - P_hydrf[i] - Surf_tensionf[i]
        end

        grad!(∇PL, PLf, config)

        for i ∈ eachindex(h.values) 
            w[i] = (h.values[i] > coeffs.h_crit)
        end

        # correct U for non-wetted
        for i ∈ eachindex(U.x)
            U[i] = U[i] .* w[i]
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

function remove_film_pressure_source!(U_eqn,  P_hyrdf, Surf_tensionf, rho, config)
    
    # backend = _get_backend(get_phi(ux_eqn).mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    cells = get_phi(U_eqn).mesh.cells
    source_sign = 1
    (; bx, by, bz) = U_eqn.equation

    
    ∇P_hydr = Grad{Gauss}(P_hyrdf)
    ∇P_surf = Grad{Gauss}(Surf_tensionf)

    grad!(∇P_hydr, P_hyrdf, config)
    grad!(∇P_surf, Surf_tensionf, config)

    ndrange = length(bx)
    kernel! = _remove_film_pressure_source!(_setup(backend, workgroup, ndrange)...)
    kernel!(cells, ∇P_hydr, ∇P_surf, rho, bx, by, bz)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _remove_film_pressure_source!(cells,  ∇P_hydr, ∇P_surf, rho, bx, by, bz) #Extend to 3D
    i = @index(Global)

    @uniform begin
        ∇P_hydr_x, ∇P_hydr_y, ∇P_hydr_z = ∇P_hydr.result.x, ∇P_hydr.result.y, ∇P_hydr.result.z
        ∇P_surf_x, ∇P_surf_y, ∇P_surf_z = ∇P_surf.result.x, ∇P_surf.result.y, ∇P_surf.result.z
    end

    @inbounds begin
        (; volume) = cells[i]
        bx[i] -= (-∇P_hydr_x[i] + ∇P_surf_x[i]/rho)*volume
        by[i] -= (-∇P_hydr_y[i] + ∇P_surf_y[i]/rho)*volume
        bz[i] -= (-∇P_hydr_z[i] + ∇P_surf_z[i]/rho)*volume
    end
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



function HbyA!(HbyA, Hv, U, U_eqn, rD, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    H!(Hv, U, U_eqn, config)

    (; bx, by, bz) = U_eqn.equation

    ndrange = length(U)

    kernel! = _HbyA!(_setup(backend, workgroup, ndrange)...)
    kernel!(HbyA, Hv, bx, by, bz, rD)
end

@kernel function _HbyA!(HbyA, Hv, bx, by, bz, rD)
    i = @index(Global)

    @uniform begin
        HbyA_x, HbyA_y, HbyA_z = HbyA.x, HbyA.y, HbyA.z
        Hv_x, Hv_y, Hv_z = Hv.x, Hv.y, Hv.z
        rDvalues = rD.values
        _bx, _by, _bz = bx, by, bz
    end

    @inbounds begin
        rDvaluesi = rDvalues[i]
        HbyA_x[i] = rDvaluesi*(Hv_x[i]+_bx[i])
        HbyA_y[i] = rDvaluesi*(Hv_y[i]+_by[i])
        HbyA_z[i] = rDvaluesi*(Hv_z[i]+_bz[i])
    end
end

function correct_velocity2!(U, HbyA, h, P_hyrdf, Surf_tensionf, rho, rD, config)
    (; mesh) = U
    (; hardware) = config
    (; backend, workgroup) = hardware

    ∇P_hydr = Grad{Gauss}(P_hyrdf)
    ∇P_surf = Grad{Gauss}(Surf_tensionf)

    grad!(∇P_hydr, P_hyrdf, config)
    grad!(∇P_surf, Surf_tensionf, config)

    ndrange = length(U)
    kernel! = _correct_velocity_film!(_setup(backend, workgroup, ndrange)...)
    kernel!(U, HbyA, h, ∇P_hydr, ∇P_surf, rho, rD)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _correct_velocity_film!(U, HbyA, h, ∇P_hydr, ∇P_surf, rho, rD)
    i = @index(Global)
    @uniform begin
        Ux, Uy, Uz = U.x, U.y, U.z
        HbyAx, HbyAy, HbyAz = HbyA.x, HbyA.y, HbyA.z
        ∇P_hydr_x, ∇P_hydr_y, ∇P_hydr_z = ∇P_hydr.result.x, ∇P_hydr.result.y, ∇P_hydr.result.z
        ∇P_surf_x, ∇P_surf_y, ∇P_surf_z = ∇P_surf.result.x, ∇P_surf.result.y, ∇P_surf.result.z
        rDvalues = rD.values
    end

    @inbounds begin
        rDvalues_i = rDvalues[i]
        Ux[i] = HbyAx[i]/h[i] + (-∇P_hydr_x[i] + ∇P_surf_x[i]/rho) * rDvalues_i
        Uy[i] = HbyAy[i]/h[i] + (-∇P_hydr_y[i] + ∇P_surf_y[i]/rho) * rDvalues_i
        Uz[i] = HbyAz[i]/h[i] + (-∇P_hydr_z[i] + ∇P_surf_z[i]/rho) * rDvalues_i
    end
end

function correct_mass_flux2!(mdotf, Df, h_eqn, config)
    (; mesh) = mdotf
    (; faces, cells, boundary_cellsID) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    h = h_eqn.model.terms[1].phi

    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    #ndrange = length(mdotf)
    ndrange = n_ifaces
    kernel! = _correct_mass_flux2!(_setup(backend, workgroup, ndrange)...)
    kernel!(mdotf, h, Df, mesh)

end

@kernel function _correct_mass_flux2!(mdotf, h, Df, mesh)
    i = @index(Global)

    ownerCells = mesh.faces[i].ownerCells
    snGrad = -(h[ownerCells[2]]-h[ownerCells[1]])/mesh.faces[i].delta

    mdotf[i] -= Df[i] * snGrad
end
