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
    nu_h = ScalarField(mesh)
    Sm = ScalarField(mesh)
    divPhi = ScalarField(mesh)
    initialise!(Sm, 0)
    h∇PL = VectorField(mesh)
    Ph = VectorField(mesh)
    τθw = VectorField(mesh)
    Df = FaceScalarField(mesh)
    

    @info "Defining models.."

    # Edit
    @info "U equation still need updating"
    U_eqn = (
        Time{schemes.U.time}(h, U) # not hf
        + Divergence{schemes.U.divergence}(phif,U)
        + Si(nu_h, U)
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
    limit_capillary_dt!(config.runtime, coeffs)
    copyto!(dt_cpu, config.runtime.dt)

    postprocess = convert_time_to_iterations(postprocess, model, dt, iterations)
    phif = get_flux(U_eqn, 2)
    nu_h = get_flux(U_eqn, 3)

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
    G = g.*[0,0,-1]

    # Define aux fields
    hUf = FaceVectorField(mesh)

    mdotf = FaceScalarField(mesh)
    PLf = FaceScalarField(mesh)
    ∇PL = Grad{Gauss}(PLf)
    P_gasf = FaceScalarField(mesh)
    P_hydr = ScalarField(mesh)
    P_hydrf = FaceScalarField(mesh)
    P_surff = FaceScalarField(mesh)
    Pf = FaceScalarField(mesh)

    tempU = VectorField(mesh)
    Δh = ScalarField(mesh)
    Δhf = FaceScalarField(mesh)

    w = ScalarField(mesh)
    wf = FaceScalarField(mesh)
    ∇w = Grad{schemes.h.gradient}(w)

    Hv = VectorField(mesh)
    HbyA = VectorField(mesh)
    rD = ScalarField(mesh)
    rDf = FaceScalarField(mesh)
    
    #plate_tangent_vector = [1,0,0] # temporary,  should be worked out later
    plate_tangent_vector = Vector{}([1,0,0])
   
    u_inlet = boundaries.U[1].value
    h_inlet = boundaries.h[1].value
    hU_inlet = [u_inlet[1] .* h_inlet, u_inlet[2] .* h_inlet, u_inlet[3] .* h_inlet]

    internal_BCs = assign(
        region=mesh,
        (
            Δh = [
            Dirichlet(:inlet, 0),
            Dirichlet(:outlet, 0),
            Dirichlet(:inlet_sides, 0),
            Dirichlet(:top_of_plate, 0),
            Dirichlet(:side_1, 0),
            Dirichlet(:side_2, 0)
            ],
            w = [
            Dirichlet(:inlet, 1),
            Zerogradient(:outlet),
            Zerogradient(:inlet_sides),
            Zerogradient(:top_of_plate),
            Zerogradient(:side_1),
            Zerogradient(:side_2)
    ]
    #         w = [
    #         Dirichlet(:inlet, 1),
    #         Extrapolated(:outlet),
    #         Extrapolated(:inlet_sides),
    #         Extrapolated(:top_of_plate),
    #         Extrapolated(:side_1),
    #         Extrapolated(:side_2)
    # ]
        )
    )

    n_cells = length(mesh.cells)

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    
    h_old = KernelAbstractions.zeros(backend, TF, n_cells)
    prev = KernelAbstractions.zeros(backend, TF, n_cells)

    # Pre-allocate vectors to hold residuals
    R_ux = zeros(TF, iterations)
    R_uy = zeros(TF, iterations)
    R_uz = zeros(TF, iterations)
    R_h = zeros(TF, iterations)
    courant = zeros(TF, iterations)
    cellsCourant = KernelAbstractions.zeros(backend, TF, n_cells)

    # Initial calculations
    time = zero(TF) # assuming time = 0

    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, boundaries.U, time, config)
    flux!(mdotf, Uf, config)
    
    # Getting the laplacian of h for first U calculation
    laplacian!(Δh, hf, h, boundaries.h, time, config)
    interpolate!(Δhf, Δh, config)
    correct_boundaries!(Δhf, Δh, internal_BCs.Δh, time, config)

    @. phif.values = mdotf.values * hf. values

    #@info "need to readd Pg term - Coupling term for other phase"
    Pg = 0# Test Pg term set to zero, as the gradient is found this value doesn't matter

    @. nu_h.values = 3*nu.values/h.values
    for i ∈ eachindex(Δhf.values)
        P_hydrf.values[i] = rho.values[1]*hf.values[i]*dot(n,G)
        P_gasf.values[i] = Pg
        P_surff[i] = coeffs.σ*Δhf[i]
        PLf[i] = P_gasf[i] - P_hydrf[i] - P_surff[i]
    end

    grad!(∇PL, PLf, config)

    for i ∈ eachindex(h.values) 
        w[i] = (h.values[i] > coeffs.h_crit)
    end

    grad!(∇w, wf, w, internal_BCs.w, time, config)

    #= BEN: Please check. I think the division by rho is not needed here so I removed it. Can you please check if that's correct? Also this needs t obe a kernel. Although, because this is a simple loop over h.values, you can use xcal_foreach (if you want to see how it's use have a look in RANS_kOmegaLKE.jl, for example)
    =#
    for i ∈ eachindex(h.values)
        Ph_local = (g*sind(coeffs.ϕ)*h[i]) .*plate_tangent_vector
        Ph.x.values[i] = Ph_local[1]
        Ph.y.values[i] = Ph_local[2]
        Ph.z.values[i] = Ph_local[3]

        h∇PL[i] = h[i].*∇PL[i]./rho.values[1]

        τθw[i] = coeffs.β*coeffs.σ/rho.values[1] * (1-cosd(coeffs.θm)) .* ∇w.result[i]
    end

    @info "Starting loops"
    
    progress = Progress(iterations; dt=1.0, showspeed=true)
            
    xdir, ydir, zdir = XDir(), YDir(), ZDir()
    #rh = 0
    rx = ry = rz = 0
    @time for iteration ∈ 1:iterations
        limit_capillary_dt!(config.runtime, coeffs)
        copyto!(dt_cpu, config.runtime.dt)
        time += dt_cpu[1]

        @. h_old = h.values # store previous h before inner loop

        rx, ry, rz = solve_equation!(U_eqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config, time=time)
        
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        remove_film_pressure_source!(U_eqn, P_hydrf, P_surff, rho.values[1], h, config)

        
        for j ∈ eachindex(U) # COMMENT: You can use @ tempU = U this would work on GPU
            tempU[j] = U[j]
        end
        rh = 0
        for i ∈ 1:inner_loops
            H!(Hv, U, U_eqn, config)

            interpolate!(Uf, Hv, config)
            correct_boundaries!(Uf, Hv, boundaries.U, time, config)

            flux!(mdotf, Uf, config)
            @. phif.values = mdotf.values * hf.values

            div!(divPhi, phif, config)

            getDf!(Df, rDf, hf, G, n, config)
            
            @. prev = h.values
            rh = solve_film_h_equation!(h_eqn, h, h_old, boundaries.h, solvers.h, config, time=time)

            if i == inner_loops
                explicit_relaxation!(h, prev, 1.0, config)
            else
                explicit_relaxation!(h, prev, solvers.h.relax, config)
            end 

            limit_h!(h, coeffs.h_floor, config)
            
            laplacian!(Δh, hf, h, boundaries.h, time, config, disp_warn=false)
            interpolate!(Δhf, Δh, config)
            correct_boundaries!(Δhf, Δh, internal_BCs.Δh, time, config)

            for j ∈ eachindex(Δhf.values)
                P_hydrf.values[j] = rho.values[1]*hf.values[j]*dot(n,G)
                P_surff.values[j] = coeffs.σ*Δhf[j]
            end

            correct_film_velocity!(U, Hv, h, P_hydrf, P_surff, rD, rho.values[1], config)
            correct_mass_flux2!(mdotf, Df, h_eqn, config)
        end
        
        @. phif.values = mdotf.values * hf.values 

        @. nu_h.values = 3*nu.values/h.values
        for i ∈ eachindex(Δhf.values)
            P_gasf.values[i] = Pg

            PLf[i] = P_gasf[i] - P_hydrf[i] - P_surff[i]
        end

        grad!(∇PL, PLf, config)

        for i ∈ eachindex(h.values) 
            w[i] = (h.values[i] > coeffs.h_crit)
        end

        # correct U for non-wetted
        for i ∈ eachindex(U.x)
            U[i] = U[i] .* w[i]
        end

        grad!(∇w, wf, w, internal_BCs.w, time, config)

        for i ∈ eachindex(h.values)
            
            P_hydr.values[i] = h.values[i] * dot(n,G)
            Ph_local = (g*sind(coeffs.ϕ)*h[i]) .*plate_tangent_vector
            Ph.x.values[i] = Ph_local[1]
            Ph.y.values[i] = Ph_local[2]
            Ph.z.values[i] = Ph_local[3]

            h∇PL[i] = h[i].*∇PL[i]./(rho.values[1])

            τθw[i] = coeffs.β*coeffs.σ/rho.values[1] * (1-cosd(coeffs.θm)) .* ∇w.result[i]
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
        courant[iteration] = maxCourant
        update_dt!(config.runtime, maxCourant)
        limit_capillary_dt!(config.runtime, coeffs)
        copyto!(dt_cpu, config.runtime.dt)

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

        #runtime_postprocessing!(postprocess, iteration, iterations)

        ∇P_hydr = Grad{Gauss}(P_hydrf)
        grad!(∇P_hydr, P_hydrf, config)
        ∇P_surf = Grad{Gauss}(P_surff)
        grad!(∇P_surf, P_surff, config)

        if iteration % write_interval + signbit(write_interval) == 0

            save_output_film(model, outputWriter, iteration, time, config, w)
            # More verbose output for extra details for debugging
            #save_output_film(model, outputWriter, iteration, time, config, w, Δh, h∇PL, nu_h, Ph, τθw, divPhi, tempU, Hv, ∇P_hydr.result, P_hydr, ∇h.result, ∇P_surf.result)
            save_postprocessing(postprocess, iteration, time, mesh, outputWriter, config.boundaries)
        end

    end # end for loop

    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, h=R_h, courant=courant)
end

function solve_film_h_equation!(
    h_eqn, h, h_old, hBCs, solversetup, config; time=nothing
)
    discretise!(h_eqn, h_old, config)
    apply_boundary_conditions!(h_eqn, hBCs, nothing, time, config)
    setReference!(h_eqn, nothing, 1, config)
    update_preconditioner!(h_eqn.preconditioner, h.mesh, config)
    solve_system!(h_eqn, solversetup, h, nothing, config)
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

function save_output_film(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config, w, Δh, h∇PL, nu_h, Ph, τθw, divPhi, tempU, Hv, ∇P_hydr, P_hydr, ∇h, ∇P_surf
    ) where {T,F,SO,M,Tu,E,D,BI}

    mesh = w.mesh
    Cids = ScalarField(mesh)
    for i ∈ eachindex(mesh.cells)
        Cids[i] = i
    end

    args = (
            ("U", model.momentum.U), 
            ("h", model.momentum.h),
            ("w", w),
            ("Δh", Δh),
            ("h∇PL", h∇PL),
            ("nu_h", nu_h),
            ("Ph", Ph),
            ("τθw", τθw),
            ("divPhi", divPhi),
            ("cID", Cids),
            ("U_temp", tempU),
            ("Hv", Hv),
            ("P_hydr", P_hydr),
            ("∇P_hydr", ∇P_hydr),
            ("∇h", ∇h),
            ("∇P_surf", ∇P_surf)
        )
    
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end

function getDf!(Df, rDf, hf, g, n, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    ndrange = length(hf)
    kernel! = _getDf!(_setup(backend, workgroup, ndrange)...)
    kernel!(Df, rDf, hf, g, n)
end

@kernel function _getDf!(Df, rDf, hf, g, n)
    i = @index(Global)
    (; area) = hf.mesh.faces[i]

    @inbounds begin
        g_n = dot(g, n)
        Df[i] = -rDf[i] * hf[i]  * g_n * area
    end
end

function remove_film_pressure_source!(U_eqn, P_hyrdf, P_surff, rho, h, config)
    
    (; hardware) = config
    (; backend, workgroup) = hardware
    cells = get_phi(U_eqn).mesh.cells
    (; bx, by, bz) = U_eqn.equation

    ∇P_hydr = Grad{Gauss}(P_hyrdf)
    ∇P_surf = Grad{Gauss}(P_surff)

    grad!(∇P_hydr, P_hyrdf, config)
    grad!(∇P_surf, P_surff, config)

    ndrange = length(h)
    kernel! = _remove_film_pressure_source!(_setup(backend, workgroup, ndrange)...)
    kernel!(cells, ∇P_hydr, ∇P_surf, rho, h, bx, by, bz)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _remove_film_pressure_source!(cells,  ∇P_hydr, ∇P_surf, rho, h, bx, by, bz)
    i = @index(Global)

    @uniform begin
        ∇P_hydr_x, ∇P_hydr_y, ∇P_hydr_z = ∇P_hydr.result.x, ∇P_hydr.result.y, ∇P_hydr.result.z
        ∇P_surf_x, ∇P_surf_y, ∇P_surf_z = ∇P_surf.result.x, ∇P_surf.result.y, ∇P_surf.result.z
        _h = h
    end

    @inbounds begin
        hi = _h[i]
        (; volume) = cells[i]
        bx[i] -= hi*(∇P_hydr_x[i] + ∇P_surf_x[i])*volume/rho
        by[i] -= hi*(∇P_hydr_y[i] + ∇P_surf_y[i])*volume/rho
        bz[i] -= hi*(∇P_hydr_z[i] + ∇P_surf_z[i])*volume/rho
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

function limit_capillary_dt!(runtime, coeffs)
    if coeffs.σ > 0 && coeffs.capillary_dt > 0
        runtime.dt[1] = min(runtime.dt[1], coeffs.capillary_dt)
    end
end

function correct_film_velocity!(U, Hv, h, P_hydrf, P_surff, rD, rho, config)
    (; mesh) = U
    (; hardware) = config
    (; backend, workgroup) = hardware

    ∇P_hydr = Grad{Gauss}(P_hydrf)
    grad!(∇P_hydr, P_hydrf, config)
    ∇P_surf = Grad{Gauss}(P_surff)
    grad!(∇P_surf, P_surff, config)

    ndrange = length(U)
    kernel! = _correct_film_velocity!(_setup(backend, workgroup, ndrange)...)
    kernel!(U, Hv, h, rD, ∇P_hydr, ∇P_surf, rho)
end

@kernel function _correct_film_velocity!(U, Hv, h, rD, ∇P_hydr, ∇P_surf, rho)
    i = @index(Global)

    @uniform begin
        Ux, Uy, Uz = U.x, U.y, U.z
        Hvx, Hvy, Hvz = Hv.x, Hv.y, Hv.z
        _h = h
        dPhdx, dPhdy, dPhdz = ∇P_hydr.result.x, ∇P_hydr.result.y, ∇P_hydr.result.z
        dPsdx, dPsdy, dPsdz = ∇P_surf.result.x, ∇P_surf.result.y, ∇P_surf.result.z
        _rD = rD
    end

    @inbounds begin
        rDi = _rD[i]
        hi = _h[i]
        Ux[i] = Hvx[i] + (dPhdx[i] + dPsdx[i]) * hi * rDi/rho
        Uy[i] = Hvy[i] + (dPhdy[i] + dPsdy[i]) * hi * rDi/rho
        Uz[i] = Hvz[i] + (dPhdz[i] + dPsdz[i]) * hi * rDi/rho
    end
end

function correct_mass_flux2!(mdotf, Df, h_eqn, config)
    (; mesh) = mdotf
    (; faces, cells, boundary_cellsID) = mesh
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

    ndrange = n_ifaces
    kernel! = _correct_mass_flux2!(_setup(backend, workgroup, ndrange)...)
    # kernel!(mdotf, h, nzval, colval, rowptr, faces, cells, n_bfaces)
    kernel!(mdotf, h, Df, faces)

end

# @kernel function _correct_mass_flux2!(mdotf, h, nzval, colval, rowptr, faces, cells, n_bfaces)
#     i = @index(Global)
#     fID = i + n_bfaces

#     @inbounds begin
#         face = faces[fID]
#         cID1 = face.ownerCells[1]
#         cID2 = face.ownerCells[2]
#         zID = spindex(rowptr, colval, cID1, cID2)
#         aN = nzval[zID]
#         mdotf[fID] += aN * (h[cID2] - h[cID1])
#     end
# end

@kernel function _correct_mass_flux2!(mdotf, h, Df, faces)
    i = @index(Global)
    fID = i + n_bfaces

    (; ownerCells, delta, area) = faces[fID]

    cID1 = ownerCells[1]
    cID2 = ownerCells[2]
    # DEFINTELY NO NEED FOR cell_nsign here! BUG!
    # snGrad = mesh.cell_nsign[i]*(h[ownerCells[2]]-h[ownerCells[1]])/faces[i].delta
    snGrad = (h[cID2] - h[cID1])/delta
    # len_me = sqrt(normal[1]^2+normal[2]^2+normal[3]^2)# probably 1

    mdotf[fID] -= Df[fID]*area*snGrad
end

# @kernel function _correct_mass_flux2!(mdotf, h, Df, mesh)
#     i = @index(Global)

#     ownerCells = mesh.faces[i].ownerCells
#     # DEFINTELY NO NEED FOR cell_nsign here! BUG!
#     snGrad = mesh.cell_nsign[i]*(h[ownerCells[2]]-h[ownerCells[1]])/mesh.faces[i].delta
#     (; normal) = mesh.faces[i]
#     len_me = sqrt(normal[1]^2+normal[2]^2+normal[3]^2)# probably 1

#     mdotf[i] -= Df[i] * len_me * snGrad
# end
