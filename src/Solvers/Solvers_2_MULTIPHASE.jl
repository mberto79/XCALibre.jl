export multiphase!

"""
    multiphase!(model, config; 
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0)

Multiphase solver for immiscible fluids. Solves coupled momentum, phase fraction (transport), and dynamic pressure equations.
Uses a "p_rgh" pressure formulation to handle gravity and hydrostatic pressure stability.

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
- `alpha` Vector of phase fraction residuals for each iteration.
"""
function multiphase!(
    model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    )

    residuals = setup_multiphase_solvers(
        MULTIPHASE, model, config; 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )

    return residuals
end

function setup_multiphase_solvers(
    solver_variant, model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    ) 

    (; solvers, schemes, runtime, hardware, boundaries) = config

    @info "Extracting configuration and input fields..."

    (; U, p, Uf, pf) = model.momentum
    (; alpha, alphaf, rho, rhof, nu, nuf, p_rgh, p_rghf) = model.fluid

    phases = model.fluid.phases
    props = model.fluid.physics_properties

    backend = hardware.backend
    workgroup = hardware.workgroup
    mesh = model.domain
    isInit = true

    @info "Pre-allocating fields..."

    TF = _get_float(mesh)
    time = zero(TF) # assuming time=0
    
    ∇p = Grad{schemes.p_rgh.gradient}(p)
    
    ∇p_rgh = Grad{schemes.p_rgh.gradient}(p_rgh)
    grad!(∇p_rgh, p_rghf, p_rgh, boundaries.p_rgh, time, config)
    limit_gradient!(schemes.p_rgh.limiter, ∇p_rgh, p_rgh, config)
    
    ∇alpha = Grad{schemes.alpha.gradient}(alpha)
    grad!(∇alpha, alphaf, alpha, boundaries.alpha, time, config)
    limit_gradient!(schemes.alpha.limiter, ∇alpha, alpha, config)

    mdotf = FaceScalarField(mesh)    
    rDf = FaceScalarField(mesh)
    initialise!(rDf, 1.0)
    nueff = FaceScalarField(mesh)
    divHv = ScalarField(mesh)

    phi_g = VectorField(mesh)
    phi_gf = FaceScalarField(mesh)

    interpolate!(alphaf, alpha, config)

    @info "Computing Fluid Properties..."

    phase_eos = [phases[1].eosModel, phases[2].eosModel]

    if typeof(model.energy) <: Nothing # Isothermal
        Temp = ConstantScalar(300.0) # THIS PROBABLY NEEDS TO BE DEFINED BY USER! Redesign Isothermal Energy ?
    else
        Temp = model.energy.T
    end

    update_phase_thermodynamics!(phase_eos[1], Val(1), nueff, Temp, model, config)
    update_phase_thermodynamics!(phase_eos[2], Val(2), nueff, Temp, model, config)

    blend_properties!(rho, alpha, phases[1].rho, phases[2].rho)
    blend_properties!(nu, alpha, phases[1].nu, phases[2].nu)

    interpolate!(rhof, rho, config)
    interpolate!(nuf, nu, config)

    gh = model.fluid.physics_properties.gravity.gh
    ghf = model.fluid.physics_properties.gravity.ghf
    g = model.fluid.physics_properties.gravity.g


    compute_gh!(gh, g, config)
    compute_ghf!(ghf, g, config)
    compute_p_rgh!(p_rgh, gh, p, rho, config)
    compute_p_rghf!(p_rghf, ghf, pf, rhof, config)

    ∇rho = Grad{schemes.p_rgh.gradient}(rho)
    grad!(∇rho, rhof, rho, time, config)
    limit_gradient!(schemes.p_rgh.limiter, ∇rho, rho, config)

    phi_g!(phi_g, gh, ∇rho, config)

    @info "Defining models..."

    # for property in props
    #     update_extra_physics!(property, ∇U, model, config, isInit, 1.0, mesh)
    # end

    zero_field = ConstantScalar(0.0)

    # momentum_rhs = - Src(∇p.result, 1)
    # momentum_rhs = construct_sources(model.momentum, momentum_rhs, model, props, alpha, rho, phases, config, mesh, time)

    U_eqn = (
        Time{schemes.U.time}(rho, U)
        + Divergence{schemes.U.divergence}(mdotf, U) 
        - Laplacian{schemes.U.laplacian}(nueff, U) 
        ==
        - Source(∇p_rgh.result)
        + Source(phi_g)
    ) → VectorEquation(U, boundaries.U)

    # alpha_rhs = - Src(zero_field, 1)
    # alpha_rhs = construct_sources(model.fluid, alpha_rhs, model, props, alpha, rho, phases, config, mesh, time)

    alpha_eqn = (
        Time{schemes.alpha.time}(rho, alpha)
        + Divergence{schemes.alpha.divergence}(mdotf, alpha) 
        == 
        Source(zero_field)
    ) → ScalarEquation(alpha, boundaries.alpha)

    p_eqn = (
        - Laplacian{schemes.p.laplacian}(rDf, p_rgh)
        ==
        - Source(divHv)
    ) → ScalarEquation(p_rgh, boundaries.p_rgh)

    @info "Initialising preconditioners..."

    @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    @reset p_eqn.preconditioner = set_preconditioner(solvers.p_rgh.preconditioner, p_eqn)
    @reset alpha_eqn.preconditioner = set_preconditioner(solvers.alpha.preconditioner, alpha_eqn)

    @info "Pre-allocating solvers..."
     
    @reset U_eqn.solver = _workspace(solvers.U.solver, _b(U_eqn, XDir()))
    @reset p_eqn.solver = _workspace(solvers.p_rgh.solver, _b(p_eqn))
    @reset alpha_eqn.solver = _workspace(solvers.alpha.solver, _b(alpha_eqn))

    @info "Initialising turbulence model..."
    turbulenceModel, config = initialise(model.turbulence, model, mdotf, p_eqn, config)

    residuals  = solver_variant(
        model, turbulenceModel, ∇p, ∇p_rgh, ∇rho, ∇alpha, U_eqn, p_eqn, alpha_eqn, gh, ghf, phi_g, phi_gf, config; 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops)

    return residuals
end # end function



function MULTIPHASE(
    model, turbulenceModel, ∇p, ∇p_rgh, ∇rho, ∇alpha, U_eqn, p_eqn, alpha_eqn, gh, ghf, phi_g, phi_gf, config; 
    # model, ∇p, ∇p_rgh, ∇rho, ∇alpha, U_eqn, p_eqn, alpha_eqn, gh, ghf, phi_g, phi_gf, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=2
    )
    
    (; U, p, Uf, pf) = model.momentum
    (; nu, nuf, rho, rhof, alpha, alphaf, p_rgh, p_rghf) = model.fluid
    mesh = model.domain
    (; solvers, schemes, runtime, hardware, boundaries, postprocess) = config
    (; iterations, write_interval, dt) = runtime
    (; backend) = hardware

    phases = model.fluid.phases
    
    postprocess = convert_time_to_iterations(postprocess,model,dt,iterations)
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
    prev = KernelAbstractions.zeros(backend, TF, n_cells)

    # Pre-allocate vectors to hold residuals 
    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_uz = ones(TF, iterations)
    R_p = ones(TF, iterations)
    R_alpha = ones(TF, iterations)
    cellsCourant = KernelAbstractions.zeros(backend, TF, n_cells)
    
    # Initial calculations
    time = zero(TF) # assuming time=0
    interpolate!(Uf, U, config)   
    correct_boundaries!(Uf, U, boundaries.U, time, config)
    
    flux!(mdotf, Uf, config)

    phase_eos = [phases[1].eosModel, phases[2].eosModel]

    if typeof(model.energy) <: Nothing # Isothermal
        Temp = ConstantScalar(300.0) # THIS PROBABLY NEEDS TO BE DEFINED BY USER! Redesign Isothermal Energy ?
    else
        Temp = model.energy.T
    end

    update_nueff!(nueff, nuf, model.turbulence, config)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    @info "Starting multiphase loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)


    @time for iteration ∈ 1:iterations
        time = iteration *dt

        update_phase_thermodynamics!(phase_eos[1], Val(1), nueff, Temp, model, config)
        update_phase_thermodynamics!(phase_eos[2], Val(2), nueff, Temp, model, config)

        blend_properties!(rho, alpha, phases[1].rho, phases[2].rho)
        blend_properties!(nu, alpha, phases[1].nu, phases[2].nu)

        interpolate!(rhof, rho, config)
        interpolate!(nuf, nu, config)

        grad!(∇rho, rhof, rho, time, config)

        rx, ry, rz = solve_equation!(
            U_eqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config; time=time)


        ∇alpha = Grad{schemes.alpha.gradient}(alpha)
        grad!(∇alpha, alphaf, alpha, boundaries.alpha, time, config)
        limit_gradient!(schemes.alpha.limiter, ∇alpha, alpha, config)        

        ralpha = solve_equation!(alpha_eqn, alpha, boundaries.alpha, solvers.alpha, config; time=time)
        interpolate!(alphaf, alpha, config)
        correct_boundaries!(alphaf, alpha, boundaries.alpha, time, config)
          
        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        remove_pressure_source!(U_eqn, ∇p_rgh, config)
        
        rp = 0.0
        for i ∈ 1:inner_loops
            H!(Hv, U, U_eqn, config)
            
            interpolate!(Uf, Hv, config)
            correct_boundaries!(Uf, Hv, boundaries.U, time, config)

            flux!(mdotf, Uf, config)

            phi_g!(phi_g, gh, ∇rho, config)
            phi_gf!(phi_gf, rho, ghf, rDf, model, config)
            @. mdotf.values += phi_gf.values

            div!(divHv, mdotf, config)
            
            @. prev = p_rgh.values
            rp = solve_equation!(p_eqn, p_rgh, boundaries.p_rgh, solvers.p_rgh, config; ref=pref, time=time)
            if i == inner_loops
                explicit_relaxation!(p_rgh, prev, 1.0, config)
            else
                explicit_relaxation!(p_rgh, prev, solvers.p_rgh.relax, config)
            end

            grad!(∇p_rgh, pf, p_rgh, boundaries.p_rgh, time, config) 
            limit_gradient!(schemes.p_rgh.limiter, ∇p_rgh, p_rgh, config)

            # for i ∈ 1:ncorrectors
            #     discretise!(p_eqn, p_rgh, config)       
            #     apply_boundary_conditions!(p_eqn, boundaries.p, nothing, time, config)
            #     setReference!(p_eqn, pref, 1, config)
            #     nonorthogonal_face_correction(p_eqn, ∇p, rDf, config)
            #     update_preconditioner!(p_eqn.preconditioner, p.mesh, config)
            #     rp = solve_system!(p_eqn, solvers.p, p, nothing, config)

            #     if i == ncorrectors
            #         explicit_relaxation!(p, prev, 1.0, config)
            #     else
            #         explicit_relaxation!(p, prev, solvers.p.relax, config)
            #     end
            #     grad!(∇p, pf, p, boundaries.p, time, config) 
            #     limit_gradient!(schemes.p.limiter, ∇p, p, config)
            # end
            
            correct_mass_flux(mdotf, p_rgh, rDf, config)
            correct_velocity_multiphase!(U, Hv, ∇p_rgh, rD, phi_g, config)

        end # corrector loop end
        
    @. p.values = p_rgh.values + (rho.values * gh.values)

    turbulence!(turbulenceModel, model, S, prev, time, config)
    update_nueff!(nueff, nuf, model.turbulence, config)

    maxCourant = max_courant_number!(cellsCourant, model, config)

    R_ux[iteration] = rx
    R_uy[iteration] = ry
    R_uz[iteration] = rz
    R_p[iteration] = rp
    R_alpha[iteration] = ralpha

    ProgressMeter.next!(
        progress, showvalues = [
            (:time, iteration*runtime.dt),
            (:Courant, maxCourant),
            (:Ux, R_ux[iteration]),
            (:Uy, R_uy[iteration]),
            (:Uz, R_uz[iteration]),
            (:p, R_p[iteration]),
            (:alpha, R_alpha[iteration]),
            # turbulenceModel.state.residuals...
            ]
        )

    runtime_postprocessing!(postprocess,iteration,iterations)
    
    if iteration%write_interval + signbit(write_interval) == 0
        save_output(model, outputWriter, iteration, time, config)
        save_postprocessing(postprocess,iteration,time,mesh,outputWriter,config.boundaries)
    end

    end # end for loop
    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p)
end




function update_phase_thermodynamics!(EoS::AbstractEosModel, phaseIndex::Val{N}, nueff, T, model, config) where {N}
    return nothing
end

function update_phase_thermodynamics!(EoS::Union{ConstEos, PerfectGas}, phaseIndex::Val{N}, nueff, T, model, config) where {N}
    phase = model.fluid.phases[N]
    phase.eosModel(phase, model, config)
    phase.viscosityModel(phase, T)
end

"""
    blend_properties!(property_field, alpha_field, property_0, property_1)

Blends a property between two phases using the phase fraction `alpha_field`.
Formula: `prop = (prop0 * alpha) + (prop1 * (1 - alpha))`
"""
function blend_properties!(property_field, alpha_field, property_0, property_1)
    @. property_field.values = (property_0.values * alpha_field.values) + (property_1.values * (1.0 - alpha_field.values))
    nothing
end

"""
    compute_gh!(gh, g, config)

Computes the dot product of gravity vector and cell centres: `g . x`.
Used for hydrostatic pressure reconstruction.
"""
function compute_gh!(gh, g, config)
    (; hardware) = config
    backend = hardware.backend
    workgroup = hardware.workgroup

    ndrange = length(gh)
    kernel! = _compute_gh!(_setup(backend, workgroup, ndrange)...)
    kernel!(gh, g)
end
@kernel inbounds=true function _compute_gh!(gh, g)
    i = @index(Global)

    gh[i] = dot(g, gh.mesh.cells[i].centre)
end

"""
    compute_ghf!(ghf, g, config)

Computes the dot product of gravity vector and face centres: `g . x_f`.
"""
function compute_ghf!(ghf, g, config)
    (; hardware) = config
    backend = hardware.backend
    workgroup = hardware.workgroup

    ndrange = length(ghf)
    kernel! = _compute_ghf!(_setup(backend, workgroup, ndrange)...)
    kernel!(ghf, g)
end
@kernel inbounds=true function _compute_ghf!(ghf, g)
    i = @index(Global)

    ghf[i] = dot(g, ghf.mesh.faces[i].centre)
end

"""
    compute_p_rgh!(p_rgh, gh, p, rho, config)

Computes dynamic pressure `p_rgh` from absolute pressure `p` and hydrostatic head.
Formula: `p_rgh = p - rho * (g . x)`
"""
function compute_p_rgh!(p_rgh, gh, p, rho, config)
    (; hardware) = config
    backend = hardware.backend
    workgroup = hardware.workgroup

    ndrange = length(p_rgh)
    kernel! = _compute_p_rgh!(_setup(backend, workgroup, ndrange)...)
    kernel!(p_rgh, gh, p, rho)
end
@kernel inbounds=true function _compute_p_rgh!(p_rgh, gh, p, rho)
    i = @index(Global)

    p_rgh[i] = p[i] - (rho[i] * gh[i])
end

"""
    compute_p_rghf!(p_rghf, ghf, pf, rhof, config)

Computes dynamic pressure at faces.
"""
function compute_p_rghf!(p_rghf, ghf, pf, rhof, config)
    (; hardware) = config
    backend = hardware.backend
    workgroup = hardware.workgroup

    ndrange = length(p_rghf)
    kernel! = _compute_p_rghf!(_setup(backend, workgroup, ndrange)...)
    kernel!(p_rghf, ghf, pf, rhof)
end
@kernel inbounds=true function _compute_p_rghf!(p_rghf, ghf, pf, rhof)
    i = @index(Global)

    p_rghf[i] = pf[i] - (rhof[i] * ghf[i])
end

"""
    phi_g!(phi_g, gh, ∇rho, config)

Computes the gravity source term for the momentum equation.
Formula: `phi_g = -(g . x) * ∇rho`
"""
function phi_g!(phi_g, gh, ∇rho, config)
    (; hardware) = config
    backend = hardware.backend
    workgroup = hardware.workgroup

    ndrange = length(phi_g)
    kernel! = _phi_g!(_setup(backend, workgroup, ndrange)...)
    kernel!(phi_g, gh, ∇rho)
end
@kernel inbounds=true function _phi_g!(phi_g, gh, ∇rho)
    i = @index(Global)

    phi_g[i] = -gh[i] * ∇rho.result[i]
end

"""
    phi_gf!(phi_gf, rho, ghf, rDf, model, config)

Computes the gravity contribution to the face flux for pressure-velocity coupling.
"""
function phi_gf!(phi_gf, rho, ghf, rDf, model, config)
    (; faces, cells, boundary_cellsID) = model.domain
    (; hardware) = config
    (; backend, workgroup) = hardware

    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    ndrange = n_ifaces
    kernel! = _phi_gf!(_setup(backend, workgroup, ndrange)...)
    kernel!(phi_gf, rho, ghf, rDf, faces, cells, n_bfaces, model)
end

@kernel function _phi_gf!(phi_gf, rho, ghf, rDf, faces, cells, n_bfaces, model)
    i = @index(Global)
    fID = i + n_bfaces

    @inbounds begin 
        face = faces[fID]
        (; area, normal, ownerCells, delta) = face 
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        rho1 = rho[cID1]
        rho2 = rho[cID2]

        face_grad = area*(rho2 - rho1)/delta

        phi_gf[fID] = -ghf[fID] * face_grad * rDf[fID] 
    end
end

"""
    correct_velocity_multiphase!(U, Hv, ∇p_rgh, rD, phi_g, config)

Corrects the velocity field using the reconstructed pressure gradient and gravity source term.
"""
function correct_velocity_multiphase!(U, Hv, ∇p_rgh, rD, phi_g, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(U)
    kernel! = _correct_velocity_multiphase!(_setup(backend, workgroup, ndrange)...)
    kernel!(U, Hv, ∇p_rgh, rD, phi_g)
end

@kernel function _correct_velocity_multiphase!(U, Hv, ∇p_rgh, rD, phi_g)
    i = @index(Global)

    @uniform begin
        Ux, Uy, Uz = U.x, U.y, U.z
        Hvx, Hvy, Hvz = Hv.x, Hv.y, Hv.z
        dpdx, dpdy, dpdz = ∇p_rgh.result.x, ∇p_rgh.result.y, ∇p_rgh.result.z
        phi_g_x, phi_g_y, phi_g_z = phi_g.x, phi_g.y, phi_g.z
        rDvalues = rD.values
    end

    @inbounds begin
        rD_i = rDvalues[i]

        x_diff = (phi_g_x[i] - dpdx[i])
        y_diff = (phi_g_y[i] - dpdy[i])
        z_diff = (phi_g_z[i] - dpdz[i])

        Ux[i] = Hvx[i] + x_diff * rD_i
        Uy[i] = Hvy[i] + y_diff * rD_i
        Uz[i] = Hvz[i] + z_diff * rD_i
    end
end






# function construct_sources(model_specific, dummy_rhs, model, props, alpha, rho, phases, config, mesh, time)

#     for (i, prop) in enumerate(props)
#         field, sign = update_source(model_specific, prop, model, alpha, rho, phases, config, mesh, time)
#         dummy_rhs += Src(field, sign)
#     end

#     return dummy_rhs
# end

# function update_sources!(model_specific, model, props, eqn, alpha, rho, phases, config, mesh, time)
#     for (i, prop) in enumerate(props)
#         field, sign = update_source(model_specific, prop, model, alpha, rho, phases, config, mesh, time)

#         temp_field = get_source(eqn, i+1)
#         temp_field = field
#     end
# end

# @kernel inbounds=true function _momentum_driftVelocity!(slipVelocityTensor, vdr_p, vdr_q, alpha, rho_p, rho_q)
#     i = @index(Global)

#     cross_p = vdr_p[i] * vdr_p[i]' #vdr_p[i] * transpose(vdr_p[i]) ?
#     cross_q = vdr_q[i] * vdr_q[i]'
#     slipVelocityTensor[i] = (alpha[i] * rho_p[i] * cross_p) + ((1.0-alpha[i]) * rho_q[i] * cross_q) #must be cross products
# end

# function update_extra_physics!(property, ∇U, model, config, isInit, dt, mesh)
#     return nothing
# end
# function update_extra_physics!(property::DriftVelocityState, ∇U, model, config, isInit, dt, mesh)
#     (; U) = model.momentum
#     (; alpha) = model.fluid
#     (; hardware) = config

#     phases = model.fluid.phases
#     props = model.fluid.physics_properties

#     backend = hardware.backend
#     workgroup = hardware.workgroup

#     U_prev      = property.U_prev
#     v_pq        = property.v_pq
#     v_pq_prev   =  property.v_pq_prev

#     if isInit
#         U_prev = U # maybe perturb e.g. * 0.9
#     else
#         U_prev = U
#     end
    
#     a_field = construct_acceleration_field(U, ∇U, U_prev, dt, mesh, props, backend, workgroup)
#     slip_velocity!(alpha, a_field, v_pq, v_pq_prev, mesh, model, phases, config, isInit)

# end

# function slip_velocity!(alpha, a_field, v_pq, v_pq_prev, mesh, model, phases, config, isInitialisation)
#     (; U) = model.momentum

#     rho = model.fluid.rho
#     props = model.fluid.physics_properties

#     d_p = ScalarField(mesh)
#     initialise!(d_p, props.driftVelocity.d_p)

#     # max_U = max_field_value(U)
#     max_U = max_BC_value(config.boundaries.U)


#     find_Vpq!(alpha, rho, d_p, a_field, v_pq_prev, isInitialisation, max_U, v_pq, mesh, phases, config)
    
#     update_velocities!(U, props, alpha, mesh, phases, config, rho) # liquid <> p ; vapour <> q
# end


# function construct_acceleration_field(U_m, ∇U_m, U_m_prev, dt, mesh, props, backend, workgroup) #, workgroup
#     grad_U = ∇U_m.result
#     a = VectorField(mesh)

#     g = props.driftVelocity.gravity.g

#     x0, y0, z0 = g[1], g[2], g[3]

#     x = ScalarField(mesh)
#     y = ScalarField(mesh)
#     z = ScalarField(mesh)
#     initialise!(x, x0)
#     initialise!(y, y0)
#     initialise!(z, z0)
#     G = VectorField(x, y, z, mesh)

#     dUdt = VectorField(mesh)


#     ndrange = length(U_m)
#     kernel! = _construct_acceleration_field!(_setup(backend, workgroup, ndrange)...)
#     kernel!(U_m, dUdt, U_m_prev, dt, a, G, grad_U)
    
#     return a
# end
# @kernel inbounds=true function _construct_acceleration_field!(U_m, dUdt, U_m_prev, dt, a, G, grad_U)
#     i = @index(Global)

#     dUdt[i] = (U_m[i] - U_m_prev[i]) / dt
#     a[i] = G[i] - (grad_U[i] * U_m[i]) - dUdt[i]
# end

# function get_E!(E, alpha, rho, rho_p, rho_q, d_p, mu_p, a_field, backend, workgroup, mesh)
#     # rho = ScalarField(mesh)
#     # blend_properties!(rho, alpha, rho_p, rho_q)

#     ndrange = length(E)
#     kernel! = _get_E!(_setup(backend, workgroup, ndrange)...)
#     kernel!(E, rho_p, d_p, mu_p, rho, a_field)


#     return nothing
# end
# @kernel inbounds=true function _get_E!(E, rho_p, d_p, mu_p, rho, a_field)
#     i = @index(Global)
 
#     E[i] = (rho_p[i]*((d_p[i])^2))/(18*mu_p[i]) * ((rho_p[i]-rho[i])/rho_p[i]) * a_field[i]
# end


# function compute_RE!(RE, rho_q, v_pq, d_p, mu_q, backend, workgroup)
#     ndrange = length(RE)
#     kernel! = _compute_RE!(_setup(backend, workgroup, ndrange)...)
#     kernel!(RE, rho_q, v_pq, d_p, mu_q)
    

#     return nothing
# end
# @kernel inbounds=true function _compute_RE!(RE, rho_q, v_pq, d_p, mu_q)
#     i = @index(Global)
 
#     RE[i] = (rho_q[i] * d_p[i] * norm(v_pq[i]))/mu_q[i]
# end


# function find_Vpq!(alpha, rho, d_p, a_field, v_pq_prev, isInitialisation, v_high, v_pq, mesh, phases, config)
#     v_low = SVector(1.0e-9, 1.0e-9, 1.0e-9)
#     backend = config.hardware.backend
#     workgroup = config.hardware.workgroup

#     rho_p = phases[1].rho
#     rho_q = phases[2].rho

#     mu_p = phases[1].mu
#     mu_q = phases[2].mu


#     E = VectorField(mesh)
#     expr = ScalarField(mesh)
#     v_pq_trial = VectorField(mesh)
#     RE_trial = ScalarField(mesh)

#     get_E!(E, alpha, rho, rho_p, rho_q, d_p, mu_p, a_field, backend, workgroup, mesh)

#     @. expr.values = (mu_q.values*rho_q.values)/(0.0183 * rho_q.values * d_p.values)

#     for i in eachindex(v_pq_trial)
#         v_pq_trial[i] = sqrt.(E[i]*expr.values[i])
#     end

#     compute_RE!(RE_trial, rho_q, v_pq_trial, d_p, mu_q, backend, workgroup)

#     K = ScalarField(mesh)
#     @. K.values = 0.15 * ((rho_q.values * d_p.values)/mu_q.values)^0.687

    
#     ndrange = length(RE_trial)
#     kernel! = _find_Vpq!(_setup(backend, workgroup, ndrange)...)
#     kernel!(RE_trial, v_pq, v_pq_trial, K, E, v_low, v_high, v_pq_prev, isInitialisation)
# end
# @kernel inbounds=true function _find_Vpq!(RE_trial, v_pq, v_pq_trial, K, E, v_low, v_high, v_pq_prev, isInitialisation)
#     i = @index(Global)
 
#     if RE_trial[i] > 1000
#             v_pq[i] = v_pq_trial[i]
#     else
#         if isInitialisation
#             v_pq[i] = Vpq_bisect(K[i], E[i], v_low, v_high)
#         else
#             v_pq[i] = Vpq_newton_raphson(E[i], K[i], v_pq_prev[i])
#         end
#     end
# end

# function Vpq_newton_raphson(E, K, v; max_iter=20, tol=1.0e-7)
#     for it in 1:max_iter # E IS A VECTOR FIELD
#         v_pow_0_687 = v .^ 0.687
#         f = v .+ K .* v_pow_0_687 .* v .- E
#         # f = v + K*(v^1.687) - E

#         f_prime = 1.0 .+ 1.687 .* K .* v_pow_0_687
#         # f_prime = 1 + 1.687 * K * (v^1.687)

#         v_new = v .- f ./ f_prime # division by zero in one direction is possible
#         # v_new = v - (f/f_prime)

#         err = abs.((v_new .- v) ./ (v_new .+ eps())) # Add eps to prevent division by zero
#         # err = abs((v_new - v) / v_new)

#         if maximum(err) < tol
#             v_pq = v_new
#             # println(v_pq)

#             return v_pq
#         end

#         v = v_new
#     end
# end


# function bisect_f(v, K, E)
#     return v .+ K .* (v .^ 1.687) .- E
# end

# function Vpq_bisect(K, E, v_low, v_high; max_iter=50, tol=1.0e-7) #v_mid=SVector(0.0, 0.0, 0.0)
#     for it in 1:max_iter
#         v_mid = (v_low .+ v_high) ./ 2.0
        
#         f_mid = bisect_f(v_mid, K, E)
#         f_low = bisect_f(v_low, K, E)

#         vectorised_check = sign.(f_mid) .== sign.(f_low)
#         v_low = ifelse.(vectorised_check, v_mid, v_low)
#         v_high = ifelse.(vectorised_check, v_high, v_mid)

#         # if sign.(f_mid) == sign.(f_low)
#         #     v_low = v_mid
#         # else
#         #     v_high = v_mid
#         # end

#         err = abs.(v_high .- v_low)

#         if maximum(err) < tol
#             v_pq = v_mid

#             return v_pq
#         end
#     end
# end

# function update_velocities!(U, props, alpha, mesh, phases, config, rho)
#     v_pq = props.driftVelocity.v_pq
    
#     v_dr_p  = props.driftVelocity.v_dr_p
#     v_dr_q  = props.driftVelocity.v_dr_q
#     v_p     = props.driftVelocity.v_p
#     v_q     = props.driftVelocity.v_q

#     backend = config.hardware.backend
#     workgroup = config.hardware.workgroup
#     rho_p = phases[1].rho
#     rho_q = phases[2].rho

#     # rho = ScalarField(mesh)
#     # blend_properties!(rho, alpha, rho_p, rho_q)

#     alpha = alpha.values
#     rho_q = rho_q.values
#     rho_p = rho_p.values
#     rho = rho.values

#     C_p = ScalarField(mesh)
#     C_q = ScalarField(mesh)

#     @. C_p.values = (alpha * rho_p) / rho          #liquid
#     @. C_q.values = ((1.0-alpha) * rho_q) / rho    #vapour

#     ndrange = length(U)
#     kernel! = _update_velocities!(_setup(backend, workgroup, ndrange)...)
#     kernel!(U, v_dr_p, v_dr_q, v_p, v_q, C_p, C_q, v_pq)
# end
# @kernel inbounds=true function _update_velocities!(U, v_dr_p, v_dr_q, v_p, v_q, C_p, C_q, v_pq)
#     i = @index(Global)

#     v_dr_p[i] = (1.0 + C_p[i]) * v_pq[i]
#     v_dr_q[i] = -(1.0 + C_q[i]) * v_pq[i]

#     v_p[i] = v_dr_p[i] - U[i]
#     v_q[i] = v_dr_q[i] - U[i]
# end




# function update_phase_thermodynamics!(EoS::AbstractEosModel, phaseIndex::Val{N}, nueff, T, model, config) where {N}
#     return nothing
# end

# function update_phase_thermodynamics!(EoS::Union{ConstEos, PerfectGas}, phaseIndex::Val{N}, nueff, T, model, config) where {N}
#     phase = model.fluid.phases[N]
#     phase.eosModel(phase, model, config)
#     phase.viscosityModel(phase, T)
# end

# function update_phase_thermodynamics!(EoS::HelmholtzEnergy, phaseIndex::Val{1}, nueff, T, model, config) # New way of dispatching via Val....
#     phase = model.fluid.phases[1]

#     (; p) = model.momentum
#     alpha = model.fluid.alpha

#     HelmholtzModel = phase.eosModel
#     rho_field = phase.rho

#     backend = config.hardware.backend
#     workgroup = config.hardware.workgroup

#     lee_model_state = if hasproperty(model.fluid.physics_properties, :leeModel)
#         model.fluid.physics_properties.leeModel
#     else
#         nothing
#     end
    
#     ndrange = length(rho_field)
#     kernel! = _helmholtz_kernel!(_setup(backend, workgroup, ndrange)...)
#     kernel!(model.fluid, HelmholtzModel, T, p, alpha, lee_model_state)
# end

# function update_phase_thermodynamics!(EoS::PengRobinson, phaseIndex::Val{1}, nueff, T, model, config)
#     phase = model.fluid.phases[1]

#     phase.eosModel(phase, model, config)
#     phase.viscosityModel(phase, T)
# end
# function update_phase_thermodynamics!(EoS::PengRobinson, phaseIndex::Val{2}, nueff, T, model, config)
#     phase = model.fluid.phases[2]

#     phase.viscosityModel(phase, T)
# end

# @kernel inbounds=true function _helmholtz_kernel!(fluid, HelmholtzModel, T, p, alpha, lee_model_state)
#     i = @index(Global)

#     _, rho_temp, _, cp_temp, _, _, _, beta_temp, mu_temp, k_temp, _, latentHeat_temp, _, m_lv_temp, m_vl_temp = HelmholtzModel(T[i], p[i], alpha[i])
            
#     fluid.phases[1].rho[i] = rho_temp[1]
#     fluid.phases[2].rho[i] = rho_temp[2]
            
#     fluid.phases[1].mu[i] = mu_temp[1]
#     fluid.phases[2].mu[i] = mu_temp[2]
            
#     fluid.phases[1].cp[i] = cp_temp[1]
#     fluid.phases[2].cp[i] = cp_temp[2]

#     fluid.phases[1].beta[i] = beta_temp[1]
#     fluid.phases[2].beta[i] = beta_temp[2]

#     fluid.phases[1].k[i] = k_temp[1]
#     fluid.phases[2].k[i] = k_temp[2]

#     _update_lee_model_fields!(lee_model_state, i, latentHeat_temp, m_lv_temp, m_vl_temp) #only works if lee model is there
# end


# @inline function _update_lee_model_fields!(::Nothing, i, latentHeat, m_lv, m_vl) 
#     return nothing
# end

# @inline function _update_lee_model_fields!(lee_model_state::LeeModelState, i, latentHeat, m_lv, m_vl)
#     lee_model_state.latentHeat[i] = latentHeat
#     lee_model_state.m_qp[i] = m_lv
#     lee_model_state.m_pq[i] = m_vl
    
#     return nothing
# end

# # function max_BC_value(BC)
# #     return nothing
# # end
# function max_BC_value(BC) # ::AbstractDirichlet TBD LATER - DOESNT WORK...
#     # example: BC = config.boundaries.U
#     output = SVector(0.0, 0.0, 0.0)

#     for i in eachindex(BC)
#         current_mag = norm(BC[i].value)
#         output_mag = norm(output)

#         if current_mag > output_mag
#             output = BC[i].value
#         end
#     end

#     return output
# end


# function max_field_value(field::ScalarField)
#     return maximum(field.values)
# end

# function max_field_value(field::VectorField)
#     max_x = maximum(field.x.values)
#     max_y = maximum(field.y.values)
#     max_z = maximum(field.z.values)

#     return SVector(max_x, max_y, max_z)
# end