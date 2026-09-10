export simple!

"""
    simple!(model_in, config; 
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0)

Incompressible variant of the SIMPLE algorithm to solving coupled momentum and mass conservation equations.

# Input arguments

- `model` reference to a `Physics` model defined by the user.
- `config` Configuration structure defined by the user with solvers, schemes, runtime and hardware structures configuration details.
- `output` select the format used for simulation results from `VTK()` or `OpenFOAM` (default = `VTK()`)
- `pref` Reference pressure value for cases that do not have a pressure defining BC. Incompressible solvers only (default = `nothing`)
- `ncorrectors` number of non-orthogonality correction loops (default = `0`)
- `inner_loops` number to inner loops used in transient solver based on PISO algorithm (default = `0`)

# Output

This function returns a `NamedTuple` for accessing the residuals (e.g. `residuals.Ux`) with the following entries:

- `Ux` Vector of x-velocity residuals for each iteration.
- `Uy` Vector of y-velocity residuals for each iteration.
- `Uz` Vector of y-velocity residuals for each iteration.
- `p` Vector of pressure residuals for each iteration.

"""
function simple!(
    model, config;
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0, consistent=false
    )

    residuals = setup_incompressible_solvers(
        SIMPLE, model, config;
        output=output,
        pref=pref,
        ncorrectors=ncorrectors,
        inner_loops=inner_loops,
        consistent=consistent
        )

    return residuals
end

# Setup for all incompressible algorithms
function setup_incompressible_solvers(
    solver_variant, model, config;
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0, consistent=false
    )

    (; solvers, schemes, runtime, hardware, boundaries) = config

    @info "Extracting configuration and input fields..."

    (; U, p, Uf, pf) = model.momentum
    mesh = model.domain

    @info "Pre-allocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    rDf = FaceScalarField(mesh)
    initialise!(rDf, 1.0)
    nueff = FaceScalarField(mesh)
    divHv = ScalarField(mesh)

    @info "Defining models..."

    U_eqn = (
        Time{schemes.U.time}(U)
        + Divergence{schemes.U.divergence}(mdotf, U) 
        - Laplacian{schemes.U.laplacian}(nueff, U) 
        == 
        - Source(∇p.result)
    ) → VectorEquation(U, boundaries.U)

    p_eqn = (
        - Laplacian{schemes.p.laplacian}(rDf, p) == - Source(divHv)
    ) → ScalarEquation(p, boundaries.p)

    @info "Initialising preconditioners..."

    @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    @reset p_eqn.preconditioner = set_preconditioner(solvers.p.preconditioner, p_eqn)

    @info "Pre-allocating solvers..."

    @reset U_eqn.solver = _workspace(solvers.U.solver, _b(U_eqn, XDir()))
    @reset p_eqn.solver = _workspace(solvers.p.solver, _b(p_eqn))

    @info "Initialising turbulence model..."
    turbulenceModel, config = initialise(model.turbulence, model, mdotf, p_eqn, config)

    residuals  = solver_variant(
        model, turbulenceModel, ∇p, U_eqn, p_eqn, config;
        output=output,
        pref=pref,
        ncorrectors=ncorrectors,
        inner_loops=inner_loops,
        consistent=consistent)

    return residuals
end # end function

function SIMPLE(
    model, turbulenceModel, ∇p, U_eqn, p_eqn, config;
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0, consistent=false
    )

    if consistent
        @info "SIMPLEC (consistent=true) enabled: pressure-velocity coupling uses the off-diagonal-corrected coefficient rAtU = V/(A_ii - sum|off-diag|), matching OpenFOAM's `simple.consistent()` path in pEqn.H."
    end

    # Extract model variables and configuration
    (; U, p, Uf, pf) = model.momentum
    (; nu) = model.fluid
    mesh = model.domain
    (; solvers, schemes, runtime, hardware, boundaries, postprocess) = config
    (; iterations, write_interval,dt) = runtime
    (; backend) = hardware

    dt_cpu = zeros(_get_float(mesh), 1)
    copyto!(dt_cpu, config.runtime.dt)
    
    postprocess = convert_time_to_iterations(postprocess,model,dt_cpu[1],iterations)
    mdotf = get_flux(U_eqn, 2)
    nueff = get_flux(U_eqn, 3)
    rDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)

    outputWriter = initialise_writer(output, model.domain)
    
    @info "Allocating working memory..."

    # Define aux fields 
    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT, U, Uf)

    n_cells = length(mesh.cells)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)
    sumOff = ScalarField(mesh) # SIMPLEC: sum of |off-diagonal| momentum coefficients per cell
    rAtU = ScalarField(mesh)   # SIMPLEC: V/(A_ii - sumOff), replaces rD when consistent=true

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    prev = KernelAbstractions.zeros(backend, TF, n_cells) 

    # Pre-allocate vectors to hold residuals 
    R_ux = zeros(TF, iterations)
    R_uy = zeros(TF, iterations)
    R_uz = zeros(TF, iterations)
    R_p = zeros(TF, iterations)
    
    # Initial calculations
    time = zero(TF) # assuming time=0
    interpolate!(Uf, U, config)   
    correct_boundaries!(Uf, U, boundaries.U, time, config)
    flux!(mdotf, Uf, config)
    grad!(∇p, pf, p, boundaries.p, time, config)
    limit_gradient!(schemes.p.limiter, ∇p, p, config)

    update_nueff!(nueff, nu, model.turbulence, config)

    @info "Starting SIMPLE loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    for iteration ∈ 1:iterations
        time = iteration

        rx, ry, rz = solve_equation!(U_eqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config)

        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)

        # SIMPLEC: build rAtU = V/(A_ii - sum|off-diag|) from the (already
        # relaxed) momentum matrix, mirroring OpenFOAM's
        # rAtU = 1/(1/rAU - UEqn.H1()) in pEqn.H. Using rAtU instead of rD
        # for the pressure coefficient and velocity correction accounts for
        # neighbour-coupling strength, which plain SIMPLE's rD = V/A_ii
        # ignores -- this is what makes SIMPLEC far more robust on cells
        # with a tiny volume strongly coupled to a much larger neighbour
        # through a skewed-weight face (exactly the sliver-cell case this
        # was added to fix).
        pCoeff = rD
        if consistent
            sum_offdiag!(sumOff, U_eqn, config)
            compute_rAtU!(rAtU, rD, sumOff, config)
            pCoeff = rAtU
        end

        interpolate!(rDf, pCoeff, config)
        correct_interpolation_periodic(rDf, pCoeff, boundaries.U, config)
        remove_pressure_source!(U_eqn, ∇p, config)
        H!(Hv, U, U_eqn, config)

        # Interpolate faces (from the *uncorrected* Hv -- matches OpenFOAM's
        # phiHbyA = fvc::flux(HbyA), computed before the consistent-branch
        # correction is applied to HbyA)
        interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, boundaries.U, time, config)

        # old approach
        # div!(divHv, Uf, config)

        # new approach
        flux!(mdotf, Uf, config)

        if consistent
            # SIMPLEC: correct the face flux directly with an snGrad(p)-based
            # term, matching OpenFOAM's
            # phiHbyA += fvc::interpolate(rAtU()-rAU)*fvc::snGrad(p)*magSf.
            # This uses the same orthogonal-part face geometry (Ef_mag/delta)
            # as the p Laplacian discretisation itself, i.e. a face-normal
            # two-point pressure gradient -- NOT the interpolated cell
            # gradient used by simplec_correct_Hv! below. Building mdotf from
            # the Hv-corrected route alone (as before) implicitly substitutes
            # interpolate(cell_grad(p))·Sf for snGrad(p)*magSf, which only
            # agrees with OpenFOAM's operator on a uniform mesh -- they
            # diverge on a graded mesh, exactly where this matters most.
            simplec_flux_correction!(mdotf, rD, rAtU, p, config)

            # HbyA -= (rAU - rAtU)*grad(p) (using the still-previous-iteration
            # ∇p) -- corrects the *cell* field only, used later purely for
            # the post-solve velocity reconstruction (correct_velocity!), not
            # for building mdotf/divHv above.
            simplec_correct_Hv!(Hv, rD, rAtU, ∇p, config)
        end

        div!(divHv, mdotf, config)
        
        # Pressure calculations
        @. prev = p.values
        rp = solve_equation!(p_eqn, p, boundaries.p, solvers.p, config; ref=pref)
        explicit_relaxation!(p, prev, solvers.p.relax, config)
        
        grad!(∇p, pf, p, boundaries.p, time, config) 
        limit_gradient!(schemes.p.limiter, ∇p, p, config)

        # non-orthogonal correction
        for i ∈ 1:ncorrectors
            # @. prev = p.values
            discretise!(p_eqn, p, config)       
            apply_boundary_conditions!(p_eqn, boundaries.p, nothing, time, config)
            # setReference!(p_eqn, pref, 1, config)
            nonorthogonal_face_correction(p_eqn, ∇p, rDf, config)
            # update_preconditioner!(p_eqn.preconditioner, p.mesh, config)
            rp = solve_system!(p_eqn, solvers.p, p, nothing, config)
            explicit_relaxation!(p, prev, solvers.p.relax, config)
            grad!(∇p, pf, p, boundaries.p, time, config) 
            limit_gradient!(schemes.p.limiter, ∇p, p, config)
        end

        # correct mass flux and velocity
        correct_mass_flux!(mdotf, p_eqn, config; time=time)
        correct_velocity!(U, Hv, ∇p, pCoeff, config)

        turbulence!(turbulenceModel, model, S, prev, time, config)
        update_nueff!(nueff, nu, model.turbulence, config)

        R_ux[iteration] = rx
        R_uy[iteration] = ry
        R_uz[iteration] = rz
        R_p[iteration] = rp

        Uz_convergence = true
        if typeof(mesh) <: Mesh3
            Uz_convergence = rz <= solvers.U.convergence
        end

        if (R_ux[iteration] <= solvers.U.convergence && 
            R_uy[iteration] <= solvers.U.convergence && 
            Uz_convergence &&
            R_p[iteration] <= solvers.p.convergence &&
            turbulenceModel.state.converged)

            progress.n = iteration
            finish!(progress)
            @info "Simulation converged in $iteration iterations!"
            if !signbit(write_interval)
                save_output(model, outputWriter, iteration, time, config)
                save_postprocessing(postprocess,iteration,time,mesh,outputWriter,config.boundaries)
            end
            break
        end

        ProgressMeter.next!(
            progress, showvalues = [
                (:iter,iteration),
                (:Ux, R_ux[iteration]),
                (:Uy, R_uy[iteration]),
                (:Uz, R_uz[iteration]),
                (:p, R_p[iteration]),
                turbulenceModel.state.residuals...
                ]
            )
        
        runtime_postprocessing!(postprocess,iteration,iterations,S,time,config)
        
        if iteration%write_interval + signbit(write_interval) == 0      
            save_output(model, outputWriter, iteration, time, config)
            save_postprocessing(postprocess,iteration,time,mesh,outputWriter,config.boundaries)
        end

    end # end for loop
    
    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p)
end

### SIMPLEC support functions (consistent=true) ###

# sumOff[i] = sum of the (signed) off-diagonal coefficients in row i, negated
# -- matches OpenFOAM's lduMatrix::H1(): H1[cell] -= offDiagCoeff for every
# face. For the standard FVM convection/diffusion assembly (negative
# off-diagonals, positive diagonal) this equals sum(|off-diagonal|).
function sum_offdiag!(sumOff::S, eqn, config) where {S<:ScalarField}
    (; hardware) = config
    (; backend, workgroup) = hardware

    A = _A(eqn)
    nzval = _nzval(A)
    colval = _colval(A)
    rowptr = _rowptr(A)

    ndrange = length(sumOff)
    kernel! = _sum_offdiag!(_setup(backend, workgroup, ndrange)...)
    kernel!(sumOff, nzval, colval, rowptr)
end

@kernel function _sum_offdiag!(sumOff, nzval, colval, rowptr)
    i = @index(Global)
    @uniform values = sumOff.values
    @inbounds begin
        cIndex = spindex(rowptr, colval, i, i)
        start_index = rowptr[i]
        end_index = rowptr[i+1] - 1
        s = zero(eltype(nzval))
        for nzi ∈ start_index:end_index
            if nzi != cIndex
                s -= nzval[nzi]
            end
        end
        values[i] = s
    end
end

# rAtU[i] = V[i] / (A_ii - sumOff[i])  -- SIMPLEC coefficient, replaces
# rD[i] = V[i]/A_ii from plain SIMPLE. Since rD[i] = V[i]/A_ii already,
# A_ii = V[i]/rD[i], avoiding a second sparse-matrix lookup.
function compute_rAtU!(rAtU::S, rD::S, sumOff::S, config) where {S<:ScalarField}
    (; hardware) = config
    (; backend, workgroup) = hardware
    cells = rAtU.mesh.cells

    ndrange = length(rAtU)
    kernel! = _compute_rAtU!(_setup(backend, workgroup, ndrange)...)
    kernel!(rAtU, rD, sumOff, cells)
end

@kernel function _compute_rAtU!(rAtU, rD, sumOff, cells)
    i = @index(Global)
    @uniform begin
        rAtUvals = rAtU.values
        rDvals = rD.values
        sumOffvals = sumOff.values
    end
    @inbounds begin
        (; volume) = cells[i]
        Aii = volume / rDvals[i]
        denom = Aii - sumOffvals[i]
        rAtUvals[i] = volume / denom
    end
end

# HbyA -= (rAU - rAtU)*grad(p)  -- OpenFOAM pEqn.H's SIMPLEC correction to
# HbyA, applied using the still-previous-iteration pressure gradient (∇p is
# not updated again until after the pressure solve later this iteration).
function simplec_correct_Hv!(Hv, rD, rAtU, ∇p, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    ndrange = length(rD)
    kernel! = _simplec_correct_Hv!(_setup(backend, workgroup, ndrange)...)
    kernel!(Hv, rD, rAtU, ∇p)
end

@kernel function _simplec_correct_Hv!(Hv, rD, rAtU, ∇p)
    i = @index(Global)
    @uniform begin
        Hx, Hy, Hz = Hv.x, Hv.y, Hv.z
        dpdx, dpdy, dpdz = ∇p.result.x, ∇p.result.y, ∇p.result.z
        rDvals = rD.values
        rAtUvals = rAtU.values
    end
    @inbounds begin
        delta = rDvals[i] - rAtUvals[i]
        Hx[i] -= delta * dpdx[i]
        Hy[i] -= delta * dpdy[i]
        Hz[i] -= delta * dpdz[i]
    end
end

# SIMPLEC face-flux correction: mdotf += interpolate(rAtU - rD) * snGrad(p) *
# magSf, matching OpenFOAM pEqn.H's
# `phiHbyA += fvc::interpolate(rAtU()-rAU)*fvc::snGrad(p)*mesh.magSf();`.
# snGrad(p)*magSf is built from the same orthogonal-part face geometry
# (Ef_mag/delta, using dPN and the face normal) as the Laplacian(p)
# discretisation in Discretise_1_schemes.jl, so this correction is
# numerically consistent with the operator the pressure equation is actually
# solved with -- unlike interpolating a cell-reconstructed grad(p) to the
# face, which only agrees with this on a uniform mesh.
function simplec_flux_correction!(mdotf, rD, rAtU, p, config)
    mesh = mdotf.mesh
    (; faces, cells, boundary_cellsID) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    ndrange = n_ifaces
    kernel! = _simplec_flux_correction!(_setup(backend, workgroup, ndrange)...)
    kernel!(mdotf, rD, rAtU, p, faces, cells, n_bfaces)
end

@kernel function _simplec_flux_correction!(mdotf, rD, rAtU, p, faces, cells, n_bfaces)
    i = @index(Global)
    @uniform begin
        mvals = mdotf.values
        pvals = p.values
        rDvals = rD.values
        rAtUvals = rAtU.values
    end
    @inbounds begin
        fID = i + n_bfaces
        face = faces[fID]
        (; ownerCells, area, normal, weight) = face
        cID1 = ownerCells[1] # owner
        cID2 = ownerCells[2] # neighbour

        # Ef_mag_over_delta == norm(Ef)/delta, the same orthogonal-part
        # geometric factor the Laplacian(p) scheme uses for its implicit
        # coefficient (Discretise_1_schemes.jl: ap = flux*Ef_mag/delta).
        # Since delta == norm(dPN) (Mesh_1_functions.jl:weight_delta_e), the
        # norm(dPN) factor in Ef_mag cancels against delta here -- do not
        # divide by delta again below.
        dPN = cells[cID2].centre - cells[cID1].centre
        Ef_mag_over_delta = (norm(normal)^2/(dPN⋅normal))*area

        # face.weight is the same owner-side linear-interpolation weight
        # used by interpolate!() elsewhere (Calculate_2_interpolation.jl),
        # kept consistent with how rDf is built for the non-consistent path.
        rDf = weight*rDvals[cID1] + (one(weight) - weight)*rDvals[cID2]
        rAtUf = weight*rAtUvals[cID1] + (one(weight) - weight)*rAtUvals[cID2]

        Atomix.@atomic mvals[fID] += (rAtUf - rDf)*(pvals[cID2] - pvals[cID1])*Ef_mag_over_delta
    end
end

### TEMP LOCATION FOR PROTOTYPING

function nonorthogonal_face_correction(eqn, grad, flux, config)
    mesh = grad.mesh
    (; faces, cells, boundary_cellsID) = mesh

    (; hardware) = config
    (; backend, workgroup) = hardware

    (; b) = eqn.equation
    
    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    ndrange = n_ifaces
    kernel! = _nonorthogonal_face_correction(_setup(backend, workgroup, ndrange)...)
    kernel!(b, grad, flux, faces, cells, n_bfaces)
end

@kernel function _nonorthogonal_face_correction(b, grad, flux, faces, cells, n_bfaces)
    i = @index(Global)
    fID = i + n_bfaces
    face = faces[fID]
    (; ownerCells, area, normal, e, delta) = face
    cID1 = ownerCells[1]
    cID2 = ownerCells[2]
    cell1 = cells[cID1]
    cell2 = cells[cID2]

    xf = face.centre
    xC = cell1.centre
    xN = cell2.centre
    
    # Calculate weights using normal functions
    # weight = norm(xf - xC)/norm(xN - xC)
    # weight = norm(xf - xN)/norm(xN - xC)

    dPN = cell2.centre - cell1.centre

    (; values) = grad.field
    weight, df = correction_weight(cells, faces, fID)
    # weight = face.weight
    gradi = weight*grad[cID1] + (one(weight) - weight)*grad[cID2]
    gradf = gradi + ((values[cID2] - values[cID1])/delta - (gradi⋅e))*e
    # gradf = gradi

    Sf = area*normal
    # Ef = ((Sf⋅Sf)/(Sf⋅e))*e # original
    Ef = dPN*(norm(normal)^2/(dPN⋅normal))*area
    T_hat = Sf - Ef # original
    faceCorrection = flux[fID]*gradf⋅T_hat

    Atomix.@atomic b[cID1] += faceCorrection
    Atomix.@atomic b[cID2] -= faceCorrection 
      
end

function correction_weight(cells, faces, fi)
    (; ownerCells, centre) = faces[fi]
    cID1 = ownerCells[1]
    cID2 = ownerCells[2]
    c1 = cells[cID1].centre
    c2 = cells[cID2].centre
    c1_f = centre - c1
    c1_c2 = c2 - c1
    q = (c1_f⋅c1_c2)/(c1_c2⋅c1_c2)
    f_prime = c1 - q*(c1 - c2)
    w = norm(c2 - f_prime)/norm(c2 - c1)
    df = centre - f_prime
    return w, df
end

### TEMP LOCATION FOR PROTOTYPING

function correct_mass_flux!(mdotf, p_eqn, config; time=nothing)
    # sngrad = FaceScalarField(mesh)
    (; faces, cells, boundary_cellsID) = mdotf.mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    p = p_eqn.model.terms[1].phi
    A = _A(p_eqn)
    nzval = _nzval(A)
    colval = _colval(A)
    rowptr = _rowptr(A)

    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    ndrange = n_ifaces # length(n_ifaces) was a BUG! should be n_ifaces only!!!!
    kernel! = _correct_mass_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(mdotf, p, nzval, colval, rowptr, faces, cells, n_bfaces)
    KernelAbstractions.synchronize(backend)

    BCs = config.boundaries.p
    for BC ∈ BCs
        correct_mass_periodic(
            BC, mdotf, p, nzval, colval, rowptr, cells, faces, backend, workgroup)
        KernelAbstractions.synchronize(backend)
    end

    correct_boundary_mass_flux!(mdotf, p_eqn, BCs, time, config)
end

@kernel function _correct_mass_flux!(
    mdotf, p, nzval, colval, rowptr, faces, cells, n_bfaces)
    i = @index(Global)
    fID = i + n_bfaces

    @inbounds begin 
        face = faces[fID]
        (; area, normal, ownerCells, delta) = face 
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        p1 = p[cID1]
        p2 = p[cID2]
        # need to get aN from sparse system
        zID = spindex(rowptr, colval, cID1, cID2)
        aN = nzval[zID]
        mdotf[fID] += aN*(p2 - p1) # positive because pressure eqn has negative sign
    end
end

### Correct mass flux at periodic boundaries

correct_mass_periodic(arg...) = nothing

function correct_mass_periodic(
    BC::PeriodicParent, mdotf, p, nzval, colval, rowptr, cells, faces, backend, workgroup)
    (; IDs_range, value) = BC
    (; face_map) = value
    ndrange = length(IDs_range)
    kernel! = _correct_mass_periodic(_setup(backend, workgroup, ndrange)...)
    kernel!(mdotf, p, nzval, colval, rowptr, cells, faces, IDs_range, face_map)
end

@kernel function _correct_mass_periodic(
    mdotf, p, nzval, colval, rowptr, cells, faces, IDs_range, face_map)
    i = @index(Global)
    fID = IDs_range[i]
    pfID = face_map[i]

    face = faces[fID]
    pface = faces[pfID]
    cID1 = face.ownerCells[1]
    cID2 = pface.ownerCells[1]

    p1 = p[cID1]
    p2 = p[cID2]
    # need to get aN from sparse system
    zID = spindex(rowptr, colval, cID1, cID2)
    aN = nzval[zID]
    correction = aN*(p2 - p1)
    mdotf[fID] += correction
    mdotf[pfID] = -mdotf[fID] 
    
end

### Correct interpolation at periodic boundaries

function correct_interpolation_periodic(phif, phi, BCs, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    for BC ∈ BCs
        _correct_interpolation_periodic_dispatch(BC, phif, phi, backend, workgroup)
        KernelAbstractions.synchronize(backend)
    end

end

_correct_interpolation_periodic_dispatch(arg...) = nothing

function _correct_interpolation_periodic_dispatch(
    BC::PeriodicParent, phif, phi, backend, workgroup)
    mesh = phif.mesh
    (; cells, faces) = mesh
    (; IDs_range, value) = BC
    (; face_map, transform) = value
    ndrange = length(IDs_range)
    kernel! = _correct_interpolation_periodic(_setup(backend, workgroup, ndrange)...)
    kernel!(phif, phi, cells, faces, IDs_range, face_map, transform)
end

@kernel function _correct_interpolation_periodic(phif, phi, cells, faces, IDs_range, face_map, transform)
    i = @index(Global)
    fID = IDs_range[i]
    pfID = face_map[i]

    face = faces[fID]
    pface = faces[pfID]
    cID = face.ownerCells[1]
    pcID = pface.ownerCells[1]

    phi1 = phi[cID]
    phi2 = phi[pcID]

    w = pface.delta/(face.delta + pface.delta)
    one_w = one(w) - w

    phifi =  w*phi1 + one_w*phi2
    phif[fID] = phifi
    phif[pfID] = phifi
end

### Correct mass flux at pressure boundaries

# Locate the Laplacian term index in an equation's term tuple at compile time. The
# boundary mass-flux correction needs the Laplacian face flux (rhorDf); finding it by
# type keeps correct_mass_flux! independent of term ordering (e.g. the transient p_eqn
# has the Time term first).
@generated function laplacian_term_index(terms)
    for (i, Op) ∈ enumerate(terms.parameters)
        Op <: Operator && Op.parameters[4] <: Laplacian && return :($i)
    end
    error("correct_mass_flux!: no Laplacian term found in the pressure equation")
end

function correct_boundary_mass_flux!(mdotf, p_eqn, BCs, time, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    terms = p_eqn.model.terms
    pterm = terms[laplacian_term_index(terms)]
    p = pterm.phi
    pflux = pterm.flux
    psign = pterm.sign

    (; faces, boundary_cellsID) = mdotf.mesh
    ndrange = length(boundary_cellsID)
    kernel! = _correct_boundary_mass_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(BCs, mdotf, p, pflux, psign, faces, boundary_cellsID, time)
    KernelAbstractions.synchronize(backend)
end

@kernel function _correct_boundary_mass_flux!(
    BCs, mdotf, p, pflux, psign, faces, boundary_cellsID, time)
    fID = @index(Global)

    @inbounds begin
        correct_boundary_mass_flux_dispatch!(
            BCs, mdotf, p, pflux, psign, faces, boundary_cellsID, time, fID)
    end
end

@generated function correct_boundary_mass_flux_dispatch!(
    BCs, mdotf, p, pflux, psign, faces, boundary_cellsID, time, fID)

    calls = Expr(:block)
    for bci ∈ 1:length(BCs.parameters)
        push!(calls.args, quote
            BC = BCs[$bci]
            (; start, stop) = BC.IDs_range
            if start <= fID <= stop
                i = fID - start + 1
                correct_boundary_mass_flux_bc!(
                    BC, mdotf, p, pflux, psign, faces, boundary_cellsID, time, i, fID)
                return nothing
            end
        end)
    end

    return calls
end

@inline correct_boundary_mass_flux_bc!(
    BC, mdotf, p, pflux, psign, faces, boundary_cellsID, time, i, fID) = nothing

@inline function correct_boundary_mass_flux_bc!(
    BC::Dirichlet, mdotf, p, pflux, psign, faces, boundary_cellsID, time, i, fID)
    @inbounds begin
        face = faces[fID]
        cID = boundary_cellsID[fID]
        (; area, delta) = face
        flux = pflux[fID] * area / delta
        ap = psign * (-flux)
        mdotf[fID] += ap * (p[cID] - BC.value)
    end
    nothing
end

@inline function correct_boundary_mass_flux_bc!(
    BC::DirichletFunction, mdotf, p, pflux, psign, faces, boundary_cellsID, time, i, fID)
    @inbounds begin
        face = faces[fID]
        cID = boundary_cellsID[fID]
        (; area, delta, centre) = face
        flux = pflux[fID] * area / delta
        ap = psign * (-flux)
        value = BC.value(centre, time, i)
        mdotf[fID] += ap * (p[cID] - value)
    end
    nothing
end

@inline function correct_boundary_mass_flux_bc!(
    BC::Neumann, mdotf, p, pflux, psign, faces, boundary_cellsID, time, i, fID)
    @inbounds begin
        face = faces[fID]
        mdotf[fID] -= pflux[fID] * face.area * BC.value
    end
    nothing
end
