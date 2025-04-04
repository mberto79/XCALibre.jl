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
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    )

    residuals = setup_incompressible_solvers(
        SIMPLE, model, config; 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )

    return residuals
end

# Setup for all incompressible algorithms
function setup_incompressible_solvers(
    solver_variant, model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    ) 

    (; solvers, schemes, runtime, hardware) = config

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

    # periodic = construct_periodic(mesh, hardware.backend, :top, :bottom)
    # periodic_connect = Discretise.periodic_matrix_connectivity(mesh, periodic...)
    # periodic_connect = Discretise.PeriodicConnectivity([],[])

    U_eqn = (
        Time{schemes.U.time}(U)
        + Divergence{schemes.U.divergence}(mdotf, U) 
        - Laplacian{schemes.U.laplacian}(nueff, U) 
        == 
        - Source(∇p.result)
    ) → VectorEquation(U)

    p_eqn = (
        - Laplacian{schemes.p.laplacian}(rDf, p) == - Source(divHv)
    ) → ScalarEquation(p)

    @info "Initialising preconditioners..."

    @reset U_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, U_eqn, U.BCs, config)
    @reset p_eqn.preconditioner = set_preconditioner(
                    solvers.p.preconditioner, p_eqn, p.BCs, config)


    @info "Pre-allocating solvers..."
     
    @reset U_eqn.solver = solvers.U.solver(_A(U_eqn), _b(U_eqn, XDir()))
    @reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))

    @info "Initialising turbulence model..."
    turbulenceModel = initialise(model.turbulence, model, mdotf, p_eqn, config)

    residuals  = solver_variant(
        model, turbulenceModel, ∇p, U_eqn, p_eqn, config; 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops)

    return residuals
end # end function

function SIMPLE(
    model, turbulenceModel, ∇p, U_eqn, p_eqn, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    )
    
    # Extract model variables and configuration
    (; U, p, Uf, pf) = model.momentum
    (; nu) = model.fluid
    mesh = model.domain
    # p_model = p_eqn.model
    (; solvers, schemes, runtime, hardware) = config
    (; iterations, write_interval) = runtime
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
    # Uf = FaceVectorField(mesh)
    S = StrainRate(gradU, gradUT, U, Uf)

    n_cells = length(mesh.cells)
    # pf = FaceScalarField(mesh)
    # gradpf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)

    # Try to assign boundary conditions to rD for use with rDf
    # periodic = construct_periodic(mesh, backend, :top, :bottom)
    # rD = ScalarField(mesh)
    # rD = assign(rD, 
    #     Neumann(:inlet, 0.0),
    #     Neumann(:outlet, 0.0),
    #     Neumann(:bottom, 0.0),
    #     Neumann(:top, 0.0),
    #     Neumann(:plate, 0.0),
    #     periodic...
    # )

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)
    prev = _convert_array!(prev, backend) 

    # Pre-allocate vectors to hold residuals 
    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_uz = ones(TF, iterations)
    R_p = ones(TF, iterations)
    
    # Initial calculations
    time = zero(TF) # assuming time=0
    interpolate!(Uf, U, config)   
    correct_boundaries!(Uf, U, U.BCs, time, config)
    flux!(mdotf, Uf, config)
    grad!(∇p, pf, p, p.BCs, time, config)
    limit_gradient!(schemes.p.limiter, ∇p, p, config)


    update_nueff!(nueff, nu, model.turbulence, config)

    @info "Starting SIMPLE loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    for iteration ∈ 1:iterations
        time = iteration

        rx, ry, rz = solve_equation!(U_eqn, U, solvers.U, xdir, ydir, zdir, config)
        
        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        # correct_boundaries!(rDf, rD, rD.BCs, time, config) # ADDED FOR PERIODIC BCS
        remove_pressure_source!(U_eqn, ∇p, config)
        H!(Hv, U, U_eqn, config)
        
        # Interpolate faces
        interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, U.BCs, time, config)

        # old approach
        # div!(divHv, Uf, config) 

        # new approach
        flux!(mdotf, Uf, config)
        div!(divHv, mdotf, config)
        
        # Pressure calculations
        @. prev = p.values
        rp = solve_equation!(p_eqn, p, solvers.p, config; ref=pref)
        explicit_relaxation!(p, prev, solvers.p.relax, config)
        
        grad!(∇p, pf, p, p.BCs, time, config) 
        limit_gradient!(schemes.p.limiter, ∇p, p, config)

        # non-orthogonal correction
        for i ∈ 1:ncorrectors
            # @. prev = p.values
            discretise!(p_eqn, p, config)       
            apply_boundary_conditions!(p_eqn, p.BCs, nothing, time, config)
            setReference!(p_eqn, pref, 1, config)
            nonorthogonal_face_correction(p_eqn, ∇p, rDf, config)
            update_preconditioner!(p_eqn.preconditioner, p.mesh, config)
            rp = solve_system!(p_eqn, solvers.p, p, nothing, config)
            explicit_relaxation!(p, prev, solvers.p.relax, config)
            grad!(∇p, pf, p, p.BCs, time, config) 
            limit_gradient!(schemes.p.limiter, ∇p, p, config)
        end

        # explicit_relaxation!(p, prev, solvers.p.relax, config)

        # Velocity and boundaries correction

        # old approach
        # correct_velocity!(U, Hv, ∇p, rD, config)
        # interpolate!(Uf, U, config)
        # correct_boundaries!(Uf, U, U.BCs, time, config)
        # flux!(mdotf, Uf, config) 

        # new approach

        # 1. using velocity from momentum equation
        interpolate!(Uf, U, config)
        correct_boundaries!(Uf, U, U.BCs, time, config)
        flux!(mdotf, Uf, config)
        correct_mass_flux(mdotf, p, rDf, config)
        correct_velocity!(U, Hv, ∇p, rD, config)

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
                save_output(model, outputWriter, time)
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

        if iteration%write_interval + signbit(write_interval) == 0      
            save_output(model, outputWriter, time)
        end

    end # end for loop
    
    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p)
end

### TEMP LOCATION FOR PROTOTYPING - NONORTHOGONAL CORRECTION 

function nonorthogonal_face_correction(eqn, grad, flux, config)
    mesh = grad.mesh
    (; faces, cells, boundary_cellsID) = mesh

    (; hardware) = config
    (; backend, workgroup) = hardware

    (; b) = eqn.equation
    
    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    kernel! = _nonorthogonal_face_correction(backend, workgroup)
    kernel!(b, grad, flux, faces, cells, n_bfaces, ndrange=n_ifaces)
    # KernelAbstractions.synchronize(backend)
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
    gradi = weight*grad[cID1] + (1 - weight)*grad[cID2]
    gradf = gradi + ((values[cID2] - values[cID1])/delta - (gradi⋅e))*e
    # gradf = gradi

    Sf = area*normal
    # Ef = ((Sf⋅Sf)/(Sf⋅e))*e # original
    Ef = dPN*(norm(normal)^2/(dPN⋅normal))*area
    T_hat = Sf - Ef # original
    faceCorrection = flux[fID]*gradf⋅T_hat

    Atomix.@atomic b[cID1] += faceCorrection #*cell1.volume
    Atomix.@atomic b[cID2] -= faceCorrection #*cell2.volume # should this be -ve?

    # Atomix.@atomic b[cID1] -= faceCorrection #*cell1.volume
    # Atomix.@atomic b[cID2] += faceCorrection #*cell2.volume # should this be -ve?
        
end

# +- => good match
# -+ => looks worse at edges for gradient
# -- => looks bad on top-right corner for gradient
# ++ => looks bad on left grad and oscillations on the right

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

function correct_mass_flux(mdotf, p, rDf, config)
    # sngrad = FaceScalarField(mesh)
    (; faces, cells, boundary_cellsID) = mdotf.mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces #+ 1

    kernel! = _correct_mass_flux(backend, workgroup)
    kernel!(mdotf, p, rDf, faces, cells, n_bfaces, ndrange=length(n_ifaces))
    # KernelAbstractions.synchronize(backend)
end

@kernel function _correct_mass_flux(mdotf, p, rDf, faces, cells, n_bfaces)
    i = @index(Global)
    fID = i + n_bfaces

    @inbounds begin 
        face = faces[fID]
        (; area, normal, ownerCells, delta) = face 
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        p1 = p[cID1]
        p2 = p[cID2]
        face_grad = area*(p2 - p1)/delta # best option so far!
        mdotf[fID] -= face_grad*rDf[fID]
    end
end