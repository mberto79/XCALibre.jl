export laplace!

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





function laplace!( #HOW ABOUT PASSING k value?
    model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    )

    residuals = setup_laplace_solvers(
        LAPLACE, model, config;
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )

    return residuals
end

# Setup for all incompressible algorithms
function setup_laplace_solvers(
    solver_variant, model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    ) 

    (; solvers, schemes, runtime, hardware, boundaries) = config

    # (; T, Tf) = model.energy
    mesh = model.domain

    rDf = FaceScalarField(mesh) #model.medium.k.values returns 16.2
    k_val = model.medium.k.values
    initialise!(rDf, 1.0/k_val)
    
    zero_field = ScalarField(mesh) #0.0 field
    T_field = model.energy.T #initialised temp field

    @info "Defining models..."


    T_eqn = (
        - Laplacian{schemes.laplacian}(rDf, T_field) #k field faces, T field centres
        ==
        - Source(zero_field) # do I need the -ve ??
    ) → ScalarEquation(T_field, boundaries.T)

    @info "Initialising preconditioners..."

    # @reset U_eqn.preconditioner = set_preconditioner(
    #                 solvers.U.preconditioner, U_eqn, boundaries.U, config)
    # @reset p_eqn.preconditioner = set_preconditioner(
    #                 solvers.p.preconditioner, p_eqn, boundaries.p, config)

    @reset T_eqn.preconditioner = set_preconditioner(solvers.preconditioner, T_eqn)
    # @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    # @reset p_eqn.preconditioner = set_preconditioner(solvers.p.preconditioner, p_eqn)

    @info "Pre-allocating solvers..."

    @reset T_eqn.solver = _workspace(solvers.solver, _b(T_eqn)) # What is this _b doiing????

    # @reset U_eqn.solver = _workspace(solvers.U.solver, _b(U_eqn, XDir()))
    # @reset p_eqn.solver = _workspace(solvers.p.solver, _b(p_eqn))

    @info "Initialising turbulence model... [NO NEED]"
    # turbulenceModel = initialise(model.turbulence, model, mdotf, p_eqn, config) #Do I even need this?

    residuals  = solver_variant(
        model, T_eqn, config;  #Previously: model, turbulenceModel, ∇p, U_eqn, p_eqn, config; 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops)

    return residuals
end # end function

function LAPLACE(
    model, T_eqn, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    )
    
    (; T, Tf) = model.energy
    interpolate!(Tf, T, config)   
    
    mesh = model.domain

    (; solvers, schemes, runtime, hardware, boundaries) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware

    outputWriter = initialise_writer(output, model.domain)
    
    @info "Allocating working memory..."

    # Define aux fields  

    n_cells = length(mesh.cells)
    rDf = FaceScalarField(mesh)
    TF = _get_float(mesh)
    prev = KernelAbstractions.zeros(backend, TF, n_cells) 
    R_T = ones(TF, iterations)
    
    # Initial calculations
    time = zero(TF) # assuming time=0
    # interpolate!(Tf, T, config)   
    # correct_boundaries!(Tf, T, boundaries.T, time, config)

    @info "Starting LAPLACE loops..."
    # println(T.values)

    progress = Progress(iterations; dt=1.0, showspeed=true)

    for iteration ∈ 1:iterations
        time = iteration
        # println(time)

        rt = solve_equation!(T_eqn, T, boundaries.T, solvers, config)
      

        # non-orthogonal correction
        # for i ∈ 1:ncorrectors
        #     # @. prev = p.values
        #     discretise!(p_eqn, p, config)       
        #     apply_boundary_conditions!(p_eqn, boundaries.p, nothing, time, config)
        #     # setReference!(p_eqn, pref, 1, config)
        #     nonorthogonal_face_correction(p_eqn, ∇p, rDf, config)
        #     # update_preconditioner!(p_eqn.preconditioner, p.mesh, config)
        #     rp = solve_system!(p_eqn, solvers.p, p, nothing, config)
        #     explicit_relaxation!(p, prev, solvers.p.relax, config)
        #     grad!(∇p, pf, p, boundaries.p, time, config) 
        #     limit_gradient!(schemes.p.limiter, ∇p, p, config)
        # end

        R_T[iteration] = rt
        # println(T.values)


        if R_T[iteration] <= solvers.convergence
            progress.n = iteration
            finish!(progress)
            @info "Simulation converged in $iteration iterations!"
            if !signbit(write_interval)
                save_output(model, outputWriter, time, config)
            end
            # println(T.values)
            break
        end

        ProgressMeter.next!(
            progress, showvalues = [
                (:iter,iteration),
                (:T_residual, R_T[iteration])
                ]
            )

        if iteration%write_interval + signbit(write_interval) == 0      
            save_output(model, outputWriter, time, config)
        end

    end # end for loop
    
    return (T=R_T)
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

    ndrange = n_ifaces
    kernel! = _nonorthogonal_face_correction(_setup(backend, workgroup, ndrange)...)
    kernel!(b, grad, flux, faces, cells, n_bfaces)
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
    gradi = weight*grad[cID1] + (1.0 - weight)*grad[cID2]
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
    n_ifaces = n_faces - n_bfaces

    ndrange = n_ifaces # length(n_ifaces) was a BUG! should be n_ifaces only!!!!
    kernel! = _correct_mass_flux(_setup(backend, workgroup, ndrange)...)
    kernel!(mdotf, p, rDf, faces, cells, n_bfaces)
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