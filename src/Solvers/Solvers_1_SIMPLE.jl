export simple!

simple!(model_in, config; resume=true, pref=nothing) = begin
    R_ux, R_uy, R_uz, R_p, model = setup_incompressible_solvers(
        SIMPLE, model_in, config;
        resume=true, pref=nothing
        )

    return R_ux, R_uy, R_uz, R_p, model
end

# Setup for all incompressible algorithms
function setup_incompressible_solvers(
    solver_variant, 
    model_in, config; resume=true, pref=nothing
    ) 

    (; solvers, schemes, runtime, hardware) = config

    @info "Extracting configuration and input fields..."

    # model = adapt(hardware.backend, model_in)
    model = model_in
    (; U, p) = model.momentum
    mesh = model.domain

    @info "Preallocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    rDf = FaceScalarField(mesh)
    nueff = FaceScalarField(mesh)
    # initialise!(rDf, 1.0)
    rDf.values .= 1.0
    divHv = ScalarField(mesh)

    @info "Defining models..."

    U_eqn = (
        Time{schemes.U.time}(U)
        + Divergence{schemes.U.divergence}(mdotf, U) 
        - Laplacian{schemes.U.laplacian}(nueff, U) 
        == 
        -Source(∇p.result)
    ) → VectorEquation(mesh)

    p_eqn = (
        Laplacian{schemes.p.laplacian}(rDf, p) == Source(divHv)
    ) → ScalarEquation(mesh)

    @info "Initialising preconditioners..."

    @reset U_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, U_eqn, U.BCs, config)
    @reset p_eqn.preconditioner = set_preconditioner(
                    solvers.p.preconditioner, p_eqn, p.BCs, config)


    @info "Pre-allocating solvers..."
     
    @reset U_eqn.solver = solvers.U.solver(_A(U_eqn), _b(U_eqn, XDir()))
    @reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))

    @info "Initialising turbulence model..."
    turbulenceModel = Turbulence.initialise(model.turbulence, model, mdotf, p_eqn, config)

    R_ux, R_uy, R_uz, R_p, model  = solver_variant(
    model, turbulenceModel, ∇p, U_eqn, p_eqn, config; resume=resume, pref=pref)

    return R_ux, R_uy, R_uz, R_p, model    
end # end function

function SIMPLE(
    model, turbulenceModel, ∇p, U_eqn, p_eqn, config ; resume, pref)
    
    # Extract model variables and configuration
    (; U, p) = model.momentum
    nu = _nu(model.fluid)
    mesh = model.domain
    p_model = p_eqn.model
    (; solvers, schemes, runtime, hardware) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware
    
    mdotf = get_flux(U_eqn, 2)
    nueff = get_flux(U_eqn, 3)
    rDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)
    
    @info "Allocating working memory..."

    # Define aux fields 
    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT)
    S2 = ScalarField(mesh)

    n_cells = length(mesh.cells)
    Uf = FaceVectorField(mesh)
    pf = FaceScalarField(mesh)
    gradpf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)

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
    interpolate!(Uf, U, config)   
    correct_boundaries!(Uf, U, U.BCs, config)
    flux!(mdotf, Uf, config)
    grad!(∇p, pf, p, p.BCs, config)

    # grad limiter test!
    # limit_gradient!(∇p, p, config)

    update_nueff!(nueff, nu, model.turbulence, config)
    
    @info "Staring SIMPLE loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    @time for iteration ∈ 1:iterations

        solve_equation!(U_eqn, U, solvers.U, xdir, ydir, zdir, config)
        
        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        remove_pressure_source!(U_eqn, ∇p, config)
        H!(Hv, U, U_eqn, config)
        
        # Interpolate faces
        interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, U.BCs, config)

        # div!(divHv, Uf, config) # old approach

        # new approach
        flux!(mdotf, Uf, config)
        div!(divHv, mdotf, config)

        # div!(divHv, Uf, config)
        # flux!(mdotf, Uf, config)
        
        
        # Pressure calculations
        @. prev = p.values
        solve_equation!(p_eqn, p, solvers.p, config; ref=pref)
        explicit_relaxation!(p, prev, solvers.p.relax, config)

        residual!(R_ux, U_eqn, U.x, iteration, xdir, config)
        residual!(R_uy, U_eqn, U.y, iteration, ydir, config)
        if typeof(mesh) <: Mesh3
            residual!(R_uz, U_eqn, U.z, iteration, zdir, config)
        end
        residual!(R_p, p_eqn, p, iteration, nothing, config)
        
        # Gradient
        grad!(∇p, pf, p, p.BCs, config) 

        # grad limiter test
        # limit_gradient!(∇p, p, config)

        # # non-orthogonal correction
        # corr = nonorthogonal_correction(∇p, rDf, config)
        # source_corr = nonorthogonal_source_correction(corr)
        # @. p_eqn.equation.b -= source_corr.values
        # setReference!(p_eqn, pref, 1, config)
        # @. prev = p.values
        # solve_system!(p_eqn, solvers.p, p, nothing, config)
        # explicit_relaxation!(p, prev, solvers.p.relax, config)
        # grad!(∇p, pf, p, p.BCs, config) 

        for _ ∈ 1:1
            discretise!(p_eqn, p, config)       
            apply_boundary_conditions!(p_eqn, p.BCs, nothing, config)
            setReference!(p_eqn, pref, 1, config)
            update_preconditioner!(p_eqn.preconditioner, p.mesh, config)
            nonorthogonal_face_correction(p_eqn, ∇p, rDf, config)
            @. prev = p.values
            solve_system!(p_eqn, solvers.p, p, nothing, config)
            explicit_relaxation!(p, prev, solvers.p.relax, config)
            grad!(∇p, pf, p, p.BCs, config)
        end


        correct = false
        if correct
            ncorrectors = 1
            for i ∈ 1:ncorrectors
                discretise!(p_eqn)
                apply_boundary_conditions!(p_eqn, p.BCs)
                setReference!(p_eqn.equation, pref, 1)
                interpolate!(gradpf, ∇p, p)
                nonorthogonal_flux!(pf, gradpf) # careful: using pf for flux (not interpolation)
                correct!(p_eqn.equation, p_model.terms.term1, pf)
                solve_equation!(p_eqn, p, solvers.p, config; ref=nothing)
                grad!(∇p, pf, p, pBCs) 
            end
        end

        # Velocity and boundaries correction
        correct_velocity!(U, Hv, ∇p, rD, config)
        interpolate!(Uf, U, config)
        correct_boundaries!(Uf, U, U.BCs, config)
        flux!(mdotf, Uf, config) # old approach

        # correct_face_interpolation!(pf, p, Uf) # not needed?
        # correct_boundaries!(pf, p, p.BCs, config) # not needed?

        # pgrad = face_normal_gradient(p, pf)
        # @. mdotf.values -= pgrad.values*rDf.values

        # new approach
        # correct_mass_flux(mdotf, p, pf, rDf, config)

        # if isturbulent(model)
            grad!(gradU, Uf, U, U.BCs, config)
            turbulence!(turbulenceModel, model, S, S2, prev, config) 
            update_nueff!(nueff, nu, model.turbulence, config)
        # end
        
        convergence = 1e-7

        if (R_ux[iteration] <= convergence && 
            R_uy[iteration] <= convergence && 
            R_uz[iteration] <= convergence &&
            R_p[iteration] <= convergence)

            print(
                """
                \n\n\n\n\n
                Simulation converged! $iteration iterations in
                """)
                if !signbit(write_interval)
                    model2vtk(model, @sprintf "iteration_%.6d" iteration)
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
                ]
            )

        if iteration%write_interval + signbit(write_interval) == 0      
            model2vtk(model, @sprintf "iteration_%.6d" iteration)
        end

    end # end for loop
    model_out = adapt(CPU(), model)
    return R_ux, R_uy, R_uz, R_p, model_out
end

### TEMP LOCATION FOR PROTOTYPING - NONORTHOGONAL CORRECTION 

function nonorthogonal_face_correction(eqn, grad, flux, config)
    # nothing # do loop over faces and add contribution to ownerCells in one go! NIIIICE!
    mesh = grad.mesh
    (; faces, boundary_cellsID) = mesh

    (; hardware) = config
    (; backend, workgroup) = hardware

    (; b) = eqn.equation
    
    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    kernel! = _nonorthogonal_face_correction(backend, workgroup)
    kernel!(b, grad, flux, faces, n_bfaces, ndrange=n_ifaces)
    KernelAbstractions.synchronize(backend)
end

@kernel function _nonorthogonal_face_correction(b, grad, flux, faces, n_bfaces)
    i = @index(Global)
    fID = i + n_bfaces
    # for fID ∈ (n_bfaces + 1):n_faces
        face = faces[fID]
        (; ownerCells, weight, area, normal, e) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        gradf = weight*grad[cID1] + (1 - weight)*grad[cID2]
        T_hat = normal - (1/(normal⋅e))*e
        faceCorrection = area*flux[fID]*gradf⋅T_hat

        # correction[cID1] += faceCorrection # remember to make atomic in kernel
        # correction[cID2] += -1*faceCorrection # -1x to correct normal direction

        Atomix.@atomic b[cID1] -= faceCorrection # remember to make atomic in kernel
        Atomix.@atomic b[cID2] -= -1*faceCorrection # -1x to correct normal direction
    # end
    # return correction
end

function nonorthogonal_correction(grad, flux, config)
    mesh = grad.mesh
    (; faces, cells, boundary_cellsID) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces + 1

    corr = FaceScalarField(mesh)

    for fID ∈ (n_bfaces + 1):n_faces
        face = faces[fID]
        (; ownerCells, weight, area, normal, e) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        gradf = weight*grad[cID1] + (1 - weight)*grad[cID2]
        T_hat = normal - (1/(normal⋅e))*e
        corr[fID] = area*flux[fID]*gradf⋅T_hat
    end
    return corr
end

function nonorthogonal_source_correction(corr)
    mesh = corr.mesh
    (; cells, cell_faces, cell_nsign) = mesh

    source_corr = ScalarField(mesh)
    for cID ∈ eachindex(cells)
        cell = cells[cID]
        faces_range = cell.faces_range
        sum = 0.0
        for fi ∈ faces_range
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]
            sum += corr[fID]*nsign
        end
    source_corr[cID] = sum#*cell.volume
    end

    return source_corr
end

### TEMP LOCATION FOR PROTOTYPING

function correct_mass_flux(mdotf, p, pf, rDf, config)
    # sngrad = FaceScalarField(mesh)
    (; faces, cells, boundary_cellsID) = mdotf.mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces + 1

    kernel! = _correct_internal_faces(backend, workgroup)
    kernel!(mdotf, p, rDf, faces, n_bfaces, ndrange=length(n_ifaces))
    KernelAbstractions.synchronize(backend)

    # kernel! = _correct_boundary_faces(backend, workgroup)
    # kernel!(..., ndrange=length(n_bfaces))
    # KernelAbstractions.synchronize(backend)
end

@kernel function _correct_internal_faces(mdotf, p, rDf, faces, n_bfaces)
    i = @index(Global)
    fID = i + n_bfaces

    # for fID ∈ start_faceID:last_faceID
        # for fID ∈ eachindex(faces)
            face = faces[fID]
            (; area, normal, ownerCells, delta) = face 
            cID1 = ownerCells[1]
            cID2 = ownerCells[2]
            # cell1 = cells[cID1]
            # cell2 = cells[cID2]
            p1 = p[cID1]
            p2 = p[cID2]
            # face_grad = area*(p2 - p1)/delta
            face_grad = (p2 - p1)/delta

            # sngrad.values[fID] = face_grad

            # mdotf.values[fID] -= face_grad*rDf.values[fID]
            mdotf[fID] -= face_grad*rDf[fID]
        # end
end

# @kernel function _correct_boundary_faces()
#     i = @index(Global)
# end

# function face_normal_gradient(phi::ScalarField, phif::FaceScalarField)
#     mesh = phi.mesh
#     sngrad = FaceScalarField(mesh)
#     (; faces, cells) = mesh
#     nbfaces = length(mesh.boundary_cellsID) #boundary_faces(mesh)
#     start_faceID = nbfaces + 1
#     last_faceID = length(faces)
#     for fID ∈ start_faceID:last_faceID
#     # for fID ∈ eachindex(faces)
#         face = faces[fID]
#         (; area, normal, ownerCells, delta) = face 
#         cID1 = ownerCells[1]
#         cID2 = ownerCells[2]
#         cell1 = cells[cID1]
#         cell2 = cells[cID2]
#         phi1 = phi[cID1]
#         phi2 = phi[cID2]
#         face_grad = area*(phi2 - phi1)/delta
#         # face_grad = (phi2 - phi1)/delta
#         sngrad.values[fID] = face_grad
#     end
#     # Now deal with boundary faces
#     for fID ∈ 1:nbfaces
#         face = faces[fID]
#         (; area, normal, ownerCells, delta) = face 
#         cID1 = ownerCells[1]
        
#         cID2 = ownerCells[2]
#         cell1 = cells[cID1]
#         cell2 = cells[cID2]
#         phi1 = phi[cID1]
#         # phi2 = phi[cID2]
#         phi2 = phif[fID]
#         face_grad = area*(phi2 - phi1)/delta
#         # face_grad = (phi2 - phi1)/delta
#         sngrad.values[fID] = face_grad
#     end
#     return sngrad
# end