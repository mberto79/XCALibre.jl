export csimple!

"""
    csimple!(
        model_in, config; 
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    )

Compressible variant of the SIMPLE algorithm with a sensible enthalpy transport equation for the energy. 

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
- `e` Vector of energy residuals for each iteration.

"""
function csimple!(model, config; output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0) 

    residuals = setup_compressible_solvers(
        CSIMPLE, model, config; 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )
    return residuals
end

# Setup for all compressible algorithms
function setup_compressible_solvers(
    solver_variant, model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    ) 

    (; solvers, schemes, runtime, hardware) = config

    @info "Extracting configuration and input fields..."

    # model = adapt(hardware.backend, model_in)
    (; U, p, Uf, pf) = model.momentum
    (; rho) = model.fluid
    mesh = model.domain

    @info "Pre-allocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    rhorDf = FaceScalarField(mesh)
    initialise!(rhorDf, 1.0)
    mueff = FaceScalarField(mesh)
    mueffgradUt = VectorField(mesh)
    divHv = ScalarField(mesh)

    @info "Defining models..."

    U_eqn = (
        Time{schemes.U.time}(rho, U)
        + Divergence{schemes.U.divergence}(mdotf, U) 
        - Laplacian{schemes.U.laplacian}(mueff, U) 
        == 
        - Source(∇p.result)
        + Source(mueffgradUt)
    ) → VectorEquation(U)

    if typeof(model.fluid) <: WeaklyCompressible

        p_eqn = (
            - Laplacian{schemes.p.laplacian}(rhorDf, p) == - Source(divHv)
        ) → ScalarEquation(p)

    elseif typeof(model.fluid) <: Compressible

        pconv = FaceScalarField(mesh)
        p_eqn = (
            Laplacian{schemes.p.laplacian}(rhorDf, p) 
            - Divergence{schemes.p.divergence}(pconv, p) == Source(divHv)
        ) → ScalarEquation(p)

    end

    @info "Initialising preconditioners..."

    @reset U_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, U_eqn, U.BCs, config)
    @reset p_eqn.preconditioner = set_preconditioner(
                    solvers.p.preconditioner, p_eqn, p.BCs, config)

    @info "Pre-allocating solvers..."
     
    @reset U_eqn.solver = solvers.U.solver(_A(U_eqn), _b(U_eqn, XDir()))
    @reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))
  
    @info "Initialising energy model..."
    energyModel = initialise(model.energy, model, mdotf, rho, p_eqn, config)

    @info "Initialising turbulence model..."
    turbulenceModel = initialise(model.turbulence, model, mdotf, p_eqn, config)

    residuals  = solver_variant(
        model, turbulenceModel, energyModel, ∇p, U_eqn, p_eqn, config;
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops)

    return residuals    
end # end function

function CSIMPLE(
    model, turbulenceModel, energyModel, ∇p, U_eqn, p_eqn, config ; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    )
    
    # Extract model variables and configuration
    (; U, p, Uf, pf) = model.momentum
    (; nu, nuf, rho, rhof) = model.fluid

    mesh = model.domain
    p_model = p_eqn.model
    (; solvers, schemes, runtime, hardware) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware
    
    # rho = get_flux(U_eqn, 1)
    mdotf = get_flux(U_eqn, 2)
    mueff = get_flux(U_eqn, 3)
    mueffgradUt = get_source(U_eqn, 2)
    rhorDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)
    if typeof(model.fluid) <: Compressible
        pconv = get_flux(p_eqn, 2)
    end

    @info "Initialise writer (Store mesh in host memory)"

    outputWriter = initialise_writer(output, model.domain)
    
    @info "Allocating working memory..."

    # Define aux fields 
    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    # Uf = FaceVectorField(mesh)
    S = StrainRate(gradU, gradUT, U, Uf)

    n_cells = length(mesh.cells)
    # pf = FaceScalarField(mesh)
    nueff = FaceScalarField(mesh)
    prevpf = FaceScalarField(mesh)
    gradpf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)
    Psi = ScalarField(mesh)
    Psif = FaceScalarField(mesh)

    mugradUTx = FaceScalarField(mesh)
    mugradUTy = FaceScalarField(mesh)
    mugradUTz = FaceScalarField(mesh)

    divmugradUTx = ScalarField(mesh)
    divmugradUTy = ScalarField(mesh)
    divmugradUTz = ScalarField(mesh)

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)
    prev = _convert_array!(prev, backend) 

    # Pre-allocate vectors to hold residuals 
    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_uz = ones(TF, iterations)
    R_p = ones(TF, iterations)
    R_e = ones(TF, iterations)
    
    # Initial calculations
    time = zero(TF) # assuming time=0
    interpolate!(Uf, U, config)   
    correct_boundaries!(Uf, U, U.BCs, time, config) 
    grad!(∇p, pf, p, p.BCs, time, config)
    thermo_Psi!(model, Psi); thermo_Psi!(model, Psif, config);
    @. rho.values = Psi.values * p.values
    @. rhof.values = Psif.values * pf.values
    flux!(mdotf, Uf, rhof, config)

    update_nueff!(nueff, nu, model.turbulence, config)
    @. mueff.values = nueff.values * rhof.values

    @info "Starting CSIMPLE loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    for iteration ∈ 1:iterations
        time = iteration

        ## CHECK GRADU AND EXPLICIT STRESSES
        # grad!(gradU, Uf, U, U.BCs, time, config) # calculated in `turbulence!``

        explicit_shear_stress!(mugradUTx, mugradUTy, mugradUTz, mueff, gradU, config)
        div!(divmugradUTx, mugradUTx, config)
        div!(divmugradUTy, mugradUTy, config)
        div!(divmugradUTz, mugradUTz, config)
        
        @. mueffgradUt.x.values = divmugradUTx.values
        @. mueffgradUt.y.values = divmugradUTy.values
        @. mueffgradUt.z.values = divmugradUTz.values

        # Set up and solve momentum equations
        
        rx, ry, rz = solve_equation!(U_eqn, U, solvers.U, xdir, ydir, zdir, config)
        energy!(energyModel, model, prev, mdotf, rho, mueff, time, config)
        thermo_Psi!(model, Psi); thermo_Psi!(model, Psif, config);

        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rhorDf, rD, config)
        @. rhorDf.values *= rhof.values

        remove_pressure_source!(U_eqn, ∇p, config)
        H!(Hv, U, U_eqn, config)
        
        # Interpolate faces
        interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, U.BCs, time, config)

        if typeof(model.fluid) <: Compressible
            flux!(pconv, Uf, config)
            @. pconv.values *= Psif.values
            flux!(mdotf, Uf, config)
            @. mdotf.values *= rhof.values
            interpolate!(pf, p, config)
            correct_boundaries!(pf, p, p.BCs, time, config)
            @. mdotf.values -= mdotf.values*Psif.values*pf.values/rhof.values
            div!(divHv, mdotf, config)

        elseif typeof(model.fluid) <: WeaklyCompressible
            flux!(mdotf, Uf, config)
            @. mdotf.values *= rhof.values
            div!(divHv, mdotf, config)
        end

        # Pressure calculations
        rp = 0.0
        @. prev = p.values
        @. prevpf.values = pf.values
        if typeof(model.fluid) <: Compressible
            # Ensure diagonal dominance for hyperbolic equations
            rp = solve_equation!(p_eqn, p, solvers.p, config; ref=nothing, irelax=solvers.U.relax)
        elseif typeof(model.fluid) <: WeaklyCompressible
            rp = solve_equation!(p_eqn, p, solvers.p, config; ref=nothing)
        end

        if ~isempty(solvers.p.limit)
            pmin = solvers.p.limit[1]; pmax = solvers.p.limit[2]
            clamp!(p.values, pmin, pmax)
        end

        explicit_relaxation!(p, prev, solvers.p.relax, config)

        grad!(∇p, pf, p, p.BCs, time, config) 
        limit_gradient!(schemes.p.limiter, ∇p, p, config)

        # non-orthogonal correction
        for i ∈ 1:ncorrectors
            discretise!(p_eqn, p, config)       
            apply_boundary_conditions!(p_eqn, p.BCs, nothing, time, config)
            setReference!(p_eqn, pref, 1, config)
            nonorthogonal_face_correction(p_eqn, ∇p, rhorDf, config)
            update_preconditioner!(p_eqn.preconditioner, p.mesh, config)
            rp = solve_system!(p_eqn, solvers.p, p, nothing, config)
            explicit_relaxation!(p, prev, solvers.p.relax, config)
            
            grad!(∇p, pf, p, p.BCs, time, config) 
            limit_gradient!(schemes.p.limiter, ∇p, p, config)
        end

        if typeof(model.fluid) <: Compressible
            rhorelax = 1.0 #0.01
            @. rho.values = rho.values * (1-rhorelax) + Psi.values * p.values * rhorelax
            @. rhof.values = rhof.values * (1-rhorelax) + Psif.values * pf.values * rhorelax
        else
            @. rho.values = Psi.values * p.values
            @. rhof.values = Psif.values * pf.values
        end

        # Velocity and boundaries correction
        # correct_face_interpolation!(pf, p, Uf) # not needed added upwind interpolation
        # correct_boundaries!(pf, p, p.BCs, time, config)
        # pgrad = face_normal_gradient(p, pf)

        if typeof(model.fluid) <: Compressible
            # @. mdotf.values += (pconv.values*(pf.values) - pgrad.values*rhorDf.values)  
            correct_mass_flux(mdotf, p, rhorDf, config)
            @. mdotf.values += pconv.values*(pf.values)
        elseif typeof(model.fluid) <: WeaklyCompressible
            # @. mdotf.values -= pgrad.values*rhorDf.values
            correct_mass_flux(mdotf, p, rhorDf, config)
        end

        correct_velocity!(U, Hv, ∇p, rD, config)
        interpolate!(Uf, U, config)
        correct_boundaries!(Uf, U, U.BCs, time, config)
        
        turbulence!(turbulenceModel, model, S, prev, time, config) 
        update_nueff!(nueff, nu, model.turbulence, config)

        @. mueff.values = rhof.values*nueff.values

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
                turbulenceModel.state.residuals...,
                energyModel.state.residuals
                ]
            )

        if iteration%write_interval + signbit(write_interval) == 0      
            save_output(model, outputWriter, time)
        end

    end # end for loop

    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p, e=R_e)
end


function explicit_shear_stress!(mugradUTx::FaceScalarField, mugradUTy::FaceScalarField, mugradUTz::FaceScalarField, mueff, gradU, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; faces, boundary_cellsID) = mugradUTx.mesh

    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    kernel! = _explicit_shear_stress_internal!(backend, workgroup)
    kernel!(
        mugradUTx, mugradUTy, mugradUTz, mueff, gradU, faces, n_bfaces, ndrange=n_ifaces)
    # KernelAbstractions.synchronize(backend)

    kernel! = _explicit_shear_stress_boundaries!(backend, workgroup)
    kernel!(
        mugradUTx, mugradUTy, mugradUTz, mueff, gradU, faces, ndrange=n_bfaces)
    # KernelAbstractions.synchronize(backend)
end

@kernel function _explicit_shear_stress_internal!(
    mugradUTx, mugradUTy, mugradUTz, mueff, gradU, faces,n_bfaces)
    i = @index(Global)

    fID = i + n_bfaces
    face = faces[fID]
    (; area, normal, ownerCells) = face 
    cID1 = ownerCells[1]
    cID2 = ownerCells[2]
    gradUf = 0.5*(gradU[cID1] + gradU[cID2]) # should this be the transpose of gradU?
    gradUf_projection = gradUf*normal
    trace = 2/3*sum(diag(gradUf))
    mueffi = mueff[fID]
    mugradUTx[fID] = mueffi*(gradUf_projection[1] - trace)*area
    mugradUTy[fID] = mueffi*(gradUf_projection[2] - trace)*area
    mugradUTz[fID] = mueffi*(gradUf_projection[3] - trace)*area
end

@kernel function _explicit_shear_stress_boundaries!(
    mugradUTx, mugradUTy, mugradUTz, mueff, gradU, faces)
    fID = @index(Global)

    face = faces[fID]
    (; area, normal, ownerCells) = face 
    cID1 = ownerCells[1]
    gradUi = gradU[cID1]
    trace = 2/3*sum(diag(gradUi))
    gradUi_projection = gradUi*normal
    mueffi = mueff[fID]
    mugradUTx[fID] = mueffi*(gradUi_projection[1] - trace)*area
    mugradUTy[fID] = mueffi*(gradUi_projection[2] - trace)*area
    mugradUTz[fID] = mueffi*(gradUi_projection[3] - trace)*area
end

######
# CHRIS: Can you please review to make sure it is a faithful reimplementation? Ta!
######

# function explicit_shear_stress!(mugradUTx::FaceScalarField, mugradUTy::FaceScalarField, mugradUTz::FaceScalarField, mueff, gradU)
#     mesh = mugradUTx.mesh
#     (; faces, cells) = mesh
#     nbfaces = length(mesh.boundary_cellsID) #boundary_faces(mesh)
#     start_faceID = nbfaces + 1
#     last_faceID = length(faces)
#     for fID ∈ start_faceID:last_faceID
#         face = faces[fID]
#         (; area, normal, ownerCells, delta) = face 
#         cID1 = ownerCells[1]
#         cID2 = ownerCells[2]

#         ## PREVIOUS IMPLEMENTATION
        
#         # gradUxxf = 0.5*(gradU.result.xx[cID1]+gradU.result.xx[cID2])
#         # gradUxyf = 0.5*(gradU.result.xy[cID1]+gradU.result.xy[cID2])
#         # gradUxzf = 0.5*(gradU.result.xz[cID1]+gradU.result.xz[cID2])
#         # gradUyxf = 0.5*(gradU.result.yx[cID1]+gradU.result.yx[cID2])
#         # gradUyyf = 0.5*(gradU.result.yy[cID1]+gradU.result.yy[cID2])
#         # gradUyzf = 0.5*(gradU.result.yz[cID1]+gradU.result.yz[cID2])
#         # gradUzxf = 0.5*(gradU.result.zx[cID1]+gradU.result.zx[cID2])
#         # gradUzyf = 0.5*(gradU.result.zy[cID1]+gradU.result.zy[cID2])
#         # gradUzzf = 0.5*(gradU.result.zz[cID1]+gradU.result.zz[cID2])
        
#         # mugradUTx[fID] = mueff[fID] * (normal[1]*gradUxxf + normal[2]*gradUyxf + normal[3]*gradUzxf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area
#         # mugradUTy[fID] = mueff[fID] * (normal[1]*gradUxyf + normal[2]*gradUyyf + normal[3]*gradUzyf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area
#         # mugradUTz[fID] = mueff[fID] * (normal[1]*gradUxzf + normal[2]*gradUyzf + normal[3]*gradUzzf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area
        
#         ## NEW IMPLEMENTATION

#         # gradUf = 0.5*(gradU[cID1] + gradU[cID2]) # should this be the transpose of gradU?
#         # gradUf_projection = gradUf*normal
#         # trace = 2/3*sum(diag(gradUf))
#         # mueffi = mueff[fID]
#         # mugradUTx[fID] = mueffi*(gradUf_projection[1] - trace)*area
#         # mugradUTy[fID] = mueffi*(gradUf_projection[2] - trace)*area
#         # mugradUTz[fID] = mueffi*(gradUf_projection[3] - trace)*area
#     end
    
#     # Now deal with boundary faces
#     for fID ∈ 1:nbfaces
#         face = faces[fID]
#         (; area, normal, ownerCells, delta) = face 
#         cID1 = ownerCells[1]
#         cID2 = ownerCells[2]
        
#         ## PREVIOUS IMPLEMENTATION

#         # gradUxxf = (gradU.result.xx[cID1])
#         # gradUxyf = (gradU.result.xy[cID1])
#         # gradUxzf = (gradU.result.xz[cID1])
#         # gradUyxf = (gradU.result.yx[cID1])
#         # gradUyyf = (gradU.result.yy[cID1])
#         # gradUyzf = (gradU.result.yz[cID1])
#         # gradUzxf = (gradU.result.zx[cID1])
#         # gradUzyf = (gradU.result.zy[cID1])
#         # gradUzzf = (gradU.result.zz[cID1])
#         # mugradUTx[fID] = mueff[fID] * (normal[1]*gradUxxf + normal[2]*gradUyxf + normal[3]*gradUzxf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area
#         # mugradUTy[fID] = mueff[fID] * (normal[1]*gradUxyf + normal[2]*gradUyyf + normal[3]*gradUzyf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area
#         # mugradUTz[fID] = mueff[fID] * (normal[1]*gradUxzf + normal[2]*gradUyzf + normal[3]*gradUzzf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area

#         ## NEW IMPLEMENTATION

#         # gradUi = gradU[cID1]
#         # trace = 2/3*sum(diag(gradUi))
#         # gradUi_projection = gradUi*normal
#         # mueffi = mueff[fID]
#         # mugradUTx[fID] = mueffi*(gradUi_projection[1] - trace)*area
#         # mugradUTy[fID] = mueffi*(gradUi_projection[2] - trace)*area
#         # mugradUTz[fID] = mueffi*(gradUi_projection[3] - trace)*area
#     end
# end 