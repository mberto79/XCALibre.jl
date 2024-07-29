export simple_comp!

simple_comp!(model_in, config; resume=true, pref=nothing) = begin
    R_ux, R_uy, R_uz, R_p, R_e, model = setup_compressible_solvers(
        CSIMPLE, model_in, config;
        resume=true, pref=nothing
        )

    return R_ux, R_uy, R_uz, R_p, R_e, model
end

# Setup for all compressible algorithms
function setup_compressible_solvers(
    solver_variant, 
    model_in, config; resume=true, pref=nothing
    ) 

    (; solvers, schemes, runtime, hardware) = config

    @info "Extracting configuration and input fields..."

    model = adapt(hardware.backend, model_in)
    (; U, p) = model.momentum
    mesh = model.domain

    @info "Preallocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    rho= ScalarField(mesh)
    mdotf = FaceScalarField(mesh)
    rhorDf = FaceScalarField(mesh)
    mueff = FaceScalarField(mesh)
    mueffgradUt = VectorField(mesh)
    # initialise!(rDf, 1.0)
    rhorDf.values .= 1.0
    divHv = ScalarField(mesh)

    @info "Defining models..."

    U_eqn = (
        Time{schemes.U.time}(rho, U)
        + Divergence{schemes.U.divergence}(mdotf, U) 
        - Laplacian{schemes.U.laplacian}(mueff, U) 
        == 
        -Source(∇p.result)
        +Source(mueffgradUt)
    ) → VectorEquation(mesh)

    if typeof(model.fluid) <: WeaklyCompressible
        p_eqn = (
            Laplacian{schemes.p.laplacian}(rhorDf, p) == Source(divHv)
        ) → ScalarEquation(mesh)
    elseif typeof(model.fluid) <: Compressible
        pconv = FaceScalarField(mesh)
        p_eqn = (
            Laplacian{schemes.p.laplacian}(rhorDf, p) 
            - Divergence{schemes.p.divergence}(pconv, p) == Source(divHv)
        ) → ScalarEquation(mesh)
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
    energyModel = Energy.initialise(model.energy, model, mdotf, rho, p_eqn, config)

    @info "Initialising turbulence model..."
    turbulenceModel = Turbulence.initialise(model.turbulence, model, mdotf, p_eqn, config)

    R_ux, R_uy, R_uz, R_p, R_e, model  = solver_variant(
    model, turbulenceModel, energyModel, ∇p, U_eqn, p_eqn, config; resume=resume, pref=pref)

    return R_ux, R_uy, R_uz, R_p, R_e, model    
end # end function

function CSIMPLE(
    model, turbulenceModel, energyModel, ∇p, U_eqn, p_eqn, config ; resume, pref)
    
    # Extract model variables and configuration
    (; U, p) = model.momentum
    mu = _mu(model.fluid)
    mesh = model.domain
    p_model = p_eqn.model
    (; solvers, schemes, runtime, hardware) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware
    
    rho = get_flux(U_eqn, 1)
    mdotf = get_flux(U_eqn, 2)
    mueff = get_flux(U_eqn, 3)
    mueffgradUt = get_source(U_eqn, 2)
    rhorDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)
    if typeof(model.fluid) <: Compressible
        pconv = get_flux(p_eqn, 2)
    end
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
    rhof = FaceScalarField(mesh)
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
    interpolate!(Uf, U, config)   
    correct_boundaries!(Uf, U, U.BCs, config)
    grad!(∇p, pf, p, p.BCs, config)
    thermo_Psi!(model, Psi); thermo_Psi!(model, Psif, config);
    @. rho.values = Psi.values * p.values
    @. rhof.values = Psif.values * pf.values
    flux!(mdotf, Uf, rhof, config)
    println("Minrhof ", minimum(rhof.values), ", Maxmrhof ", maximum(rhof.values))

    update_nueff!(mueff, mu, model.turbulence, config)
    
    @info "Staring SIMPLE loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    @time for iteration ∈ 1:iterations

        ## CHECK GRADU AND EXPLICIT STRESSES
        grad!(gradU, Uf, U, U.BCs, config)
        # println(gradU.result)
        # # Set up and solve momentum equations
        explicit_shear_stress!(mugradUTx, mugradUTy, mugradUTz, mueff, gradU)
        div!(divmugradUTx, mugradUTx, config)
        div!(divmugradUTy, mugradUTy, config)
        div!(divmugradUTz, mugradUTz, config)

        @. mueffgradUt.x.values = 0.0#divmugradUTx.values
        @. mueffgradUt.y.values = 0.0#divmugradUTy.values
        @. mueffgradUt.z.values = 0.0#divmugradUTz.values

        solve_equation!(U_eqn, U, solvers.U, xdir, ydir, zdir, config)

        energy!(energyModel, model, prev, mdotf, rho, mueff, config)

        thermo_Psi!(model, Psi); thermo_Psi!(model, Psif, config);

        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rhorDf, rD, config)
        @. rhorDf.values *= rhof.values

        remove_pressure_source!(U_eqn, ∇p, config)
        H!(Hv, U, U_eqn, config)
        
        # Interpolate faces
        interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, Hv, U.BCs, config)

        if typeof(model.fluid) <: Compressible
            flux!(pconv, Uf, config)
            @. pconv.values *= Psif.values
            flux!(mdotf, Uf, config)
            @. mdotf.values *= rhof.values
            interpolate!(pf, p, config)
            correct_boundaries!(pf, p, p.BCs, config)
            @. mdotf.values -= mdotf.values*Psif.values*pf.values/rhof.values
            div!(divHv, mdotf, config)

        elseif typeof(model.fluid) <: WeaklyCompressible
            flux!(mdotf, Uf, config)
            @. mdotf.values *= rhof.values
            div!(divHv, mdotf, config)
        end

        # println("MinrhorDf ", minimum(rhorDf.values), ", MaxPsi ", maximum(rhorDf.values))
        # println("Minpconv ", minimum(pconv.values), ", MaxPsi ", maximum(pconv.values))
        # println("MinPsi ", minimum(divHv.values), ", MaxPsi ", maximum(divHv.values))

        # Pressure calculations
        @. prev = p.values
        if typeof(model.fluid) <: Compressible
            # Ensure diagonal dominance for hyperbolic equations
            # solve_equation!(p_eqn, p, solvers.p, config; ref=nothing)
            solve_equation!(p_eqn, p, solvers.p, config; ref=nothing, irelax=solvers.U.relax)
            pmin = 1000; pmax = 1000000
            clamp!(p.values, pmin, pmax)
        elseif typeof(model.fluid) <: WeaklyCompressible
            solve_equation!(p_eqn, p, solvers.p, config; ref=nothing)
        end
        explicit_relaxation!(p, prev, solvers.p.relax, config)

        residual!(R_ux, U_eqn, U.x, iteration, xdir, config)
        residual!(R_uy, U_eqn, U.y, iteration, ydir, config)
        if typeof(mesh) <: Mesh3
            residual!(R_uz, U_eqn, U.z, iteration, zdir, config)
        end
        residual!(R_p, p_eqn, p, iteration, nothing, config)
        residual!(R_e, energyModel.energy_eqn, model.energy.h, iteration, nothing, config)
        
        # Gradient
        grad!(∇p, pf, p, p.BCs, config) 

        # grad limiter test!
        limit_gradient!(∇p, p, config)

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

        if typeof(model.fluid) <: Compressible
            rhorelax = 0.01
            @. rho.values = rho.values * (1-rhorelax) + Psi.values * p.values * rhorelax
            @. rhof.values = rhof.values * (1-rhorelax) + Psif.values * pf.values * rhorelax
        else
            @. rho.values = Psi.values * p.values
            @. rhof.values = Psif.values * pf.values
        end

        # Velocity and boundaries correction
        correct_face_interpolation!(pf, p, Uf)
        correct_boundaries!(pf, p, p.BCs, config)
        pgrad = face_normal_gradient(p, pf)

        if typeof(model.fluid) <: Compressible
            @. mdotf.values += (pconv.values*pf.values - pgrad.values*rhorDf.values)    
        elseif typeof(model.fluid) <: WeaklyCompressible
            @. mdotf.values -= pgrad.values*rhorDf.values
        end

        correct_velocity!(U, Hv, ∇p, rD, config)
        interpolate!(Uf, U, config)
        correct_boundaries!(Uf, U, U.BCs, config)

        

        # if isturbulent(model)
            grad!(gradU, Uf, U, U.BCs, config)
            turbulence!(turbulenceModel, model, S, S2, prev, config) 
            update_nueff!(mueff, mu, model.turbulence, config)
        # end
        
        convergence = 1e-7

        if (R_ux[iteration] <= convergence && 
            R_uy[iteration] <= convergence && 
            R_uz[iteration] <= convergence &&
            R_p[iteration] <= convergence &&
            R_e[iteration] <= convergence)

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
                (:h, R_e[iteration]),
                ]
            )

        if iteration%write_interval + signbit(write_interval) == 0      
            model2vtk(model, @sprintf "iteration_%.6d" iteration)
        end

    end # end for loop
    model_out = adapt(CPU(), model)
    return R_ux, R_uy, R_uz, R_p, R_e, model_out
end


function correct_face_interpolation!(phif::FaceScalarField, phi, Uf::FaceScalarField)
    mesh = phif.mesh
    (; faces, cells) = mesh
    for fID ∈ eachindex(faces)
        face = faces[fID]
        (; ownerCells, area, normal) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        phi1 = phi[cID1]
        phi2 = phi[cID2]
        flux = Uf[fID]
        if flux >= 0.0
            phif.values[fID] = phi1
        else
            phif.values[fID] = phi2
        end
    end
end

function correct_face_interpolation!(phif::FaceScalarField, phi, Uf)
    mesh = phif.mesh
    (; faces, cells) = mesh
    for fID ∈ eachindex(faces)
        face = faces[fID]
        (; ownerCells, area, normal) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        phi1 = phi[cID1]
        phi2 = phi[cID2]
        flux = area*normal⋅Uf[fID]
        if flux > 0.0
            phif.values[fID] = phi1
        else
            phif.values[fID] = phi2
        end
    end
end

function face_normal_gradient(phi::ScalarField, phif::FaceScalarField)
    mesh = phi.mesh
    sngrad = FaceScalarField(mesh)
    (; faces, cells) = mesh
    nbfaces = length(mesh.boundary_cellsID) #boundary_faces(mesh)
    start_faceID = nbfaces + 1
    last_faceID = length(faces)
    for fID ∈ start_faceID:last_faceID
    # for fID ∈ eachindex(faces)
        face = faces[fID]
        (; area, normal, ownerCells, delta) = face 
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        cell1 = cells[cID1]
        cell2 = cells[cID2]
        phi1 = phi[cID1]
        phi2 = phi[cID2]
        face_grad = area*(phi2 - phi1)/delta
        # face_grad = (phi2 - phi1)/delta
        sngrad.values[fID] = face_grad
    end
    # Now deal with boundary faces
    for fID ∈ 1:nbfaces
        face = faces[fID]
        (; area, normal, ownerCells, delta) = face 
        cID1 = ownerCells[1]
        
        cID2 = ownerCells[2]
        cell1 = cells[cID1]
        cell2 = cells[cID2]
        phi1 = phi[cID1]
        # phi2 = phi[cID2]
        phi2 = phif[fID]
        face_grad = area*(phi2 - phi1)/delta
        # face_grad = (phi2 - phi1)/delta
        sngrad.values[fID] = face_grad
    end
    return sngrad
end

function explicit_shear_stress!(mugradUTx::FaceScalarField, mugradUTy::FaceScalarField, mugradUTz::FaceScalarField, mueff, gradU)
    mesh = mugradUTx.mesh
    (; faces, cells) = mesh
    nbfaces = length(mesh.boundary_cellsID) #boundary_faces(mesh)
    start_faceID = nbfaces + 1
    last_faceID = length(faces)
    for fID ∈ start_faceID:last_faceID
        face = faces[fID]
        (; area, normal, ownerCells, delta) = face 
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        cell1 = cells[cID1]
        cell2 = cells[cID2]
        gradUxxf = 0.5*(gradU.result.xx[cID1]+gradU.result.xx[cID2])
        gradUxyf = 0.5*(gradU.result.xy[cID1]+gradU.result.xy[cID2])
        gradUxzf = 0.5*(gradU.result.xz[cID1]+gradU.result.xz[cID2])
        gradUyxf = 0.5*(gradU.result.yx[cID1]+gradU.result.yx[cID2])
        gradUyyf = 0.5*(gradU.result.yy[cID1]+gradU.result.yy[cID2])
        gradUyzf = 0.5*(gradU.result.yz[cID1]+gradU.result.yz[cID2])
        gradUzxf = 0.5*(gradU.result.zx[cID1]+gradU.result.zx[cID2])
        gradUzyf = 0.5*(gradU.result.zy[cID1]+gradU.result.zy[cID2])
        gradUzzf = 0.5*(gradU.result.zz[cID1]+gradU.result.zz[cID2])
        mugradUTx[fID] = mueff[fID] * (normal[1]*gradUxxf + normal[2]*gradUxyf + normal[3]*gradUxzf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area
        mugradUTy[fID] = mueff[fID] * (normal[1]*gradUyxf + normal[2]*gradUyyf + normal[3]*gradUyzf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area
        mugradUTz[fID] = mueff[fID] * (normal[1]*gradUzxf + normal[2]*gradUzyf + normal[3]*gradUzzf - 0.667 *(gradUxxf + gradUyyf + gradUzzf)) * area
    end
    
    # Now deal with boundary faces
    for fID ∈ 1:nbfaces
        face = faces[fID]
        (; area, normal, ownerCells, delta) = face 
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        cell1 = cells[cID1]
        cell2 = cells[cID2]
        gradUxxf = (gradU.result.xx[cID1])
        gradUxyf = (gradU.result.xy[cID1])
        gradUxzf = (gradU.result.xz[cID1])
        gradUyxf = (gradU.result.yx[cID1])
        gradUyyf = (gradU.result.yy[cID1])
        gradUyzf = (gradU.result.yz[cID1])
        gradUzxf = (gradU.result.zx[cID1])
        gradUzyf = (gradU.result.zy[cID1])
        gradUzzf = (gradU.result.zz[cID1])
        mugradUTx[fID] = mueff[fID] * (normal[1]*gradUxxf + normal[2]*gradUxyf + normal[3]*gradUxzf) * area
        mugradUTy[fID] = mueff[fID] * (normal[1]*gradUyxf + normal[2]*gradUyyf + normal[3]*gradUyzf) * area
        mugradUTz[fID] = mueff[fID] * (normal[1]*gradUzxf + normal[2]*gradUzyf + normal[3]*gradUzzf) * area
    end
end 