export simple_rho_K_transonic!

# Employing the compressible simple format outlined Moukalled et al. - The finite volume method in computational fluid dynamics

function simple_rho_K_transonic!(model, thermodel, config; resume=true, pref=nothing) 

    @info "Extracting configuration and input fields..."
    (; U, p, energy, nu, mesh) = model
    (; solvers, schemes, runtime) = config

    (; rho) = thermodel

    transonic = false

    @info "Preallocating fields..."
    
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    pconv = FaceScalarField(mesh)
    psi = ScalarField(mesh)
    # nuf = ConstantScalar(nu) # Implement constant field!
    rhorDf = FaceScalarField(mesh)
    mueff = FaceScalarField(mesh)
    keff_by_cp = FaceScalarField(mesh)
    initialise!(rhorDf, 1.0)
    momx_source = ScalarField(mesh)
    momy_source = ScalarField(mesh)
    momz_source = ScalarField(mesh)
    p_source = ScalarField(mesh)
    energy_source = ScalarField(mesh)
    initialise!(rho, 1.0)
    divK = ScalarField(mesh)
    divU = ScalarField(mesh)
    gradDivU = Grad{schemes.U.gradient}(divU)

    @info "Defining models..."

    ux_eqn = (
        Time{schemes.U.time}(rho, U.x)
        + Divergence{schemes.U.divergence}(mdotf, U.x) 
        - Laplacian{schemes.U.laplacian}(mueff, U.x) 
        == 
        -Source(∇p.result.x) + Source(momx_source)
    ) → Equation(mesh)
    
    uy_eqn = (
        Time{schemes.U.time}(rho, U.y)
        + Divergence{schemes.U.divergence}(mdotf, U.y) 
        - Laplacian{schemes.U.laplacian}(mueff, U.y) 
        == 
        -Source(∇p.result.y) + Source(momy_source)
    ) → Equation(mesh)

    uz_eqn = (
        Time{schemes.U.time}(rho, U.z)
        + Divergence{schemes.U.divergence}(mdotf, U.z) 
        - Laplacian{schemes.U.laplacian}(mueff, U.z) 
        == 
        -Source(∇p.result.z) + Source(momz_source)
    ) → Equation(mesh)

    p_eqn = (
        Time{schemes.p.time}(psi, p)
        + Divergence{schemes.p.divergence}(pconv, p)
        - Laplacian{schemes.p.laplacian}(rhorDf, p) 
        == Source(p_source)
    ) → Equation(mesh)

    # Actually using enthalpy -> energy = cp * T
    energy_eqn = (
        Time{schemes.energy.time}(rho, energy)
        + Divergence{schemes.energy.divergence}(mdotf, energy) 
        - Laplacian{schemes.energy.laplacian}(keff_by_cp, energy) 
        == 
        -Source(energy_source)
    ) → Equation(mesh)

    @info "Initialising preconditioners..."
    
    @reset ux_eqn.preconditioner = set_preconditioner(
                    solvers.U.preconditioner, ux_eqn, U.x.BCs, runtime)
    @reset uy_eqn.preconditioner = ux_eqn.preconditioner
    @reset uz_eqn.preconditioner = ux_eqn.preconditioner
    @reset p_eqn.preconditioner = set_preconditioner(
                    solvers.p.preconditioner, p_eqn, p.BCs, runtime)
    @reset energy_eqn.preconditioner = set_preconditioner(
                    solvers.energy.preconditioner, energy_eqn, energy.BCs, runtime)

    @info "Pre-allocating solvers..."
     
    @reset ux_eqn.solver = solvers.U.solver(_A(ux_eqn), _b(ux_eqn))
    @reset uy_eqn.solver = solvers.U.solver(_A(uy_eqn), _b(uy_eqn))
    @reset uz_eqn.solver = solvers.U.solver(_A(uz_eqn), _b(uz_eqn))
    @reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))
    @reset energy_eqn.solver = solvers.energy.solver(_A(energy_eqn), _b(energy_eqn))

    if isturbulent(model)
        @info "Initialising turbulence model..."
        turbulence = initialise_RANS(mdotf, p_eqn, config, model)
        config = turbulence.config
    else
        turbulence = nothing
    end

    R_ux, R_uy, R_uz, R_p, R_e = SIMPLE_RHO_K_transonic_loop(
    model, thermodel, ∇p, gradDivU, ux_eqn, uy_eqn, uz_eqn, p_eqn, energy_eqn, turbulence, transonic, config ; resume=resume, pref=pref)

    return R_ux, R_uy, R_uz, R_p, R_e     
end # end function

function SIMPLE_RHO_K_transonic_loop(
    model, thermodel, ∇p, gradDivU, ux_eqn, uy_eqn, uz_eqn, p_eqn, energy_eqn, turbulence, transonic, config ; resume, pref)
    
    # Extract model variables and configuration
    (;mesh, U, p, energy, nu) = model
    (; rho, rhof, Psi, Psif) = thermodel
    # ux_model, uy_model = ux_eqn.model, uy_eqn.model
    p_model = p_eqn.model
    (; solvers, schemes, runtime) = config
    (; iterations, write_interval) = runtime

    # Need to replace this with ThermoModel
    R = 287.0
    Cp = 1005.0
    Pr = 0.7
    
    rho = get_flux(ux_eqn, 1)
    mdotf = get_flux(ux_eqn, 2)
    mueff = get_flux(ux_eqn, 3)
    momx_source = get_source(ux_eqn, 2)
    momy_source = get_source(uy_eqn, 2)
    momz_source = get_source(uz_eqn, 2)
    psi = get_flux(p_eqn, 1)
    pconv = get_flux(p_eqn, 2)
    rhorDf = get_flux(p_eqn, 3)
    p_source = get_source(p_eqn, 1)
    keff_by_cp = get_flux(energy_eqn, 3)
    energy_source = get_source(energy_eqn, 1)
    
    @info "Allocating working memory..."

    # Define aux fields 
    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT)
    S2 = ScalarField(mesh)
    # ∇U = Grad{schemes.U.gradient}(U)
    divU = gradDivU.field
    divUf = FaceScalarField(mesh)
    # gradDivU = Grad{schemes.U.gradient}(divU)

    # Temp sources to test GradUT explicit source
    # divUTx = zeros(Float64, length(mesh.cells))
    # divUTy = zeros(Float64, length(mesh.cells))

    n_cells = length(mesh.cells)
    Uf = FaceVectorField(mesh)
    pf = FaceScalarField(mesh)
    energyf = FaceScalarField(mesh)
    # rhof = FaceScalarField(mesh)
    rDf = FaceScalarField(mesh)
    gradpf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    rD = ScalarField(mesh)
    rhorD = ScalarField(mesh)
    # Kf = FaceVectorField(mesh)
    pfdash = FaceScalarField(mesh)
    pdash = ScalarField(mesh)
    Psi = ScalarField(mesh)
    Psif = FaceScalarField(mesh)
    phiKf = FaceVectorField(mesh)
    prevmdotf = FaceScalarField(mesh)

    mugradUTx = FaceScalarField(mesh)
    mugradUTy = FaceScalarField(mesh)
    mugradUTz = FaceScalarField(mesh)

    divmugradUTx = ScalarField(mesh)
    divmugradUTy = ScalarField(mesh)
    divmugradUTz = ScalarField(mesh)

    # Pre-allocate auxiliary variables

    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)  

    # Pre-allocate vectors to hold residuals 

    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_uz = ones(TF, iterations)
    R_p = ones(TF, iterations)
    R_e = ones(TF, iterations)

    # INITIAL GUESS OF mdot and thermodynamic information

    interpolate!(Uf, U)   
    correct_boundaries!(Uf, U, U.BCs)
    interpolate!(energyf, energy)   
    correct_boundaries!(energyf, energy, energy.BCs)

    @. Psi.values = Cp/(R*energy.values)
    @. Psif.values = Cp/(R*energyf.values)

    interpolate!(pf, p)   
    correct_boundaries!(pf, p, p.BCs)

    @. rho.values = p.values*Psi.values
    @. rhof.values = pf.values*Psif.values

    flux!(mdotf, Uf, rhof)

    grad!(∇p, pf, p, p.BCs)

    update_nueff!(mueff, nu, rhof, turbulence)
    @. mueff.values = 0.001

    # Calculate keff_by_cp
    @. keff_by_cp.values = mueff.values/Pr

    volumes = getproperty.(mesh.cells, :volume)

    @info "Staring SIMPLE loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations
        
        # # Assemble and solve Momentum equations
        # explicit_shear_stress!(mugradUTx, mugradUTy, mugradUTz, mueff, gradU)
        # divnovol!(divmugradUTx, mugradUTx)
        # divnovol!(divmugradUTy, mugradUTx)
        # divnovol!(divmugradUTz, mugradUTz)

        # @. momx_source.values = divmugradUTx
        # @. momy_source.values = divmugradUTy
        # @. momz_source.values = divmugradUTz

        discretise!(ux_eqn, prev, runtime)
        apply_boundary_conditions!(ux_eqn, U.x.BCs)
        discretise!(uy_eqn, prev, runtime)
        apply_boundary_conditions!(uy_eqn, U.y.BCs)
        wallBC!(ux_eqn, uy_eqn, U, mesh, mueff)
        @. ux_eqn.equation.b += divmugradUTx.values
        @. uy_eqn.equation.b += divmugradUTy.values
        
        @. prev = U.x.values
        # ux_eqn.b .-= divUTx
        implicit_relaxation!(ux_eqn.equation, prev, solvers.U.relax)
        update_preconditioner!(ux_eqn.preconditioner)
        run!(ux_eqn, solvers.U) #opP=Pu.P, solver=solver_U)
        residual!(R_ux, ux_eqn.equation, U.x, iteration)

        @. prev = U.y.values        
        # uy_eqn.b .-= divUTy
        implicit_relaxation!(uy_eqn.equation, prev, solvers.U.relax)
        update_preconditioner!(uy_eqn.preconditioner)
        run!(uy_eqn, solvers.U)
        residual!(R_uy, uy_eqn.equation, U.y, iteration)

        if typeof(mesh) <: Mesh3
            @. prev = U.z.values
            discretise!(uz_eqn, prev, runtime)
            apply_boundary_conditions!(uz_eqn, U.z.BCs)
            # uy_eqn.b .-= divUTy
            implicit_relaxation!(uz_eqn.equation, prev, solvers.U.relax)
            update_preconditioner!(uz_eqn.preconditioner)
            run!(uz_eqn, solvers.U)
            residual!(R_uz, uz_eqn.equation, U.z, iteration)
        end

        # Calculate rho*
        interpolate!(energyf, energy)   
        correct_boundaries!(energyf, energy, energy.BCs)

        @. Psi.values = Cp/(R*energy.values)
        @. Psif.values = Cp/(R*energyf.values)

        interpolate!(pf, p)   
        correct_boundaries!(pf, p, p.BCs)

        @. rho.values = p.values*Psi.values
        @. rhof.values = pf.values*Psif.values

        # Calculate m* with Rhie-Chow
        remove_pressure_source!(ux_eqn, uy_eqn, uz_eqn, ∇p)
        H!(Hv, U, ux_eqn, uy_eqn, uz_eqn)

        interpolate!(Uf, U) # Careful: reusing Uf for interpolation
        correct_boundaries!(Uf, U, U.BCs)
        flux!(mdotf, Uf, rhof)
        
        inverse_diagonal!(rD, ux_eqn.equation, uy_eqn.equation, uz_eqn.equation)
        # inverse_diagonal!(rD, ux_eqn.equation)
        interpolate!(rhof, rho)
        interpolate!(rDf, rD)
        @. rhorDf.values = rDf.values * rhof.values

        rhie_chow_correction!(mdotf, rhorDf, p, ∇p)
        
        pgrad = face_normal_gradient(p, pf)
        @. pgrad.values *= rhorDf.values
        # @. mdotf.values -= pgrad.values
        @. pconv.values = (mdotf.values/rhof.values)*Psif.values

        divnovol!(p_source, pgrad)

        println("Min h: ", minimum(energy.values))
        println("Min hf: ", minimum(energyf.values))

        # Solve pressure equation
        @. prev = p.values
        discretise!(p_eqn, prev, runtime)
        apply_boundary_conditions!(p_eqn, p.BCs)
        implicit_relaxation!(p_eqn.equation, prev, solvers.U.relax) # Relax for diagonal dominance
        setReference!(p_eqn.equation, pref, 1)
        update_preconditioner!(p_eqn.preconditioner)
        run!(p_eqn, solvers.p)
        clamp!(p.values, 1000, 1000000)
        explicit_relaxation!(p, prev, solvers.p.relax)
        residual!(R_p, p_eqn.equation, p, iteration)

        grad!(∇p, pf, p, p.BCs) 

        # Correct velocity and mass flux
        correct_velocity!(U, Hv, ∇p, rD)
        # interpolate!(Uf, U)
        # correct_boundaries!(Uf, U, U.BCs)

        @. rho.values = Psi.values * p.values
        @. rhof.values = Psif.values * pf.values

        @. pdash.values = p.values - prev
        interpolate!(pfdash, pdash)
        correct_face_interpolation!(pfdash, pdash, Uf) 
        pgrad = face_normal_gradient(pdash, pfdash)
        @. pgrad.values *= rDf.values * rhof.values
        @. mdotf.values += (mdotf.values/rhof.values)*(Psif.values*pfdash.values)
        @. mdotf.values -= pgrad.values

        @. energy_source.values = U.x.values*∇p.result.x.values + U.y.values*∇p.result.y.values + U.z.values*∇p.result.z.values

        # Set up and solve energy equation
        @. prev = energy.values
        discretise!(energy_eqn, prev, runtime)
        apply_boundary_conditions!(energy_eqn, energy.BCs)
        implicit_relaxation!(energy_eqn.equation, prev, solvers.energy.relax)
        update_preconditioner!(energy_eqn.preconditioner)
        run!(energy_eqn, solvers.energy)
        residual!(R_e, energy_eqn.equation, energy, iteration)

        # @. energy.values = prev
        clamp!(energy.values, 100*1005, 1000*1005)

        # γ = solvers.energy.relax
        γ = 0.2
        @. Psi.values = (1-γ)*Psi.values + γ*Cp/(R*energy.values)
        interpolate!(energyf, energy)
        correct_face_interpolation!(energyf, energy, Uf) 
        correct_boundaries!(energyf, energy, energy.BCs)
        clamp!(energyf.values, 100*1005, 1000*1005)
        @. Psif.values = (1-γ)*Psif.values + γ*Cp/(R*energyf.values)

        println("Min h: ", minimum(energy.values))
        println("Min hf: ", minimum(energyf.values))

           
        if isturbulent(model)
            grad!(gradU, Uf, U, U.BCs)
            turbulence!(turbulence, model, S, S2, prev) 
            # update_nueff!(muff, nu, turbulence)
            update_nueff!(mueff, nu, rhof, turbulence)
        end
        
        # Update stuff
        update_nueff!(mueff, nu, rhof, turbulence)
        @. mueff.values = 0.001
        @. keff_by_cp.values = mueff.values/Pr
   
        convergence = 1e-12

        if (R_uz[iteration] == one(TF) &&
            R_ux[iteration] <= convergence && 
            R_uy[iteration] <= convergence || 
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
                (:energy, R_e[iteration]),
                ]
            )

        if iteration%write_interval + signbit(write_interval) == 0
            model2vtk(model, @sprintf "iteration_%.6d" iteration)
            write_vtk((@sprintf "rho_%.6d" iteration), mesh, ("rho", rho))
            write_vtk((@sprintf "gradDivU_%.6d" iteration), mesh, ("gradDivU", gradDivU.result))
        end

    end # end for loop
    return R_ux, R_uy, R_uz, R_p, R_e
end

function rhie_chow_correction!(mdotf::FaceScalarField, rhorDf::FaceScalarField, p::ScalarField, ∇p)
    mesh = p.mesh
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
        phi1 = p[cID1]
        phi2 = p[cID2]
        gradphif_x = 0.5*(∇p.result.x.values[cID1] + ∇p.result.x.values[cID2])
        gradphif_y = 0.5*(∇p.result.y.values[cID1] + ∇p.result.y.values[cID2])
        gradphif_z = 0.5*(∇p.result.z.values[cID1] + ∇p.result.z.values[cID2])

        face_grad1 = area*(phi2 - phi1)/delta
        face_grad2 = (normal[1]*gradphif_x + normal[2]*gradphif_y + normal[3]*gradphif_z) * area

        mdotf.values[fID] -= rhorDf.values[fID]*(face_grad1 - face_grad2)
    end
    # Now deal with boundary faces
    for fID ∈ 1:nbfaces
        face = faces[fID]
        (; area, normal, ownerCells, delta) = face 
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        cell1 = cells[cID1]
        cell2 = cells[cID2]
        phi1 = p[cID1]
        phi2 = p[cID2]
        gradphif_x = 0.5*(∇p.result.x.values[cID1] + ∇p.result.x.values[cID2])
        gradphif_y = 0.5*(∇p.result.y.values[cID1] + ∇p.result.y.values[cID2])
        gradphif_z = 0.5*(∇p.result.z.values[cID1] + ∇p.result.z.values[cID2])

        face_grad1 = area*(phi2 - phi1)/delta
        face_grad2 = (normal[1]*gradphif_x + normal[2]*gradphif_y + normal[3]*gradphif_z) * area

        mdotf.values[fID] -= (face_grad1 - face_grad2)
    end
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
        mugradUTx[fID] = mueff[fID] * (normal[1]*gradUxxf + normal[2]*gradUxyf + normal[3]*gradUxzf) * area
        mugradUTy[fID] = mueff[fID] * (normal[1]*gradUyxf + normal[2]*gradUyyf + normal[3]*gradUyzf) * area
        mugradUTz[fID] = mueff[fID] * (normal[1]*gradUzxf + normal[2]*gradUzyf + normal[3]*gradUzzf) * area
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
        sngrad.values[fID] = face_grad
    end
    sngrad
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

function correct_face_interpolation!(phif::FaceVectorField, phi, Uf)
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
            phif.x.values[fID] = phi1[1]
            phif.y.values[fID] = phi1[2]
            phif.z.values[fID] = phi1[3]
        else
            phif.x.values[fID] = phi2[1]
            phif.y.values[fID] = phi2[2]
            phif.z.values[fID] = phi2[3]
        end
    end
end

function wallBC!(ux_eqn, uy_eqn, U, mesh, nueff)
    (; boundaries, boundary_cellsID, faces, cells) = mesh
    for bci ∈ 1:length(U.x.BCs)
        if U.x.BCs[bci] isa Wall{}
            (; IDs_range) = boundaries[U.x.BCs[bci].ID]
            
            @inbounds for i ∈ eachindex(IDs_range)
                faceID = IDs_range[i]
                cellID = boundary_cellsID[faceID]
                face = faces[faceID]
                cell = cells[cellID]
                (; area, normal, delta) = face 

                Uc = U.x.values[cellID]
                Vc = U.y.values[cellID]
                nueff_face = nueff.values[faceID]

                ux_eqn.equation.A[cellID, cellID] += nueff_face*area*(1)/delta
                uy_eqn.equation.A[cellID, cellID] += nueff_face*area*(1)/delta

                ux_eqn.equation.b[cellID] += nueff_face*area*((0)*(1-normal[1]*normal[1]) + (Vc-0)*(normal[2]*normal[1]))/delta - Uc*nueff_face*area*(normal[1]*normal[1])/delta
                uy_eqn.equation.b[cellID] += nueff_face*area*((Uc-0)*(normal[1]*normal[2]) + (0)*(1-normal[2]*normal[2]))/delta - Vc*nueff_face*area*(normal[2]*normal[2])/delta
            end
        end
        if U.x.BCs[bci] isa Symmetry{}
            (; IDs_range) = boundaries[U.x.BCs[bci].ID]
            
            @inbounds for i ∈ eachindex(IDs_range)
                faceID = IDs_range[i]
                cellID = boundary_cellsID[faceID]
                face = faces[faceID]
                cell = cells[cellID]
                (; area, normal, delta) = face 

                Uc = U.x.values[cellID]
                Vc = U.y.values[cellID]
                nueff_face = nueff.values[faceID]

                # ux_eqn.equation.A[cellID, cellID] += nueff_face*area*(1)/delta
                # uy_eqn.equation.A[cellID, cellID] += nueff_face*area*(1)/delta

                ux_eqn.equation.b[cellID] -= 2.0*nueff_face*area*(Uc*normal[1] + Vc*normal[2])*normal[1]/delta
                uy_eqn.equation.b[cellID] -= 2.0*nueff_face*area*(Uc*normal[1] + Vc*normal[2])*normal[2]/delta
            end
        end
    end
    return ux_eqn, uy_eqn
end