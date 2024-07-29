export piso_comp!

piso_comp!(model_in, config; resume=true, pref=nothing) = begin

    R_ux, R_uy, R_uz, R_p, R_e, model = setup_unsteady_compressible_solvers(
        CPISO, model_in, config;
        resume=true, pref=nothing
        )
        
    return R_ux, R_uy, R_uz, R_p, model
end

# Setup for all compressible algorithms
function setup_unsteady_compressible_solvers(
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
            Time{schemes.p.time}(psi, p)  # correction(fvm::ddt(p)) means d(p)/d(t) - d(pold)/d(t)
            - Laplacian{schemes.p.laplacian}(rhorDf, p)
            ==
            -Source(divHv)
            -Source(ddtrho) # Needs to capture the correction part of dPdT and the explicit drhodt
        ) → ScalarEquation(mesh)
    elseif typeof(model.fluid) <: Compressible
        pconv = FaceScalarField(mesh)
        p_eqn = (
            Time{schemes.p.time}(psi, p)
            - Laplacian{schemes.p.laplacian}(rhorDf, p) 
            + Divergence{schemes.p.divergence}(pconv, p)
            ==
            -Source(divHv)
            -Source(ddtrho) # Needs to capture the correction part of dPdT and the explicit drhodt
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

function CPISO(
    model, turbulenceModel, ∇p, U_eqn, p_eqn, config; resume=resume, pref=pref)
    
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

    # grad limiter test
    limit_gradient!(∇p, p, config)

    update_nueff!(nueff, nu, model.turbulence, config)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    @info "Staring PISO loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations

        solve_equation!(U_eqn, U, solvers.U, xdir, ydir, zdir, config)
          
        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rhorDf, rD, config)
        @. rhorDf.values *= rhof.values

        remove_pressure_source!(U_eqn, ∇p, config)
        
        for i ∈ 1:2
            H!(Hv, U, U_eqn, config)
            
            # Interpolate faces
            interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
            correct_boundaries!(Uf, Hv, U.BCs, config)
            # div!(divHv, Uf, config)

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
            
            # Pressure calculations
            @. prev = p.values
            solve_equation!(p_eqn, p, solvers.p, config; ref=nothing)

            # Gradient
            grad!(∇p, pf, p, p.BCs, config) 

            # grad limiter test
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

            # Velocity and boundaries correction
            correct_velocity!(U, Hv, ∇p, rD, config)
            interpolate!(Uf, U, config)
            correct_boundaries!(Uf, U, U.BCs, config)
            # flux!(mdotf, Uf, config) # old approach

            correct_mass_flux(mdotf, p, pf, rDf, config)

            grad!(gradU, Uf, U, U.BCs, config)
            turbulence!(turbulenceModel, model, S, S2, prev, config) 
            update_nueff!(nueff, nu, model.turbulence, config)
    end # corrector loop end

    residual!(R_ux, U_eqn, U.x, iteration, xdir, config)
    residual!(R_uy, U_eqn, U.y, iteration, ydir, config)
    if typeof(mesh) <: Mesh3
        residual!(R_uz, U_eqn, U.z, iteration, zdir, config)
    end
    residual!(R_p, p_eqn, p, iteration, nothing, config)
        
        # for i ∈ eachindex(divUTx)
        #     vol = mesh.cells[i].volume
        #     divUTx = -sqrt(2)*(nuf[i] + νt[i])*(gradUT[i][1,1]+ gradUT[i][1,2] + gradUT[i][1,3])*vol
        #     divUTy = -sqrt(2)*(nuf[i] + νt[i])*(gradUT[i][2,1]+ gradUT[i][2,2] + gradUT[i][2,3])*vol
        # end
        
        # convergence = 1e-7

        # if (R_ux[iteration] <= convergence && 
        #     R_uy[iteration] <= convergence && 
        #     R_p[iteration] <= convergence)

        #     print(
        #         """
        #         \n\n\n\n\n
        #         Simulation converged! $iteration iterations in
        #         """)
        #         if !signbit(write_interval)
        #             model2vtk(model, @sprintf "timestep_%.6d" iteration)
        #         end
        #     break
        # end

        # co = courant_number(U, mesh, runtime) # MUST IMPLEMENT!!!!!!

        ProgressMeter.next!(
            progress, showvalues = [
                (:time,iteration*runtime.dt),
                # (:Courant,co),
                (:Ux, R_ux[iteration]),
                (:Uy, R_uy[iteration]),
                (:Uz, R_uz[iteration]),
                (:p, R_p[iteration]),
                ]
            )

        if iteration%write_interval + signbit(write_interval) == 0
            model2vtk(model, @sprintf "timestep_%.6d" iteration)
        end

    end # end for loop
    model_out = adapt(CPU(), model)
    return R_ux, R_uy, R_uz, R_p, model_out
end

function limit_gradient!(∇F, F, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = F.mesh
    (; cells, cell_neighbours, cell_faces, cell_nsign, faces) = mesh

    minPhi0 = maximum(F.values) # use min value so all values compared are larger
    maxPhi0 = minimum(F.values)

    kernel! = _limit_gradient!(backend, workgroup)
    kernel!(∇F, F, cells, cell_neighbours, cell_faces, cell_nsign, faces, minPhi0, maxPhi0, ndrange=length(cells))
    KernelAbstractions.synchronize(backend)
end

@kernel function _limit_gradient!(∇F, F, cells, cell_neighbours, cell_faces, cell_nsign, faces, minPhi, maxPhi)
    cID = @index(Global)
    # mesh = F.mesh
    # (; cells, cell_neighbours, cell_faces, cell_nsign, faces) = mesh

    # minPhi0 = maximum(F.values) # use min value so all values compared are larger
    # maxPhi0 = minimum(F.values)

    # for (cID, cell) ∈ enumerate(cells)
        cell = cells[cID]
        # minPhi = minPhi0 # reset for next cell
        # maxPhi = maxPhi0

        # find min and max values around cell
        faces_range = cell.faces_range
        
        phiP = F[cID]
        # minPhi = phiP # reset for next cell
        # maxPhi = phiP
        for fi ∈ faces_range
            nID = cell_neighbours[fi]
            phiN = F[nID]
            maxPhi = max(phiN, maxPhi)
            minPhi = min(phiN, minPhi)
        end

        g0 = ∇F[cID]
        cc = cell.centre

        for fi ∈ faces_range 
            fID = cell_faces[fi]
            face = faces[fID]
            nID = face.ownerCells[2]
            # phiN = F[nID]
            normal = face.normal
            nsign = cell_nsign[fi]
            na = nsign*normal

            fc = face.centre 
            cc_fc = fc - cc
            n0 = cc_fc/norm(cc_fc)
            gn = g0⋅n0
            δϕ = g0⋅cc_fc
            gτ = g0 - gn*n0
            if (maxPhi > phiP) && (δϕ > maxPhi - phiP)
                g0 = gτ + na*(maxPhi - phiP)
            elseif (minPhi < phiP) && (δϕ < minPhi - phiP)
                g0 = gτ + na*(minPhi - phiP)
            end            
        end
        ∇F.result.x.values[cID] = g0[1]
        ∇F.result.y.values[cID] = g0[2]
        ∇F.result.z.values[cID] = g0[3]
    # end
end