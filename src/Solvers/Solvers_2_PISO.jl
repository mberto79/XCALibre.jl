export piso!

piso!(model_in, config; resume=true, pref=nothing) = begin

    R_ux, R_uy, R_uz, R_p, model = setup_incompressible_solvers(
        PISO, model_in, config;
        resume=true, pref=pref
        )
        
    return R_ux, R_uy, R_uz, R_p, model
end

function PISO(
    model, turbulenceModel, ∇p, U_eqn, p_eqn, config; resume=resume, pref=pref)
    
    # Extract model variables and configuration
    (; U, p) = model.momentum
    (; nu) = model.fluid
    mesh = model.domain
    p_model = p_eqn.model
    (; solvers, schemes, runtime, hardware) = config
    (; iterations, write_interval, dt) = runtime
    (; backend) = hardware
    
    mdotf = get_flux(U_eqn, 2)
    nueff = get_flux(U_eqn, 3)
    rDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)

    @info "Initialise VTKWriter (Store mesh in host memory)"

    VTKMeshData = initialise_writer(model.domain)
    
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
    TI = _get_int(mesh)
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

    # grad limiter test
    # limit_gradient!(∇p, p, config)

    update_nueff!(nueff, nu, model.turbulence, config)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    @info "Staring PISO loops..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations
        time = TI(iteration - 1)*dt

        solve_equation!(U_eqn, U, solvers.U, xdir, ydir, zdir, config; time=time)
          
        # Pressure correction
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        remove_pressure_source!(U_eqn, ∇p, config)
        
        ncorrectors = 3
        for i ∈ 1:ncorrectors
            H!(Hv, U, U_eqn, config)
            
            # Interpolate faces
            interpolate!(Uf, Hv, config) # Careful: reusing Uf for interpolation
            correct_boundaries!(Uf, Hv, U.BCs, time, config)
            # div!(divHv, Uf, config)

            # new approach
            flux!(mdotf, Uf, config)
            div!(divHv, mdotf, config)
            
            # Pressure calculations
            @. prev = p.values
            solve_equation!(p_eqn, p, solvers.p, config; ref=pref, time=time)
            if i == ncorrectors
                explicit_relaxation!(p, prev, 1.0, config)
            else
                explicit_relaxation!(p, prev, solvers.p.relax, config)
            end

            # Gradient
            grad!(∇p, pf, p, p.BCs, time, config) 

            # grad limiter test
            # limit_gradient!(∇p, p, config)

            correct = false
            if correct
                ncorrectors = 1
                for i ∈ 1:ncorrectors
                    discretise!(p_eqn, p, config)       
                    apply_boundary_conditions!(p_eqn, p.BCs, nothing, time, config)
                    setReference!(p_eqn, pref, 1, config)
                    # update_preconditioner!(p_eqn.preconditioner, p.mesh, config)
                    nonorthogonal_face_correction(p_eqn, ∇p, rDf, config)
                    # @. prev = p.values # this is unstable
                    # @. p.values = prev
                    solve_system!(p_eqn, solvers.p, p, nothing, config)
                    # explicit_relaxation!(p, prev, solvers.p.relax, config)
                    grad!(∇p, pf, p, p.BCs, time, config)
                    # limit_gradient!(∇p, p, config)
                end
            end

            # Velocity and boundaries correction
            # correct_velocity!(U, Hv, ∇p, rD, config)
            # interpolate!(Uf, U, config)
            # correct_boundaries!(Uf, U, U.BCs, time, config)
            # flux!(mdotf, Uf, config) # old approach

            # new approach
            interpolate!(Uf, U, config) # velocity from momentum equation
            correct_boundaries!(Uf, U, U.BCs, time, config)
            flux!(mdotf, Uf, config)
            correct_mass_flux(mdotf, p, pf, rDf, config)
            correct_velocity!(U, Hv, ∇p, rD, config)

        end # corrector loop end
        
        # correct_mass_flux(mdotf, p, pf, rDf, config) # new approach


    grad!(gradU, Uf, U, U.BCs, time, config)
    turbulence!(turbulenceModel, model, S, S2, prev, time, config) 
    update_nueff!(nueff, nu, model.turbulence, config)

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
            model2vtk(model, VTKMeshData, @sprintf "timestep_%.6d" iteration)
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