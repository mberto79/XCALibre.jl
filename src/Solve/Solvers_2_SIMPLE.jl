export isimple!

function isimple!(
    mesh::Mesh2{TI,TF}, velocity, nu, ux, uy, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    setup_U, setup_p, iterations
    ; resume=true, pref=nothing) where {TI,TF}

    n_cells = m = n = length(mesh.cells)

    # Pre-allocate fields
    U = VectorField(mesh)
    Uf = FaceVectorField(mesh)
    mdot = ScalarField(mesh)
    mdotf = FaceScalarField(mesh)
    pf = FaceScalarField(mesh)
    ∇p = Grad{Linear}(p)
    
    Hv = VectorField(mesh)
    divHv = Div(Hv, FaceVectorField(mesh), zeros(TF, n_cells), mesh)
    rD = ScalarField(mesh)
    rDf = FaceScalarField(mesh)

    # Pre-allocated auxiliary variables
    ux0 = zeros(TF, n_cells)
    uy0 = zeros(TF, n_cells)
    p0 = zeros(TF, n_cells)

    ux0 .= velocity[1]
    uy0 .= velocity[2]
    p0 .= zero(TF)
    rDf.values .= 1.0

    # Define models and equations
    x_momentum_eqn = Equation(mesh)
    opAx = LinearOperator(x_momentum_eqn.A)
    Px = ilu0(x_momentum_eqn.A)
    opPUx = LinearOperator(Float64, m, n, false, false, (y, v) -> ldiv!(y, Px, v))
    x_momentum_model = create_model(ConvectionDiffusion, Uf, nu, ux, ∇p.x)
    
    y_momentum_eqn = Equation(mesh)
    opAy = LinearOperator(y_momentum_eqn.A)
    Py = ilu0(y_momentum_eqn.A)
    opPUy = LinearOperator(Float64, m, n, false, false, (y, v) -> ldiv!(y, Py, v))
    y_momentum_model = create_model(ConvectionDiffusion, Uf, nu, uy, ∇p.y)
    
    pressure_eqn = Equation(mesh)
    pressure_correction = create_model(Diffusion, rDf, p, divHv.values)
    discretise!(pressure_eqn, pressure_correction)
    apply_boundary_conditions!(pressure_eqn, pressure_correction, pBCs)
    opAp = LinearOperator(pressure_eqn.A)
    opPP = opLDL(pressure_eqn.A)

    solver_p = setup_p.solver(pressure_eqn.A, pressure_eqn.b)
    solver_U = setup_U.solver(x_momentum_eqn.A, x_momentum_eqn.b)

    #### NEED TO IMPLEMENT A SENSIBLE INITIALISATION TO INCLUDE WARM START!!!!
    # Update initial (guessed) fields

    @inbounds ux0 .= ux.values
    @inbounds uy0 .= uy.values 
    @inbounds p0 .= p.values
    @inbounds U.x .= ux.values #velocity[1]
    @inbounds U.y .= uy.values# velocity[2]
    # @inbounds ux.values .= velocity[1]
    # @inbounds uy.values .= velocity[2]
    # end
    volume  = volumes(mesh)
    
    # interpolate!(Uf, U, UBCs)   
    interpolate!(Uf, U)   
    correct_boundaries!(Uf, U, UBCs)
    source!(∇p, pf, p, pBCs)
    
    # Perform SIMPLE loops 
    R_ux = TF[]
    @time for iteration ∈ 1:iterations

        print("\nIteration ", iteration, "\n") # 91 allocations
        
        # source!(∇p, pf, p, pBCs)
        # grad!(∇p, pf, p, pBCs)
        neg!(∇p)
        
        print("Solving x-momentum. ")

        discretise!(x_momentum_eqn, x_momentum_model)
        @inbounds @. y_momentum_eqn.A.nzval = x_momentum_eqn.A.nzval
        apply_boundary_conditions!(x_momentum_eqn, x_momentum_model, uxBCs)
        implicit_relaxation!(x_momentum_eqn, ux0, setup_U.relax)
        ilu0!(Px, x_momentum_eqn.A)
        run!(
            x_momentum_eqn, x_momentum_model, uxBCs, 
            setup_U, opA=opAx, opP=opPUx, solver=solver_U
        )

        res = residual(x_momentum_eqn, ux, opAx, solver_U)
        push!(R_ux, res)
        if res <= 1e-6
            print("\nSimulation met convergence criterion. Stop!\n")
            break
        end

        print("Solving y-momentum. ")

        @inbounds @. y_momentum_eqn.b = 0.0
        apply_boundary_conditions!(y_momentum_eqn, y_momentum_model, uyBCs)
        implicit_relaxation!(y_momentum_eqn, uy0, setup_U.relax)
        ilu0!(Py, y_momentum_eqn.A)
        run!(
            y_momentum_eqn, y_momentum_model, uyBCs, 
            setup_U, opA=opAy, opP=opPUy, solver=solver_U
        )

        @inbounds for i ∈ eachindex(ux0)
            ux0[i] = U.x[i]
            uy0[i] = U.y[i]
            U.x[i] = ux.values[i]
            U.y[i] = uy.values[i]
        end
        
        inverse_diagonal!(rD, x_momentum_eqn)
        interpolate!(rDf, rD)
        remove_pressure_source!(x_momentum_eqn, y_momentum_eqn, ∇p, rD)
        H!(Hv, U, x_momentum_eqn, y_momentum_eqn)
        
        @inbounds for i ∈ eachindex(ux0)
            U.x[i] = ux0[i]
            U.y[i] = uy0[i]
        end
        div!(divHv, UBCs) # 7 allocations
        # @inbounds @. divHv.values *= 1.0./volume
        @inbounds @. rD.values *= volume#^2
        interpolate!(rDf, rD)
        @inbounds @. rD.values /= volume#^2

        print("Solving pressure correction. ")

        discretise!(pressure_eqn, pressure_correction)
        apply_boundary_conditions!(pressure_eqn, pressure_correction, pBCs)
        setReference!(pressure_eqn, pref)
        run!(
            pressure_eqn, pressure_correction, pBCs, 
            setup_p, opA=opAp, opP=opPP, solver=solver_p
        )
        
        source!(∇p, pf, p, pBCs)
        # grad!(∇p, pf, p, pBCs) 
        correct_velocity!(U, Hv, ∇p, rD)
        interpolate!(Uf, U)
        correct_boundaries!(Uf, U, UBCs)
        
        explicit_relaxation!(p, p0, setup_p.relax)
        source!(∇p, pf, p, pBCs)
        # grad!(∇p, pf, p, pBCs) 
        correct_velocity!(ux, uy, Hv, ∇p, rD)
    end # end for loop
    return R_ux, U        
end # end function


function residual(equation::Equation{TI,TF}, phi, opA, solver) where {TI,TF}
    (; A, b, R, Fx) = equation
    values = phi.values
    # Option 1
    # mul!(Fx, opA, values)
    # @inbounds @. R = abs(Fx - b)
    # res = norm(R)/mean(values)

    # Option 2
    # mul!(Fx, opA, values)
    # @inbounds @. R = b - Fx
    # sum = zero(TF)
    # @inbounds for i ∈ eachindex(R)
    #     sum += (R[i])^2 
    # end
    # res = sqrt(sum/length(R))

    # Option 3 (OpenFOAM definition)
    solMean = mean(values)
    term1 = abs.(opA*(values .- solMean))
    # term2 = abs.(b .- opA*solMean*values./values)
    term2 = abs.(b .- opA*solMean*values./values)
    N = sum(term1 + term2)
    res = (1/N)*sum(abs.(b - opA*values))

    print("Residual: ", res, " (", niterations(solver), " iterations)\n") 
    return res
end

function calculate_residuals(
    divU::ScalarField, UxEqn::Equation, UyEqn::Equation, Ux::ScalarField, Uy::ScalarField)
    
    continuityError = abs(mean(-divU.values))
    # UxResidual = norm(UxEqn.A*Ux.values - UxEqn.b)/Rx₀
    # UyResidual = norm(UyEqn.A*Uy.values - UyEqn.b)/Ry₀

    solMean = mean(Ux.values)
    N = sum(
        abs.(UxEqn.A*(Ux.values .- solMean)) + 
        abs.(UxEqn.b .- UxEqn.A*ones(length(UxEqn.b))*solMean)
        )
    UxResidual = (1/N)*sum(abs.(UxEqn.b - UxEqn.A*Ux.values))

    solMean = mean(Uy.values)
    N = sum(
        abs.(UyEqn.A*(Uy.values .- solMean)) + 
        abs.(UyEqn.b .- UyEqn.A*ones(length(UyEqn.b))*solMean)
        )
    UyResidual = (1/N)*sum(abs.(UyEqn.b - UyEqn.A*Uy.values))

    # UxResidual = sum(sqrt.((UxEqn.b - UxEqn.A*Ux.values).^2))/length(UxEqn.b)/Rx₀
    # UyResidual = sum(sqrt.((UyEqn.b - UyEqn.A*Uy.values).^2))/length(UyEqn.b)/Ry₀

    return continuityError, UxResidual, UyResidual
end


function mass_flux!(mdotf::FaceScalarField{I,F}, Uf::FaceVectorField{I,F}) where {I,F}
    (; mesh, values) = mdotf
    (; cells, faces) = mesh 
    @inbounds for fID ∈ eachindex(faces)
        (; area, normal) = faces[fID]
        values[fID] = Uf(fID)*area⋅normal
    end
end

function flux!()
    nothing
end

function implicit_relaxation!(eqn::Equation{I,F}, field, alpha) where {I,F}
    (; A, b) = eqn
    @inbounds @simd for i ∈ eachindex(b)
        A[i,i] /= alpha
        b[i] += (1.0 - alpha)*A[i,i]*field[i]
    end
end

function correct_face_velocity!(Uf, p, )
    mesh = Uf.mesh
    (; cells, faces) = mesh
    nbfaces = total_boundary_faces(mesh)
    for fID ∈ (nbfaces + 1):length(faces)
        face = faces[fID]
        gradp = 0.0
        Uf.x = nothing
        ################
        # CONTINUE 
        ################
    end
end

volumes(mesh) = [mesh.cells[i].volume for i ∈ eachindex(mesh.cells)]

function correct_boundary_Hvf!(Hvf, ux, uy, ∇pf, UBCs)
    mesh = ux.mesh
    for BC ∈ UBCs
        if typeof(BC) <: Neumann
            bi = boundary_index(mesh, BC.name)
            boundary = mesh.boundaries[bi]
            correct_flux_boundary!(BC, phif, phi, boundary, faces)
        end
    end
end

function correct_flux_boundary!(
    BC::Neumann, phif::FaceScalarField{I,F}, phi, boundary, faces) where {I,F}
    (; facesID, cellsID) = boundary
    for fID ∈ facesID
        phif.values[fID] = BC.value 
    end
end

function inverse_diagonal!(rD::ScalarField{I,F}, eqn) where {I,F}
    # D = @view eqn.A[diagind(eqn.A)]
    # @. rD.values = 1.0./D
    A = eqn.A
    rD_values = rD.values
    @inbounds for i ∈ eachindex(rD_values)
        # ap = @view A[i,i]
        # rD_values[i] = 1.0/ap
        rD_values[i] = 1.0/view(A, i, i)[1]
    end
end

function explicit_relaxation!(phi, phi0, alpha)
    # @. phi.values = alpha*phi.values + (1.0 - alpha)*phi0
    # @. phi0 = phi.values
    values = phi.values
    @inbounds for i ∈ eachindex(values)
        values[i] = alpha*values[i] + (1.0 - alpha)*phi0[i]
        phi0[i] = values[i]
    end
end

function correct_velocity!(U, Hv, ∇p, rD)
    # @. U.x = Hv.x - ∇p.x*rD.values
    # @. U.y = Hv.y - ∇p.y*rD.values
    Ux = U.x; Uy = U.y; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.x; dpdy = ∇p.y; rDvalues = rD.values
    @inbounds for i ∈ eachindex(Ux)
        Ux[i] = Hvx[i] - dpdx[i]*rDvalues[i]
        Uy[i] = Hvy[i] - dpdy[i]*rDvalues[i]
    end
end

function correct_velocity!(ux, uy, Hv, ∇p, rD)
    # @. ux.values = Hv.x - ∇p.x*rD.values
    # @. uy.values = Hv.y - ∇p.y*rD.values
    ux = ux.values; uy = uy.values; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.x; dpdy = ∇p.y; rDvalues = rD.values
    @inbounds for i ∈ eachindex(ux)
        ux[i] = Hvx[i] - dpdx[i]*rDvalues[i]
        uy[i] = Hvy[i] - dpdy[i]*rDvalues[i]
    end
end

function neg!(∇p)
    # ∇p.x .*= -1.0
    # ∇p.y .*= -1.0
    dpdx = ∇p.x; dpdy = ∇p.y
    @inbounds for i ∈ eachindex(dpdx)
        dpdx[i] *= -1.0
        dpdy[i] *= -1.0
    end
end

function remove_pressure_source!(x_momentum_eqn, y_momentum_eqn, ∇p, rD)
    # @. x_momentum_eqn.b -= ∇p.x/rD.values
    # @. y_momentum_eqn.b -= ∇p.y/rD.values
    dpdx, dpdy, rD = ∇p.x, ∇p.y, rD.values
    bx, by = x_momentum_eqn.b, y_momentum_eqn.b
    @inbounds for i ∈ eachindex(bx)
        bx[i] -= dpdx[i] #/rD[i]
        by[i] -= dpdy[i] #/rD[i]
    end
end

function setReference!(pEqn::Equation{TI,TF}, pRef) where {TI,TF}
    if pRef === nothing
        return nothing
    else
        pEqn.b[1] += pEqn.A[1,1]*pRef
        pEqn.A[1,1] += pEqn.A[1,1]
    end
end