export isimple!

function isimple!(
    mesh, velocity, nu, ux, uy, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    setup, setup_p, iterations
    ; resume=true)
    # Pre-allocate fields

    U = VectorField(mesh)
    Uf = FaceVectorField(mesh)
    Hv = VectorField(mesh)
    divHv = Div(Hv)
    rD = ScalarField(mesh)
    rDf = FaceScalarField(mesh)
    pf = FaceScalarField(mesh)
    ∇p = Grad{Linear}(p)

    # Pre-allocated auxiliary variables
    ux0 = zeros(length(ux.values))
    ux0 .= velocity[1]
    uy0 = zeros(length(ux.values))
    uy0 .= velocity[2]
    p0 = zeros(length(p.values))

    # Define models and equations
    x_momentum_eqn = Equation(mesh)
    x_momentum_model = create_model(ConvectionDiffusion, Uf, nu, ux, ∇p.x)
    generate_boundary_conditions!(:ux_boundary_update!, mesh, x_momentum_model, uxBCs)

    y_momentum_eqn = Equation(mesh)
    y_momentum_model = create_model(ConvectionDiffusion, Uf, nu, uy, ∇p.y)
    generate_boundary_conditions!(:uy_boundary_update!, mesh, y_momentum_model, uyBCs)

    pressure_eqn = Equation(mesh)
    pressure_correction = create_model(Diffusion, rDf, p, divHv.values) #.*D)
    generate_boundary_conditions!(:p_boundary_update!, mesh, pressure_correction, pBCs)

    # Preallocate preconditioners
    Fm = ilu0(x_momentum_eqn.A)
    Fp = ilu0(pressure_eqn.A)

    # Update initial (guessed) fields
    if resume
        @. U.x = ux.values
        @. U.y = uy.values
        @. p0 = p.values
    else
        U.x .= velocity[1]; U.y .= velocity[2]
        ux.values .= velocity[1]; uy.values .= velocity[2]
    end
    interpolate!(Uf, U, UBCs)

    # Perform SIMPLE loops 
    @time for iteration ∈ 1:iterations

        println("Iteration ", iteration)
        
        source!(∇p, pf, p, pBCs)
        negative_vector_source!(∇p)
        
        discretise!(x_momentum_eqn, x_momentum_model)
        Discretise.ux_boundary_update!(x_momentum_eqn, x_momentum_model, uxBCs)
        # println("Solving x-momentum")
        alpha_U = 0.9
        implicit_relaxation!(x_momentum_eqn, ux0, alpha_U)
        # Fm = ilu(x_momentum_eqn.A)
        ilu0!(Fm, x_momentum_eqn.A)
        run!(x_momentum_eqn, x_momentum_model, uxBCs, setup, F=Fm)
        
        discretise!(y_momentum_eqn, y_momentum_model)
        Discretise.uy_boundary_update!(y_momentum_eqn, y_momentum_model, uyBCs)
        # println("Solving y-momentum")
        implicit_relaxation!(y_momentum_eqn, uy0, alpha_U)
        run!(y_momentum_eqn, y_momentum_model, uyBCs, setup, F=Fm)
        
        # alpha = 0.9
        # @. U.x = alpha*ux.values + (1.0 - alpha)*U.x
        # @. U.y = alpha*uy.values + (1.0 - alpha)*U.y
        
        # alpha = 1.0
        # @. U.x = alpha*ux.values + (1.0 - alpha)*ux0
        # @. U.y = alpha*uy.values + (1.0 - alpha)*uy0
        
        @. ux0 = U.x
        @. uy0 = U.y
        
        @. U.x = ux.values 
        @. U.y = uy.values
        
        inverse_diagonal!(rD, x_momentum_eqn)
        interpolate!(rDf, rD)
        remove_pressure_source!(x_momentum_eqn, y_momentum_eqn, ∇p, rD)
        # H!(Hv, U, x_momentum_eqn, y_momentum_eqn, B, V, H)
        H_new!(Hv, U, x_momentum_eqn, y_momentum_eqn)
        div!(divHv, UBCs) 
        # divHv.values .*= vols
        
        discretise!(pressure_eqn, pressure_correction)
        Discretise.p_boundary_update!(pressure_eqn, pressure_correction, pBCs)
        # println("Solving pressure correction")
        # Fp = ilu(pressure_eqn.A, τ = 0.1)
        ilu0!(Fp, pressure_eqn.A)
        run!(pressure_eqn, pressure_correction, pBCs, setup_p, F=Fp)
        
        if iteration == 1
            @. p0 = p.values
        end
        
        source!(∇p, pf, p, pBCs) 
        @. U.x = ux0
        @. U.y = uy0
        correct_velocity!(U, Hv, ∇p, rD)
        interpolate!(Uf, U, UBCs)
        
        explicit_relaxation!(p, p0, 0.9)
        source!(∇p, pf, p, pBCs) 
        # grad!(∇p, pf, p, pBCs) 
        correct_velocity!(ux, uy, Hv, ∇p, rD)

    end
        
end

function create_model(::Type{ConvectionDiffusion}, U, J, phi, S)
    model = ConvectionDiffusion(
        Divergence{Linear}(U, phi),
        Laplacian{Linear}(J, phi),
        S
        )
    model.terms.term2.sign[1] = -1
    return model
end

function create_model(::Type{Diffusion}, J, phi, S)
    model = Diffusion(
        Laplacian{Linear}(J, phi),
        S
        )
    return model
end

function implicit_relaxation!(eqn::Equation{I,F}, field, alpha) where {I,F}
    (; A, b) = eqn
    for i ∈ eachindex(b)
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

function inverse_diagonal!(rD::ScalarField, eqn)
    D = @view eqn.A[diagind(eqn.A)]
    rD.values .= 1.0./D
    nothing
end

function explicit_relaxation!(phi, phi0, alpha)
    # @. phi.values = phi.values + alpha*(phi.values - phi0)
    @. phi.values = alpha*phi.values + (1.0 - alpha)*phi0
    @. phi0 = phi.values
    nothing
end

function correct_velocity!(U, Hv, ∇p, rD)
    @. U.x = Hv.x - ∇p.x*rD.values
    @. U.y = Hv.y - ∇p.y*rD.values
    nothing
end

function correct_velocity!(ux, uy, Hv, ∇p, rD)
    @. ux.values = Hv.x - ∇p.x*rD.values
    @. uy.values = Hv.y - ∇p.y*rD.values
    # @. ux.values = (Hv.x - ∇p.x)*rD.values
    # @. uy.values = (Hv.y - ∇p.y)*rD.values
    nothing
end

function negative_vector_source!(∇p)
    ∇p.x .*= -1.0
    ∇p.y .*= -1.0
    nothing
end

function remove_pressure_source!(x_momentum_eqn, y_momentum_eqn, ∇p, rD)
    @. x_momentum_eqn.b -= ∇p.x
    @. y_momentum_eqn.b -= ∇p.y
    # @. x_momentum_eqn.b -= ∇p.x/rD.values
    # @. y_momentum_eqn.b -= ∇p.y/rD.values
    nothing
end
