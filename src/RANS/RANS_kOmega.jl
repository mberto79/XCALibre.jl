export kOmega
export initialise_RANS
export turbulence!

struct kOmegaCoefficients{T}
    β⁺::T
    α1::T
    β1::T
    σk::T
    σω::T
end

get_coeffs(FloatType) = begin
    kOmegaCoefficients{FloatType}(
        0.09,
        0.52, #5/9,
        0.072, #3/40,
        0.5,
        0.5
    )
end


struct kOmega{MK,MW,P1,P2,FK,FW,FN,C}
    k_model::MK
    ω_model::MW
    PK::P1
    PW::P2
    kf::FK
    ωf::FW
    νtf::FN
    coeffs::C
end

function initialise_RANS(k, ω, mdotf, eqn)
    mesh = mdotf.mesh

    kf = FaceScalarField(mesh)
    ωf = FaceScalarField(mesh)
    νtf = FaceScalarField(mesh)
    
    nueffk = FaceScalarField(mesh)
    nueffω = FaceScalarField(mesh)
    Dkf = ScalarField(mesh)
    Dωf = ScalarField(mesh)
    Pk = ScalarField(mesh)
    Pω = ScalarField(mesh)
    
    k_model = eqn → (
            Divergence{Linear}(mdotf, k) 
            - Laplacian{Linear}(nueffk, k) 
            + Si(Dkf,k) # Dkf = β⁺*ω
            ==
            Source(Pk)
        )
    
    ω_model = eqn → (
        Divergence{Linear}(mdotf, ω) 
        - Laplacian{Linear}(nueffω, ω) 
        + Si(Dωf,ω)  # Dωf = β1*ω
        ==
        Source(Pω)
    )

    # PK = set_preconditioner(DILU(), k_eqn, k_model, k.BCs)
    # PW = set_preconditioner(DILU(), ω_eqn, ω_model, ω.BCs)
    PK = set_preconditioner(ILU0(), k_model, k.BCs)
    PW = set_preconditioner(ILU0(), ω_model, ω.BCs)


    float_type = eltype(mesh.nodes[1].coords)
    coeffs = get_coeffs(float_type)

    kOmega_model = kOmega(
        k_model,
        ω_model,
        PK,
        PW,
        kf,
        ωf,
        νtf,
        coeffs
    )

    return kOmega_model

end

function turbulence!(
    kOmega::M, νt, nu, S, S2, solver, setup, prev,relax!) where M

    (;k_model,ω_model,PK,PW,kf,ωf,νtf,coeffs) = kOmega

    k = get_phi(k_model)
    ω = get_phi(ω_model)

    nueffk = get_flux(k_model, 2)
    Dkf = get_flux(k_model, 3)
    Pk = get_source(k_model, 1)

    nueffω = get_flux(ω_model, 2)
    Dωf = get_flux(ω_model, 3)
    Pω = get_source(ω_model, 1)

    # double_inner_product!(Pk, S, S.gradU)
    # cells = k.mesh.cells
    # for i ∈ eachindex(Pk)
    #     # Pk[i] = 2.0*Pk[i]*cells[i].volume
    #     Pk[i] = 2.0*Pk[i]*cells[i].volume
    # end

    magnitude2!(S2, S, scale_factor=2.0) # multiplied by 2 (def of Sij)
    # cells = k.mesh.cells
    # for i ∈ eachindex(Pk)
    #     Pk[i] = S2[i]*cells[i].volume
    # end

    @. Pω.values = coeffs.α1*Pk.values
    @. Dωf.values = coeffs.β1*ω.values

    interpolate!(kf, k)
    correct_boundaries!(kf, k, k.BCs)
    interpolate!(ωf, ω)
    correct_boundaries!(ωf, ω, ω.BCs)
    diffusion_flux!(nueffω, nu, kf, ωf, coeffs.σω)

    # Solve ω equation

    discretise!(ω_model)
    # ω_model.equation.b .+= Pω
    apply_boundary_conditions!(ω_model, ω.BCs)
    prev .= ω.values
    relax!(ω_model.equation, prev, setup.relax)
    constrain_equation!(ω_model.equation, ω, ω.BCs) # Only if using wall function?
    update_preconditioner!(PW)
    run!(ω_model, setup, opP=PW.P, solver=solver)
   
    constrain_boundary!(ω, ω.BCs)
    interpolate!(ωf, ω)
    correct_boundaries!(ωf, ω, ω.BCs)
    bound!(ω, ωf, eps())

    # update k fluxes

    @. Pk.values = νt.values*Pk.values
    @. Dkf.values = coeffs.β⁺*ω.values

    interpolate!(kf, k)
    correct_boundaries!(kf, k, k.BCs)
    interpolate!(ωf, ω)
    correct_boundaries!(ωf, ω, ω.BCs)
    diffusion_flux!(nueffk, nu, kf, ωf, coeffs.σk)

    # Solve k equation

    discretise!(k_model)
    # k_model.equation.b .+= Pk
    apply_boundary_conditions!(k_model, k.BCs)
    prev .= k.values
    relax!(k_model.equation, prev, setup.relax)
    update_preconditioner!(PK)
    run!(k_model, setup, opP=PK.P, solver=solver)
    interpolate!(kf, k)
    correct_boundaries!(kf, k, k.BCs)
    bound!(k, kf, eps())

    update_eddy_viscosity!(νt, k, ω)
    interpolate!(νtf, νt)
    correct_boundaries!(νtf, νt, νt.BCs)
end

update_eddy_viscosity!(νt::F, k, ω) where F<:AbstractScalarField = begin
    for i ∈ eachindex(νt)
        νt[i] = k[i]/ω[i]
    end
end

diffusion_flux!(nueff, nu, kf::F, ωf, σ) where F<:FaceScalarField = begin
    @inbounds for i ∈ eachindex(nueff)
        nueff[i] = nu[i] + σ*kf[i]/ωf[i]
    end
end

destruction_flux!(Dxf::F, coeff, ω) where F<:ScalarField = begin
    for i ∈ eachindex(Dxf.values)
        Dxf[i] = coeff*ω[i]
    end
end

@generated constrain_equation!(eqn, field, fieldBCs) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: OmegaWallFunction # || BC <: Dirichlet
            call = quote
                constraint!(eqn, field, fieldBCs[$i])
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

constraint!(eqn, field, BC) = begin
    ID = BC.ID
    # cmu = BC.value.cmu
    # κ = BC.value.κ
    # k = BC.value.k
    mesh = field.mesh
    (; faces, cells, boundaries) = mesh
    (; A, b) = eqn
    boundary = boundaries[ID]
    (; cellsID, facesID) = boundary
    # for fi ∈ eachindex(facesID)
    #     fID = facesID[fi]
    #     cID 
    for i ∈ eachindex(cellsID)
    # for i ∈ 1:length(cellsID)-1 
        cID = cellsID[i]
        # nID = cellsID[i+1]
        fID = facesID[i]
        # nfID = facesID[i+1]
        face = faces[fID]
        cell = cells[cID]
        y = face.delta
        ωc = 6*1e-3/(0.075*y^2)
        b[cID] = A[cID,cID]*ωc
        # b[cID] += A[cID,cID]*ωc
        # A[cID,cID] += A[cID,cID]
        # for nID ∈ cell.neighbours
        #     for boundaryCell ∈ cellsID
        #         if nID == boundaryCell
        #             # b[nID] += A[cID,nID]*ωc
        #             # b[cID] += A[nID,cID]*ωc
        #         end
        #     end
        # end
        # b[nID] -= A[cID, nID]*ωc
    end
end

@generated constrain_boundary!(field, fieldBCs) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: OmegaWallFunction  #|| BC <: Dirichlet
            call = quote
                set_cell_value!(field, fieldBCs[$i])
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

set_cell_value!(field, BC) = begin
    ID = BC.ID
    # cmu = BC.value.cmu
    # κ = BC.value.κ
    # k = BC.value.k
    mesh = field.mesh
    (; faces, cells, boundaries) = mesh
    boundary = boundaries[ID]
    (; cellsID, facesID) = boundary
    for i ∈ eachindex(cellsID)
        cID = cellsID[i]
        fID = facesID[i]
        y = faces[fID].delta
        ωc = 6*1e-3/(0.075*y^2)
        field.values[cID] = ωc
    end
end

bound!(field, fieldf, bound) = begin
    mesh = field.mesh
    (; cells, faces) = mesh
    for i ∈ eachindex(field)
        sum_flux = 0.0
        sum_area = 0
        average = 0.0
        
        # Cell based average
        cellsID = cells[i].neighbours
        for cID ∈ cellsID
            sum_flux += max(field[cID], eps()) # bounded sum?
            sum_area += 1
        end
        average = sum_flux/sum_area

        field[i] = max(
            max(
                field[i],
                average*signbit(field[i])
            ),
            bound
        )
    end
end