export kOmega, kk
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
        5/9,
        3/40,
        0.5,
        0.5
    )
end

struct kOmega{EK,EW,MK,MW,P1,P2,FK,FW,FN,C}
    k_eqn::EK
    ω_eqn::EW
    k_model::MK
    ω_model::MW
    PK::P1
    PW::P2
    kf::FK
    ωf::FW
    νtf::FN
    coeffs::C
end


function initialise_RANS(k, ω, mdotf)
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
    
    k_model = (
            Divergence{Linear}(mdotf, k) 
            - Laplacian{Linear}(nueffk, k) 
            + Si(Dkf,k) # Dkf = β⁺*ω
            ==
            Source(Pk)
        )
    
    ω_model = (
        Divergence{Linear}(mdotf, ω) 
        - Laplacian{Linear}(nueffω, ω) 
        + Si(Dωf,ω)  # Dωf = β1*ω
        ==
        Source(Pω)
    )

    k_eqn    = Equation(mesh)
    ω_eqn    = Equation(mesh)

    PK = set_preconditioner(DILU(), k_eqn, k_model, k.BCs)
    PW = set_preconditioner(DILU(), ω_eqn, ω_model, ω.BCs)


    float_type = eltype(mesh.nodes[1].coords)
    coeffs = get_coeffs(float_type)

    kOmega_model = kOmega(
        k_eqn,
        ω_eqn,
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

function turbulence!(kOmega::M, νt, nu, S, S2, solver, setup, relax!) where M

    prev = zeros(eltype(kOmega.coeffs.α1), length(S2))
    
    

    (;k_eqn,ω_eqn,k_model,ω_model,PK,PW,kf,ωf,νtf,coeffs) = kOmega

    k = get_phi(k_model)
    ω = get_phi(ω_model)

    nueffk = get_flux(k_model, 2)
    Dkf = get_flux(k_model, 3)
    Pk = k_model.sources[1].field

    nueffω = get_flux(ω_model, 2)
    Dωf = get_flux(ω_model, 3)
    Pω = ω_model.sources[1].field

    

    # double_inner_product!(Pk, S, S.gradU)
    # cells = k.mesh.cells
    # for i ∈ eachindex(Pk)
    #     # Pk[i] = 2.0*Pk[i]*cells[i].volume
    #     Pk[i] = 2.0*Pk[i]*cells[i].volume
    # end

    magnitude2!(S2, S) # should be multiplied by 2 (def of Sij)
    cells = k.mesh.cells
    for i ∈ eachindex(Pk)
        Pk[i] = 2.0*S2[i]*cells[i].volume
    end

    
    # # # correct_production!(Pk, k, k.BCs) # based on choice of wall function
    # # production_ω!(Pω, Pk, k, ω, νt, coeffs.α1)
    Pω .= coeffs.α1*Pk
    
    # constrain_boundary!(ω, ω.BCs)
    @. Dωf.values = coeffs.β1*ω.values
    # destruction_flux!(Dωf, coeffs.β1, ω) 
    
    @. νt.values = k.values/ω.values
    interpolate!(νtf, νt)
    correct_boundaries!(νtf, νt, νt.BCs)
    diffusion_flux!(nueffω, nu,  νtf, coeffs.σω)

    

    # Solve ω equation
    prev .= ω.values
    discretise!(ω_eqn, ω_model)
    apply_boundary_conditions!(ω_eqn, ω_model, ω.BCs)
    ω_eqn.b .+= Pω
    relax!(ω_eqn, prev, setup.relax)
    # constrain_equation!(ω_eqn, ω, ω.BCs) # Only if using wall function?
    update_preconditioner!(PW)
    run!(ω_eqn, ω_model, setup, opP=PW.P, solver=solver)
   
    # constrain_boundary!(ω, ω.BCs)
    interpolate!(ωf, ω)
    correct_boundaries!(ωf, ω, ω.BCs)
    bound!(ω, ωf, eps())
    # constrain_boundary!(ω, ω.BCs)

    # @. νt.values = k.values/ω.values
    # interpolate!(νtf, νt)
    # correct_boundaries!(νtf, νt, νt.BCs)

    # update k fluxes
    
    Pk .= νt.values.*Pk # add eddy viscosity
    @. Dkf.values = coeffs.β⁺*ω.values
    diffusion_flux!(nueffk, nu,  νtf, coeffs.σk)
    # @. nueffk.values = nueffω.values

    # Solve k equation
    # constrain_boundary!(k, k.BCs)
    prev .= k.values
    discretise!(k_eqn, k_model)
    apply_boundary_conditions!(k_eqn, k_model, k.BCs)
    k_eqn.b .+= Pk
    relax!(k_eqn, prev, setup.relax)
    update_preconditioner!(PK)
    run!(k_eqn, k_model, setup, opP=PK.P, solver=solver)
    # constrain_boundary!(k, k.BCs)
    interpolate!(kf, k)
    correct_boundaries!(kf, k, k.BCs)
    bound!(k, kf, eps())
    # constrain_boundary!(k, k.BCs)


    update_eddy_viscosity!(νt, k, ω)
    # @. νt.values = max(νt.values, eps())
    interpolate!(νtf, νt)
    correct_boundaries!(νtf, νt, νt.BCs)
end

update_eddy_viscosity!(νt::F, k, ω) where F<:AbstractScalarField = begin
    for i ∈ eachindex(νt)
        νt[i] = k[i]/ω[i]
    end
end

# diffusion_flux!(nueff, nu, kf::F, ωf, σ) where F<:FaceScalarField = begin
#     for i ∈ eachindex(nueff)
#         nueff[i] = nu[i] + σ*kf[i]/ωf[i]
#     end
# end

diffusion_flux!(nueff, nu, νtf::F, σ) where F<:FaceScalarField = begin
    for i ∈ eachindex(nueff)
        nueff[i] = nu[i] + σ*νtf[i]
    end
end

destruction_flux!(Dxf::F, coeff, ω) where F<:ScalarField = begin
    for i ∈ eachindex(Dxf.values)
        Dxf[i] = coeff*ω[i]
        # Dxf[i] = max(coeff*ω[i], eps()^2)
    end
end

production_k!(Pk, k, ω, νt, S2) = begin
    mesh = k.mesh
    cells = mesh.cells
    boundaries = mesh.boundaries
    start_cell = boundaries[end].cellsID[end] + 1
    end_cell = length(cells)
    # for i ∈ start_cell:end_cell
    for i ∈ eachindex(Pk)
        # Pk[i] = 2.0*νt[i]*S2[i]*cells[i].volume
        # Pk[i] = 2.0*k[i]/ω[i]*S2[i]*cells[i].volume
        Pk[i] = 1.0*S2[i]*cells[i].volume
    end
end

production_ω!(Pω, Pk, k, ω, νt, α1) = begin
    mesh = ω.mesh
    boundaries = mesh.boundaries
    start_cell = boundaries[end].cellsID[end] + 1
    end_cell = length(mesh.cells)
    # for i ∈ start_cell:end_cell
    for i ∈ eachindex(Pω)
        # Pω[i] =  α1*Pk[i]/νt[i]
        # Pω[i] =  α1*Pk[i]*ω[i]/max(k[i], 1e-100)
        Pω[i] =  α1*Pk[i]
    end
end

@generated constrain_equation!(eqn, field, fieldBCs) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: OmegaWallFunction || BC <: Dirichlet
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
    for i ∈ eachindex(cellsID)
        cID = cellsID[i]
        fID = facesID[i]
        b[cID] += A[cID,cID]*BC.value
        A[cID,cID] += A[cID,cID]
        # b[cID] = A[cID,cID]*BC.value
        # A[cID,cID] = 2*A[cID,cID]
    end
end

@generated constrain_boundary!(field, fieldBCs) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: OmegaWallFunction || BC <: Dirichlet
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
        field.values[cID] = BC.value
    end
end

@generated correct_production!(Pk, k, kBCs) = begin
    BCs = kBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: KWallFunction
            call = quote
                apply_wall_function!(Pk, k, kBCs[$i])
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end            
end

apply_wall_function!(Pk, k, kBC) = begin
    ID = kBC.ID
    cmu = kBC.value.cmu
    κ = kBC.value.κ
    mesh = k.mesh
    (; faces, cells, boundaries) = mesh
    boundary = boundaries[ID]
    (; cellsID, facesID) = boundary
    for i ∈ eachindex(cellsID)
        cID = cellsID[i]
        fID = facesID[i]
        cell = cells[cID]
        face = faces[fID]
        Pk[cID] = k[cID]^1.5*cmu^0.75/(κ*face.delta)*cell.volume
    end
end

bound!(field, fieldf, bound) = begin
    mesh = field.mesh
    (; cells, faces) = mesh
    for i ∈ eachindex(field)
        sum_flux = 0.0
        sum_area = 0
        average = 0.0
        # Face based average 

        # facesID = cells[i].facesID
        # for fID ∈ facesID
        #     area = faces[fID].area
        #     # sum_flux += max(fieldf[fID]*area, eps()) # bounded sum?
        #     sum_flux += max(fieldf[fID], eps()) # bounded sum?
        #     # sum_flux += abs(fieldf[fID]*area)
        #     sum_area += 1
        # end

        # Cell based average
        cellsID = cells[i].neighbours
        for cID ∈ cellsID
            # area = faces[fID].area
            # sum_flux += max(fieldf[fID]*area, eps()) # bounded sum?
            sum_flux += max(field[cID], eps()) # bounded sum?
            # sum_flux += abs(fieldf[fID]*area)
            # sum_area += 1*max(sign(field[cID]), 0.0)
            sum_area += 1
        end

        # npatches = length(mesh.boundaries)
        # last_boundary_cell = mesh.boundaries[npatches].cellsID[end]
        # if i < last_boundary_cell
        #     area = faces[fID].area
        #     sum_flux += max(fieldf[fID]*area, eps()) # bounded sum?
        #     sum_area += area
        # end

        average = sum_flux/sum_area
        field[i] = max(
            max(
                field[i],
                average*signbit(field[i])
            ),
            bound
        )
    end

    # field.values .= max.(field.values, val)
end

function kk()
    nothing
end