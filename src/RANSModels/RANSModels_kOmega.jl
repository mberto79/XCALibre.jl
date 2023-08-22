export KOmega
export initialise_RANS
export turbulence!

struct KOmega <: AbstractTurbulenceModel end

# Constructor 

RANS{KOmega}(; mesh, viscosity) = begin
    U = VectorField(mesh); F1 = typeof(U)
    p = ScalarField(mesh); F2 = typeof(p)
    V = typeof(viscosity)
    k = ScalarField(mesh); omega = ScalarField(mesh); nut = ScalarField(mesh)
    turb = (k=k , omega=omega, nut=nut); T = typeof(turb)
    flag = false; F = typeof(flag)
    D = typeof(mesh)
    RANS{KOmega,F1,F2,V,T,F,D}(
        KOmega(), U, p, viscosity, turb, flag, mesh
    )
end

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

struct KOmegaModel{MK,MW,FK,FW,FN,C,S}
    k_eqn::MK
    ω_eqn::MW
    kf::FK
    ωf::FW
    νtf::FN
    coeffs::C
    config::S
end

function initialise_RANS(mdotf, peqn, config, model)
    # unpack turbulent quantities and configuration
    turbulence = model.turbulence
    (; k, omega, nut) = turbulence
    (; solvers, schemes) = config
    mesh = mdotf.mesh
    eqn = peqn.equation

    kf = FaceScalarField(mesh)
    ωf = FaceScalarField(mesh)
    νtf = FaceScalarField(mesh)
    
    nueffk = FaceScalarField(mesh)
    nueffω = FaceScalarField(mesh)
    Dkf = ScalarField(mesh)
    Dωf = ScalarField(mesh)
    Pk = ScalarField(mesh)
    Pω = ScalarField(mesh)
    
    k_eqn = (
            Divergence{schemes.k.divergence}(mdotf, k) 
            - Laplacian{schemes.k.laplacian}(nueffk, k) 
            + Si(Dkf,k) # Dkf = β⁺*omega
            ==
            Source(Pk)
        ) → eqn
    
    ω_eqn = (
        Divergence{schemes.omega.divergence}(mdotf, omega) 
        - Laplacian{schemes.omega.laplacian}(nueffω, omega) 
        + Si(Dωf,omega)  # Dωf = β1*omega
        ==
        Source(Pω)
    ) → eqn

    # k_eqn = ModelEquation(k_model, eqn, (), ())
    # ω_eqn = ModelEquation(ω_model, eqn, (), ())

    # Set up preconditioners

    @reset k_eqn.preconditioner = set_preconditioner(
                    solvers.k.preconditioner, k_eqn, k.BCs)

    @reset ω_eqn.preconditioner = set_preconditioner(
                    solvers.omega.preconditioner, ω_eqn, omega.BCs)
    
    # preallocating solvers

    @reset k_eqn.solver = solvers.k.solver(_A(k_eqn), _b(k_eqn))
    @reset ω_eqn.solver = solvers.omega.solver(_A(ω_eqn), _b(ω_eqn))

    float_type = _get_float(mesh)
    coeffs = get_coeffs(float_type)

    return KOmegaModel(
        k_eqn,
        ω_eqn,
        kf,
        ωf,
        νtf,
        coeffs,
        config
    )
end

function turbulence!( # Sort out dispatch when possible
    KOmega::KOmegaModel, model, S, S2, prev)

    nu = model.nu
    nut = model.turbulence.nut

    (;k_eqn, ω_eqn, kf, ωf, νtf, coeffs, config) = KOmega
    (; solvers) = config

    k = get_phi(k_eqn)
    omega = get_phi(ω_eqn)

    nueffk = get_flux(k_eqn, 2)
    Dkf = get_flux(k_eqn, 3)
    Pk = get_source(k_eqn, 1)

    nueffω = get_flux(ω_eqn, 2)
    Dωf = get_flux(ω_eqn, 3)
    Pω = get_source(ω_eqn, 1)

    # double_inner_product!(Pk, S, S.gradU)
    # cells = k.mesh.cells
    # for i ∈ eachindex(Pk)
    #     # Pk[i] = 2.0*Pk[i]*cells[i].volume
    #     Pk[i] = 2.0*Pk[i]*cells[i].volume
    # end

    magnitude2!(Pk, S, scale_factor=2.0) # multiplied by 2 (def of Sij)
    # cells = k.mesh.cells
    # for i ∈ eachindex(Pk)
    #     # Pk[i] = S2[i]*cells[i].volume
    #     Pk[i] = S2[i]
    # end

    @. Pω.values = coeffs.α1*Pk.values
    @. Dωf.values = coeffs.β1*omega.values

    interpolate!(kf, k)
    correct_boundaries!(kf, k, k.BCs)
    interpolate!(ωf, omega)
    correct_boundaries!(ωf, omega, omega.BCs)
    diffusion_flux!(nueffω, nu, kf, ωf, coeffs.σω)

    # Solve omega equation

    discretise!(ω_eqn)
    apply_boundary_conditions!(ω_eqn, omega.BCs)
    prev .= omega.values
    implicit_relaxation!(ω_eqn.equation, prev, solvers.omega.relax)
    constrain_equation!(ω_eqn.equation, omega, omega.BCs) # Only if using wall function?
    update_preconditioner!(ω_eqn.preconditioner)
    run!(ω_eqn, solvers.omega)
   
    constrain_boundary!(omega, omega.BCs)
    interpolate!(ωf, omega)
    correct_boundaries!(ωf, omega, omega.BCs)
    bound!(omega, ωf, eps())

    # update k fluxes

    @. Pk.values = nut.values*Pk.values
    @. Dkf.values = coeffs.β⁺*omega.values

    interpolate!(kf, k)
    correct_boundaries!(kf, k, k.BCs)
    interpolate!(ωf, omega)
    correct_boundaries!(ωf, omega, omega.BCs)
    diffusion_flux!(nueffk, nu, kf, ωf, coeffs.σk)

    # Solve k equation

    discretise!(k_eqn)
    apply_boundary_conditions!(k_eqn, k.BCs)
    prev .= k.values
    implicit_relaxation!(k_eqn.equation, prev, solvers.k.relax)
    update_preconditioner!(k_eqn.preconditioner)
    run!(k_eqn, solvers.k)
    interpolate!(kf, k)
    correct_boundaries!(kf, k, k.BCs)
    bound!(k, kf, eps())

    update_eddy_viscosity!(nut, k, omega)
    interpolate!(νtf, nut)
    correct_boundaries!(νtf, nut, nut.BCs)
end

update_eddy_viscosity!(nut::F, k, omega) where F<:AbstractScalarField = begin
    for i ∈ eachindex(nut)
        nut[i] = k[i]/omega[i]
    end
end

diffusion_flux!(nueff, nu, kf::F, ωf, σ) where F<:FaceScalarField = begin
    @inbounds for i ∈ eachindex(nueff)
        nueff[i] = nu[i] + σ*kf[i]/ωf[i]
    end
end

destruction_flux!(Dxf::F, coeff, omega) where F<:ScalarField = begin
    for i ∈ eachindex(Dxf.values)
        Dxf[i] = coeff*omega[i]
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