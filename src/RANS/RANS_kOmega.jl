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



# Dk(β⁺, k, ω) = β⁺.*k.values.*ω.values
# Pk = 2.0.*νt.values.*S2.values

function initialise_RANS(k, ω, mdotf)
    mesh = mdotf.mesh
    # k = ScalarField(mesh)
    # ω = ScalarField(mesh)
    # νt= ScalarField(mesh)

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
    
    magnitude2!(S2, S)

    (;k_eqn,ω_eqn,k_model,ω_model,PK,PW,kf,ωf,νtf,coeffs) = kOmega

    k = get_phi(k_model)
    ω = get_phi(ω_model)

    nueffk = get_flux(k_model, 2)
    Dkf = get_flux(k_model, 3)
    Pk = k_model.sources[1].field

    nueffω = get_flux(ω_model, 2)
    Dωf = get_flux(ω_model, 3)
    Pω = ω_model.sources[1].field

    interpolate!(kf, k)
    correct_boundaries!(kf, k, k.BCs)

    interpolate!(ωf, ω)
    correct_boundaries!(ωf, ω, ω.BCs)

    update_eddy_viscosity!(νtf, kf, ωf)
    # diffusion_flux!(nueffk, nu, νtf, coeffs.σk)
    # diffusion_flux!(nueffω, nu, νtf, coeffs.σω)
    diffusion_flux!(nueffk, nu, kf, ωf, coeffs.σk)
    diffusion_flux!(nueffω, nu, kf, ωf, coeffs.σω)

    production_k!(Pk, νt, S2)
    production_ω!(Pω, Pk, k, ω, coeffs.α1)

    destruction_flux!(Dkf, coeffs.β⁺, ω) 
    destruction_flux!(Dωf, coeffs.β1, ω) 

    discretise!(ω_eqn, ω_model)
    apply_boundary_conditions!(ω_eqn, ω_model, ω.BCs)
    update_preconditioner!(PW)
    prev .= ω.values
    relax!(ω_eqn, prev, setup.relax)
    run!(ω_eqn, ω_model, setup, opP=PW.P, solver=solver)
    bound!(ω)
    # relax!(ω, prev, setup.relax)

    discretise!(k_eqn, k_model)
    apply_boundary_conditions!(k_eqn, k_model, k.BCs)
    update_preconditioner!(PK)
    prev .= k.values
    relax!(k_eqn, prev, setup.relax)
    run!(k_eqn, k_model, setup, opP=PK.P, solver=solver)
    bound!(k)
    # relax!(k, prev, setup.relax)
    
    update_eddy_viscosity!(νt, k, ω)

    interpolate!(kf, k)
    correct_boundaries!(kf, k, k.BCs)

    interpolate!(ωf, ω)
    correct_boundaries!(ωf, ω, ω.BCs)
    
    update_eddy_viscosity!(νtf, kf, ωf)
end

update_eddy_viscosity!(νt::F, k, ω) where F<:AbstractScalarField = begin
    for i ∈ eachindex(νt)
        νt[i] = k[i]/ω[i]
    end
end

# diffusion_flux!(nueff, nu, νt::F, σ) where F<:FaceScalarField = begin
#     for i ∈ eachindex(νt.values)
#         # nueff[i] = nu[i] + νt[i]*σ
#         nueff[i] = nu[i] + νt[i]*σ
#     end
# end

diffusion_flux!(nueff, nu, k::F, ωf, σ) where F<:FaceScalarField = begin
    for i ∈ eachindex(nueff)
        nueff[i] = nu[i] + σ*k[i]/ωf[i]
    end
end

destruction_flux!(Dxf::F, coeff, ω) where F<:ScalarField = begin
    for i ∈ eachindex(Dxf.values)
        Dxf[i] = coeff*ω[i]
    end
end

production_k!(Pk, νt, S2) = begin
    mesh = νt.mesh
    cells = mesh.cells
    for i ∈ eachindex(Pk)
        Pk[i] = 2.0*νt[i]*S2[i]*cells[i].volume
    end
end

production_ω!(Pω, Pk, k, ω, α1) = begin
    for i ∈ eachindex(Pk)
        Pω[i] =  α1*Pk[i]*ω[i]/max(k[i], 1e-20)
    end
end

bound!(field) = begin
    for i ∈ eachindex(field)
        field[i] = max(field[i], 0.0)
    end
end

function kk()
    nothing
end