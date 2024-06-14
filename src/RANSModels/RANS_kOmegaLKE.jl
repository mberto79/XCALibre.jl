export KOmegaLKE

# Model type definition (hold fields)
struct KOmegaLKE{S1,S2,S3,S4,F1,F2,F3,F4,C} <: AbstractTurbulenceModel 
    k::S1
    omega::S2
    kl::S3
    nut::S4
    kf::F1
    omegaf::F2
    klf::F3
    nutf::F4
    coeffs::C
end 
Adapt.@adapt_structure KOmegaLKE

# Model type definition (hold equation definitions and data)
struct KOmegaLKEModel <: AbstractTurbulenceModel # ADD PARAMETRIC TYPES
    k_eqn
    ω_eqn
    kl_eqn
    nueffkLS
    nueffkS
    nueffωS
    nuL
    nuts
    Ω
    γ
    fv
    ∇k
    ∇ω
    normU
    Reυ
end 
Adapt.@adapt_structure KOmegaLKEModel

# Model API constructor
RANS{KOmegaLKE}(mesh) = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    kl = ScalarField(mesh)
    nut = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    klf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = (
        C1 = 0.02974,
        C2 = 59.79,
        C3 = 1.191,
        C4 = 1.65*10^-13,
        Cμ = 0.09,
        Cω1 = 0.52,
        Cω2 = 0.0708,
        Ccrit = 76500,
        CSS = 1.45,
        Cv = 0.43,
        σk = 0.5,
        σd = 0.125,
        σkL = 0.0125,
        σω = 0.5
    )
    KOmegaLKE(k, omega, kl, nut, kf, omegaf, klf, nutf, coeffs)
end

# Model initialisation
function initialise(
    turbulence::KOmegaLKE, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}

    @info "Initialising k-ω LKE model..."
    # unpack turbulent quantities and configuration
    turbulence = model.turbulence
    (; kl, k, omega) = turbulence
    (; solvers, schemes, runtime) = config
    mesh = mdotf.mesh
    eqn = peqn.equation

    calc_wall_distance!(model, config)

    nueffkLS = ScalarField(mesh)
    nueffkS = ScalarField(mesh)
    nueffωS = ScalarField(mesh)
    nueffkL = FaceScalarField(mesh)
    nueffk = FaceScalarField(mesh)
    nueffω = FaceScalarField(mesh)
    DkLf = ScalarField(mesh)
    Dkf = ScalarField(mesh)
    Dωf = ScalarField(mesh)
    PkL = ScalarField(mesh)
    Pk = ScalarField(mesh)
    Pω = ScalarField(mesh)
    dkdomegadx = ScalarField(mesh)
    normU = ScalarField(mesh)
    Reυ = ScalarField(mesh)
    nuL = ScalarField(mesh)
    nuts = ScalarField(mesh)
    Ω = ScalarField(mesh)
    γ = ScalarField(mesh)
    fv = ScalarField(mesh)
    ∇k = Grad{schemes.k.gradient}(k)
    ∇ω = Grad{schemes.p.gradient}(omega)

    kl_eqn = (
            Time{schemes.kl.time}(kl)
            + Divergence{schemes.kl.divergence}(mdotf, kl) 
            - Laplacian{schemes.kl.laplacian}(nueffkL, kl) 
            + Si(DkLf,kl) # DkLf = 2*nu/y^2
            ==
            Source(PkL)
        ) → eqn

    k_eqn = (
            Time{schemes.k.time}(k)
            + Divergence{schemes.k.divergence}(mdotf, k) 
            - Laplacian{schemes.k.laplacian}(nueffk, k) 
            + Si(Dkf,k) # Dkf = β⁺*omega
            ==
            Source(Pk)
        ) → eqn
    
    ω_eqn = (
            Time{schemes.omega.time}(omega)
            + Divergence{schemes.omega.divergence}(mdotf, omega) 
            - Laplacian{schemes.omega.laplacian}(nueffω, omega)
            + Si(Dωf,omega)  # Dωf = β1*omega
            ==
            Source(Pω)
            + Source(dkdomegadx)
    ) → eqn

    
    # Set up preconditioners

    @reset kl_eqn.preconditioner = set_preconditioner(
                solvers.kl.preconditioner, kl_eqn, kl.BCs, runtime)

    @reset k_eqn.preconditioner = set_preconditioner(
                solvers.k.preconditioner, k_eqn, k.BCs, runtime)

    @reset ω_eqn.preconditioner = set_preconditioner(
                solvers.omega.preconditioner, ω_eqn, omega.BCs, runtime)
    
    # preallocating solvers

    @reset kl_eqn.solver = solvers.kl.solver(_A(kl_eqn), _b(kl_eqn))
    @reset k_eqn.solver = solvers.k.solver(_A(k_eqn), _b(k_eqn))
    @reset ω_eqn.solver = solvers.omega.solver(_A(ω_eqn), _b(ω_eqn))

    grad!(∇ω,ωf,omega,omega.BCs)
    grad!(∇k,kf,k,k.BCs)

    # float_type = _get_float(mesh)
    # coeffs = get_LKE_coeffs(float_type)

    # Wall distance calculation
    y = ScalarField(mesh) # dummy entry for now

    return KOmegaLKEModel(
        k_eqn,
        ω_eqn,
        kl_eqn,
        nueffkLS,
        nueffkS,
        nueffωS,
        nuL,
        nuts,
        Ω,
        γ,
        fv,
        ∇k,
        ∇ω,
        normU,
        Reυ,
        y
    )
end

# Model solver call (implementation)
function turbulence!(rans::KOmegaLKEModel, model::Physics{T,F,M,Turb,E,D,BI}, S, S2, prev, config
    ) where {T,F,M,Turb<:KOmegaLKE,E,D,BI}
    (; momentum, turbulence) = model
    (; nu, U) = momentum
    (; nut, Tu) = turbulence
    
    (; k_eqn, ω_eqn, kl_eqn, nueffkLS, nueffkS, nueffωS, nuL, nuts, Ω, γ, fv, ∇k, ∇ω, normU, Reυ) = rans
    (; solvers, runtime) = config

    kl = get_phi(kl_eqn)
    k = get_phi(k_eqn)
    omega = get_phi(ω_eqn)

    nueffkL = get_flux(kl_eqn, 3)
    DkLf = get_flux(kl_eqn, 4)
    PkL = get_source(kl_eqn, 1)

    nueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    Pk = get_source(k_eqn, 1)

    nueffω = get_flux(ω_eqn, 3)
    Dωf = get_flux(ω_eqn, 4)
    Pω = get_source(ω_eqn, 1)
    dkdomegadx = get_source(ω_eqn, 2) # cross diffusion term

    magnitude2!(Pk, S, scale_factor=2.0) # this is S^2

    # Update kl fluxes and terms

    for i ∈ eachindex(normU.values)
        normU.values[i] = norm(U[i])
    end

    ReLambda = @. normU.values*y.values/nu.values;
    @. Reυ.values = (2*nu.values^2*kl.values/(y.values^2))^0.25*y.values/nu.values;
    η = coeffs.C1*tanh(coeffs.C2*(Tu^coeffs.C3)+coeffs.C4)
    @. PkL.values = sqrt(Pk.values)*η*kl.values*Reυ.values^(-1.30)*ReLambda^(0.5)
    @. DkLf.values = (2*nu.values)/(y.values^2)
    @. nueffkLS.values = nu.values+(coeffs.σkL*sqrt(kl.values)*y.values)
    interpolate!(nueffkL,nueffkLS)
    correct_boundaries!(nueffkL, nueffkLS, nut.BCs)

    # Solve kl equation
    prev .= kl.values
    discretise!(kl_eqn, prev, runtime)
    apply_boundary_conditions!(kl_eqn, kl.BCs)
    implicit_relaxation!(kl_eqn.equation, prev, solvers.kl.relax)
    update_preconditioner!(kl_eqn.preconditioner)
    run!(kl_eqn, solvers.kl)
    bound!(kl, eps())

    #Damping and trigger
    @. fv.values = 1-exp(-sqrt(k.values/(nu.values*omega.values))/coeffs.Cv)
    @. γ.values = min((kl.values/(min(nu.values,nuL.values)*sqrt(S2.values)))^2,coeffs.Ccrit)/coeffs.Ccrit
    fSS = @. exp(-(coeffs.CSS*nu.values*sqrt(S2.values)/k.values)^2) # should be Ω but S works

    #Update ω fluxes
    # double_inner_product!(Pk, S, gradU) # multiplied by 2 (def of Sij) (Pk = S² at this point)
    interpolate!(kf, k)
    correct_boundaries!(νtf, k, k.BCs)
    interpolate!(ωf, omega)
    correct_boundaries!(νtf, omega, omega.BCs)
    grad!(∇ω,ωf,omega,omega.BCs)
    grad!(∇k,kf,k,k.BCs)
    inner_product!(dkdomegadx,∇k,∇ω)
    @. Pω.values = coeffs.Cω1*Pk.values*nut.values*(omega.values/k.values)
    @. dkdomegadx.values = max((coeffs.σd/omega.values)*dkdomegadx.values, 0.0)
    @. Dωf.values = coeffs.Cω2*omega.values
    @. nueffωS.values = nu.values+(coeffs.σω*nut_turb.values*γ.values)
    interpolate!(nueffω,nueffωS)
    correct_boundaries!(nueffω, nueffωS, nut.BCs)

    #Update k fluxes
    @. Dkf.values = coeffs.Cμ*omega.values*γ.values
    @. nueffkS.values = nu.values+(coeffs.σk*nut_turb.values*γ.values)
    interpolate!(nueffk,nueffkS)
    correct_boundaries!(nueffk, nueffkS, nut.BCs)
    @. Pk.values = nut.values*Pk.values*γ.values*fv.values
    correct_production!(Pk, k.BCs, model)

    # Solve omega equation
    prev .= omega.values
    discretise!(ω_eqn, prev, runtime)
    apply_boundary_conditions!(ω_eqn, omega.BCs)
    implicit_relaxation!(ω_eqn.equation, prev, solvers.omega.relax)
    constrain_equation!(ω_eqn, omega.BCs, model) # active with WFs only
    update_preconditioner!(ω_eqn.preconditioner)
    run!(ω_eqn, solvers.omega)
    constrain_boundary!(omega, omega.BCs, model) # active with WFs only
    bound!(omega, eps())

    # Solve k equation
    prev .= k.values
    discretise!(k_eqn, prev, runtime)
    apply_boundary_conditions!(k_eqn, k.BCs)
    implicit_relaxation!(k_eqn.equation, prev, solvers.k.relax)
    update_preconditioner!(k_eqn.preconditioner)
    run!(k_eqn, solvers.k)
    bound!(k, eps())

    grad!(∇ω,ωf,omega,omega.BCs)
    grad!(∇k,kf,k,k.BCs)

    #Eddy viscosity
    # magnitude2!(S2, S, scale_factor=2.0)
    # double_inner_product!(Ω,S,S, scale_factor=2.0)

    @. nut_turb.values = k.values/omega.values
    ReLambda = @. normU.values*y.values/nu.values
    @. Reυ.values = (2*nu.values^2*kl.values/(y.values^2))^0.25*y.values/nu.values;
    @. PkL.values = sqrt(Pk.values)*η*kl.values*Reυ.values^(-1.30)*ReLambda^(0.5) # update
    @. nuL.values = PkL.values/max(S2.values,(normU.values/y.values)^2)

    fSS = @. exp(-(coeffs.CSS*nu.values*sqrt(S2.values)/k.values)^2) # should be Ω but S works
    @. nuts.values = fSS*(k.values/omega.values)
    @. nut.values = nuts.values+nuL.values

    interpolate!(νtf, nut)
    correct_boundaries!(νtf, nut, nut.BCs)
    correct_eddy_viscosity!(νtf, nut.BCs, model)
end

# Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, name) where {T,F,M,Tu<:KOmegaLKE,E,D,BI}
    args = (
        ("U", model.momentum.U), 
        ("p", model.momentum.p)
    )
    write_vtk(name, model.domain, args...)
end