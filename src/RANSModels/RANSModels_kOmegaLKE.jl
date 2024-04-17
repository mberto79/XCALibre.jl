export KOmegaLKE
export initialise_RANS_LKE
export turbulence!

struct KOmegaLKE <: AbstractTurbulenceModel end

# Constructor 

RANS{KOmegaLKE}(; mesh, viscosity, Tu) = begin
    U = VectorField(mesh); F1 = typeof(U)
    p = ScalarField(mesh); F2 = typeof(p)
    phi = ScalarField(mesh); F3 = typeof(p)
    y = ScalarField(mesh); F4 = typeof(p)
    V = typeof(viscosity)
    kL = ScalarField(mesh); k = ScalarField(mesh); omega = ScalarField(mesh); nut = ScalarField(mesh)
    turb = (kL=kL, k=k, omega=omega, nut=nut, Tu = Tu); T = typeof(turb)
    flag = false; F = typeof(flag)
    D = typeof(mesh)
    RANS{KOmegaLKE,F1,F2,F3,F4,V,T,F,D}(
        KOmegaLKE(), U, p, phi, y, viscosity, turb, flag, mesh
    )
end

struct KOmegaLKECoefficients{T}
    C1::T
    C2::T
    C3::T
    C4::T
    Cμ::T
    Cω1::T
    Cω2::T
    Ccrit::T
    CSS::T
    Cv::T
    σk::T
    σd::T
    σkL::T
    σω::T
end

get_LKE_coeffs(FloatType) = begin
    KOmegaLKECoefficients{FloatType}(
        0.02974,
        59.79,
        1.191,
        1.65*10^-13,
        0.09,
        0.52, #13/25
        0.0708,
        76500,
        1.45,
        0.43,
        0.5,
        0.125,
        0.0125,
        0.5
    ) #Medina 2018
end

struct KOmegaLKEModel{ML,MK,MW,EL,EK,EO,FK,FW,FN,NL,NS,NT,O,Y,D,NK,NO,NU,RV,C,S}
    kL_eqn::ML
    k_eqn::MK
    ω_eqn::MW
    nueffkLS::EL
    nueffkS::EK
    nueffωS::EO
    kf::FK
    ωf::FW
    νtf::FN
    nuL::NL
    nuts::NS
    nut_turb::NT
    Ω::O
    γ::Y
    fv::D
    ∇k::NK
    ∇ω::NO
    normU::NU
    Reυ::RV
    coeffs::C
    config::S
end

function initialise_RANS(mdotf, peqn, config, model::RANS{KOmegaLKE})
    @info "Initialising k-ω LKE model..."
    # unpack turbulent quantities and configuration
    turbulence = model.turbulence
    (; kL, k, omega) = turbulence
    (; solvers, schemes, runtime) = config
    mesh = mdotf.mesh
    eqn = peqn.equation

    calc_wall_distance!(model, config)

    kf = FaceScalarField(mesh)
    ωf = FaceScalarField(mesh)
    νtf = FaceScalarField(mesh)
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
    nut_turb = ScalarField(mesh)
    Ω = ScalarField(mesh)
    γ = ScalarField(mesh)
    fv = ScalarField(mesh)
    # ∇k = VectorField(mesh)
    # ∇ω = VectorField(mesh)
    ∇k = Grad{schemes.k.gradient}(k)
    ∇ω = Grad{schemes.p.gradient}(omega)

    kL_eqn = (
            Time{schemes.kL.time}(kL)
            + Divergence{schemes.kL.divergence}(mdotf, kL) 
            - Laplacian{schemes.kL.laplacian}(nueffkL, kL) 
            + Si(DkLf,kL) # Dkf = β⁺*omega
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

    @reset kL_eqn.preconditioner = set_preconditioner(
                solvers.kL.preconditioner, kL_eqn, kL.BCs, runtime)

    @reset k_eqn.preconditioner = set_preconditioner(
                solvers.k.preconditioner, k_eqn, k.BCs, runtime)

    @reset ω_eqn.preconditioner = set_preconditioner(
                solvers.omega.preconditioner, ω_eqn, omega.BCs, runtime)
    
    # preallocating solvers

    @reset kL_eqn.solver = solvers.kL.solver(_A(kL_eqn), _b(kL_eqn))
    @reset k_eqn.solver = solvers.k.solver(_A(k_eqn), _b(k_eqn))
    @reset ω_eqn.solver = solvers.omega.solver(_A(ω_eqn), _b(ω_eqn))

    grad!(∇ω,ωf,omega,omega.BCs)
    grad!(∇k,kf,k,k.BCs)

    float_type = _get_float(mesh)
    coeffs = get_LKE_coeffs(float_type)

    return KOmegaLKEModel(
        kL_eqn,
        k_eqn,
        ω_eqn,
        nueffkLS,
        nueffkS,
        nueffωS,
        kf,
        ωf,
        νtf,
        nuL,
        nuts,
        nut_turb,
        Ω,
        γ,
        fv,
        ∇k,
        ∇ω,
        normU,
        Reυ,
        coeffs,
        config
    )
end

function turbulence!( # Sort out dispatch when possible
    KOmegaLKE::KOmegaLKEModel, model, S, S2, gradU, prev)
    (;nu, U, y, turbulence) = model
    (;Tu, nut) = turbulence
    
    (;kL_eqn, k_eqn, ω_eqn, nueffkLS, nueffkS, nueffωS, kf, ωf, νtf, nuL, nuts, nut_turb, Ω, γ, fv, ∇k, ∇ω, normU, Reυ, coeffs, config) = KOmegaLKE
    (; solvers, runtime) = config

    kL = get_phi(kL_eqn)
    k = get_phi(k_eqn)
    omega = get_phi(ω_eqn)

    nueffkL = get_flux(kL_eqn, 3)
    DkLf = get_flux(kL_eqn, 4)
    PkL = get_source(kL_eqn, 1)

    nueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    Pk = get_source(k_eqn, 1)

    nueffω = get_flux(ω_eqn, 3)
    Dωf = get_flux(ω_eqn, 4)
    Pω = get_source(ω_eqn, 1)
    dkdomegadx = get_source(ω_eqn, 2) # cross diffusion term

    #Update ω fluxes
    double_inner_product!(Pk, S, gradU) # multiplied by 2 (def of Sij) (Pk = S² at this point)
    inner_product!(dkdomegadx,∇k,∇ω)
    @. Pω.values = coeffs.Cω1*Pk.values*nut.values*(omega.values/k.values)
    @. dkdomegadx.values = (coeffs.σd/omega.values)*dkdomegadx.values
    @. Dωf.values = coeffs.Cω2*omega.values
    @. nueffωS.values = nu.values+(coeffs.σω*nut_turb.values*γ.values)
    interpolate!(nueffω,nueffωS)
    correct_boundaries!(nueffω, nueffωS, nut.BCs)

    #Damping and trigger
    @. fv.values = 1-exp(sqrt(k.values/(nu.values*omega.values))/coeffs.Cv)
    @. γ.values = min((kL.values/(min(nu.values*nuL.values)*Ω.values))^2,coeffs.Ccrit)/coeffs.Ccrit

    #Update k fluxes
    @. Dkf.values = coeffs.Cμ*omega.values*γ.values
    @. nueffkS.values = nu.values+(coeffs.σk*nut_turb.values*γ.values)
    interpolate!(nueffk,nueffkS)
    correct_boundaries!(nueffk, nueffkS, nut.BCs)
    @. Pk.values = nut.values*Pk.values*γ.values*fv.values
    correct_production!(Pk, k.BCs, model)

    #Update kL fluxes
    magnitude!(PkL, S,scale_factor=2.0)
    # magnitude!(normU,U) # tim's
    for i ∈ eachindex(normU.values)
        normU.values[i] = norm(U[i])
    end
    #norm!(normU, U)
    η = coeffs.C1*tanh(coeffs.C2*(Tu^coeffs.C3)+coeffs.C4)
    @. Reυ.values = ((((2*kL.values*(nu.values^2))/(y.values^2))^(1/4))*y.values)/nu.values
    @. PkL.values = PkL.values*η*kL.values*(Reυ.values^(-13/10))*(((normU.values*y.values)/nu.values)^(1/2))
    @. DkLf.values = (2*nu.values*kL.values)/(y.values^2)
    @. nueffkLS.values = nu.values+(coeffs.σkL*sqrt(kL.values)*y.values)
    interpolate!(nueffkL,nueffkLS)
    correct_boundaries!(nueffkL, nueffkLS, nut.BCs)

    # Solve kL equation
    prev .= kL.values
    discretise!(kL_eqn, prev, runtime)
    apply_boundary_conditions!(kL_eqn, kL.BCs)
    implicit_relaxation!(kL_eqn.equation, prev, solvers.kL.relax)
    update_preconditioner!(kL_eqn.preconditioner)
    run!(kL_eqn, solvers.kL)
    bound!(kL, eps())

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
    magnitude2!(S2, S, scale_factor=2.0)
    double_inner_product!(Ω,S,S, scale_factor=2.0)
    @. nut_turb.values = k.values/omega.values
    @. nuL.values = PkL.values/max(S2.values,(normU.values/y.values)^2)
    @. nuts.values = exp(-(coeffs.CSS/(k.values/(nu.values*Ω.values)))^2)*(k.values/omega.values)
    @. nut.values = nuts.values+nuL.values

    # #Damping and trigger
    # @. fv.values = 1-exp(sqrt(k.values/(nu.values*omega.values))/coeffs.Cv)
    # @. γ.values = min((kL.values/(min(nu.values*nuL.values)*Ω.values))^2,coeffs.Ccrit)/coeffs.Ccrit

    # @. kL.values = Reυ.values^(-13/10)

    interpolate!(νtf, nut)
    correct_boundaries!(νtf, nut, nut.BCs)
    correct_eddy_viscosity!(νtf, nut.BCs, model)
end

inner_product!(S::F, ∇1::Grad, ∇2::Grad) where F<:ScalarField = begin
    for i ∈ eachindex(S.values)
        S[i] = ∇1[i]⋅∇2[i]
    end
end

#=
y_plus_laminar(E, kappa) = begin
    yL = 11.0; for i ∈ 1:10; yL = log(max(yL*E, 1.0))/kappa; end
    yL
end

ω_vis(nu, y, beta1) = 6.0*nu/(Cω2*y^2)

ω_log(k, y, cmu, kappa) = sqrt(k)/(cmu^0.25*kappa*y)

y_plus(k, nu, y, cmu) = cmu^0.25*y*sqrt(k)/nu

sngrad(Ui, Uw, delta, normal) = begin
    Udiff = (Ui - Uw)
    Up = Udiff - (Udiff⋅normal)*normal # parallel velocity difference
    grad = Up/delta
    return grad
end

mag(vector) = sqrt(vector[1]^2 + vector[2]^2 + vector[3]^2) 

nut_wall(nu, yplus, kappa, E) = begin
    max(nu*(yplus*kappa/log(max(E*yplus, 1.0 + 1e-4)) - 1), zero(typeof(E)))
end

@generated constrain_equation!(eqn, fieldBCs, model) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: OmegaWallFunction
            call = quote
                constraint!(eqn, fieldBCs[$i], model)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

constraint!(eqn, BC, model) = begin
    ID = BC.ID
    nu = model.nu
    k = model.turbulence.k
    (; kappa, beta1, cmu, B, E) = BC.value
    field = get_phi(eqn)
    mesh = field.mesh
    (; faces, cells, boundaries) = mesh
    (; A, b) = eqn.equation
    boundary = boundaries[ID]
    (; cellsID, facesID) = boundary
    ylam = y_plus_laminar(E, kappa)
    ωc = zero(_get_float(mesh))
    for i ∈ eachindex(cellsID)
        cID = cellsID[i]
        fID = facesID[i]
        face = faces[fID]
        cell = cells[cID]
        y = face.delta
        ωvis = ω_vis(nu[cID], y, beta1)
        ωlog = ω_log(k[cID], y, cmu, kappa)
        yplus = y_plus(k[cID], nu[cID], y, cmu) 

        if yplus > ylam 
            ωc = ωlog
        else
            ωc = ωvis
        end
        # Line below is weird but worked
        # b[cID] = A[cID,cID]*ωc

        # Classic approach
        b[cID] += A[cID,cID]*ωc
        A[cID,cID] += A[cID,cID]
    end
end

@generated constrain_boundary!(field, fieldBCs, model) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: OmegaWallFunction
            call = quote
                set_cell_value!(field, fieldBCs[$i], model)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

set_cell_value!(field, BC, model) = begin
    ID = BC.ID
    nu = model.nu
    k = model.turbulence.k
    (; kappa, beta1, cmu, B, E) = BC.value
    mesh = field.mesh
    (; faces, cells, boundaries) = mesh
    boundary = boundaries[ID]
    (; cellsID, facesID) = boundary
    ylam = y_plus_laminar(E, kappa)
    ωc = zero(_get_float(mesh))
    for i ∈ eachindex(cellsID)
        cID = cellsID[i]
        fID = facesID[i]
        face = faces[fID]
        cell = cells[cID]
        y = face.delta
        ωvis = ω_vis(nu[cID], y, beta1)
        ωlog = ω_log(k[cID], y, cmu, kappa)
        yplus = y_plus(k[cID], nu[cID], y, cmu) 

        if yplus > ylam 
            ωc = ωlog
        else
            ωc = ωvis
        end

        field.values[cID] = ωc
    end
end

@generated correct_production!(P, fieldBCs, model) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: KWallFunction
            call = quote
                set_production!(P, fieldBCs[$i], model)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

set_production!(P, BC, model) = begin
    ID = BC.ID
    (; kappa, beta1, cmu, B, E) = BC.value
    (; U, nu, mesh) = model
    (; k, nut) = model.turbulence
    (; faces, cells, boundaries) = mesh
    boundary = boundaries[ID]
    (; cellsID, facesID) = boundary
    ylam = y_plus_laminar(E, kappa)
    Uw = SVector{3,_get_float(mesh)}(0.0,0.0,0.0)
    for i ∈ eachindex(cellsID)
        cID = cellsID[i]
        fID = facesID[i]
        face = faces[fID]
        cell = cells[cID]
        nuc = nu[cID]
        (; delta, normal)= face
        uStar = cmu^0.25*sqrt(k[cID])
        dUdy = uStar/(kappa*delta)
        yplus = y_plus(k[cID], nuc, delta, cmu)
        nutw = nut_wall(nuc, yplus, kappa, E)
        mag_grad_U = mag(sngrad(U[cID], Uw, delta, normal))
        if yplus > ylam
            P.values[cID] = (nu[cID] + nutw)*mag_grad_U*dUdy
        end
    end
end

@generated correct_eddy_viscosity!(νtf, nutBCs, model) = begin
    BCs = nutBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: NutWallFunction
            call = quote
                correct_nut_wall!(νtf, nutBCs[$i], model)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

correct_nut_wall!(νtf, BC, model) = begin
    ID = BC.ID
    (; kappa, beta1, cmu, B, E) = BC.value
    (; U, nu, mesh) = model
    (; k, omega, nut) = model.turbulence
    (; faces, cells, boundaries) = mesh
    boundary = boundaries[ID]
    (; cellsID, facesID) = boundary
    ylam = y_plus_laminar(E, kappa)
    for i ∈ eachindex(cellsID)
        cID = cellsID[i]
        fID = facesID[i]
        face = faces[fID]
        cell = cells[cID]
        nuf = nu[fID]
        (; delta, normal)= face
        yplus = y_plus(k[cID], nuf, delta, cmu)
        nutw = nut_wall(nuf, yplus, kappa, E)
        if yplus > ylam
            νtf.values[fID] = nutw
        end
    end
end
=#