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

struct KOmegaCoefficients{T}
    β⁺::T
    α1::T
    β1::T
    σk::T
    σω::T
end

get_coeffs(FloatType) = begin
    KOmegaCoefficients{FloatType}(
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

    @. Pω.values = coeffs.α1*Pk.values
    @. Dωf.values = coeffs.β1*omega.values

    interpolate!(kf, k)
    correct_boundaries!(kf, k, k.BCs)
    interpolate!(ωf, omega)
    correct_boundaries!(ωf, omega, omega.BCs)
    diffusion_flux!(nueffω, nu, kf, ωf, coeffs.σω)

    # update k fluxes

    @. Dkf.values = coeffs.β⁺*omega.values

    # interpolate!(kf, k)
    # correct_boundaries!(kf, k, k.BCs)
    # interpolate!(ωf, omega)
    # correct_boundaries!(ωf, omega, omega.BCs)
    diffusion_flux!(nueffk, nu, kf, ωf, coeffs.σk)

    # Solve omega equation

    discretise!(ω_eqn)
    apply_boundary_conditions!(ω_eqn, omega.BCs)
    prev .= omega.values
    implicit_relaxation!(ω_eqn.equation, prev, solvers.omega.relax)
    constrain_equation!(ω_eqn, omega.BCs, model) # active with WFs only
    update_preconditioner!(ω_eqn.preconditioner)
    run!(ω_eqn, solvers.omega)
   
    constrain_boundary!(omega, omega.BCs, model) # active with WFs only
    bound!(omega, eps())

    # Solve k equation

    @. Pk.values = nut.values*Pk.values
    correct_production!(Pk, k.BCs, model)

    discretise!(k_eqn)
    apply_boundary_conditions!(k_eqn, k.BCs)
    prev .= k.values
    implicit_relaxation!(k_eqn.equation, prev, solvers.k.relax)
    update_preconditioner!(k_eqn.preconditioner)
    run!(k_eqn, solvers.k)
    bound!(k, eps())

    update_eddy_viscosity!(nut, k, omega)
    # correct_production!(Pk, k.BCs, model)

    # correct_eddy_viscosity!(nut, nut.BCs, model) # to implement
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

y_plus_laminar(B, kappa) = begin
    yL = 11.0; for i ∈ 1:10; yL = log(yL*B)/kappa; end
    yL
end

ω_vis(nu, y, beta1) = 6.0*nu/(beta1*y^2)

ω_log(k, y, cmu, kappa) = sqrt(k)/(cmu^0.25*kappa*y)
# ω_log(k, y, cmu, kappa) = sqrt(k)/(cmu*kappa*y)

y_plus(k, nu, y, cmu) = cmu^0.25*y*sqrt(k)/nu

sngrad(Ui, Uw, delta, normal) = begin
    Udiff = (Ui - Uw)
    Up = Udiff - (Udiff⋅normal)*normal # oarallel velocity difference
    grad = Up/delta
    return grad
end

mag(vector) = sqrt(vector[1]^2 + vector[2]^2 + vector[3]^2) 

# nut_wall(nu, yP, kappa, E) = nu*(
#     yP*kappa/log(max(E*yP, 1.0 + 1e-4)) - 1.0) # E = 9.8 E*yP, 1 + 1e-4)

# nut_wall(k, delta, yplus, kappa, cmu, B) = begin
nut_wall(nu, yplus, kappa, E) = begin
    # cmu^0.25*sqrt(k)*delta/(log(yplus)/kappa + B)
    max(nu*(yplus*kappa/log(E*yplus) - 1), zero(typeof(E)))
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
    ylam = y_plus_laminar(B, kappa) # must add B to KomegaWF type
    ωc = zero(_get_float(mesh))
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
        ωvis = ω_vis(nu[cID], y, beta1)
        ωlog = ω_log(k[cID], y, cmu, kappa)
        yplus = y_plus(k[cID], nu[cID], y, cmu) 

        if yplus > ylam 
            ωc = ωlog
        else
            ωc = ωvis
        end

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

@generated constrain_boundary!(field, fieldBCs, model) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: OmegaWallFunction  #|| BC <: Dirichlet
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
    (; kappa, beta1, cmu, B) = BC.value
    mesh = field.mesh
    (; faces, cells, boundaries) = mesh
    boundary = boundaries[ID]
    (; cellsID, facesID) = boundary
    ylam = y_plus_laminar(B, kappa) # must add B to KomegaWF type
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
    ylam = y_plus_laminar(B, kappa) # must add B to KomegaWF type
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
        nut[cID] = nutw
        mag_grad_U = mag(sngrad(U[cID], Uw, delta, normal))
        if yplus > ylam
            # P.values[cID] = (nu[cID] + nutw)*mag_grad_U*dUdy
            P.values[cID] = (nu[cID] + nutw)*mag_grad_U*dUdy
        else
            P.values[cID] = zero(_get_float(mesh))
        end
    end
end

