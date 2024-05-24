export KOmega
export initialise_RANS
export turbulence!

struct KOmega <: AbstractTurbulenceModel end
Adapt.@adapt_structure KOmega
# Constructor 

RANS{KOmega}(; mesh, viscosity) = begin
    U = VectorField(mesh); F1 = typeof(U)
    p = ScalarField(mesh); F2 = typeof(p)
    V = typeof(viscosity)
    k = ScalarField(mesh); omega = ScalarField(mesh); nut = ScalarField(mesh)
    turb = (k=k , omega=omega, nut=nut); T = typeof(turb)
    flag = false; E = typeof(flag)
    D = typeof(mesh)
    boundary_info = @time begin boundary_map(mesh) end; BI = typeof(boundary_info)
    RANS{KOmega,F1,F2,V,T,E,D,BI}(
        KOmega(), U, p, viscosity, turb, flag, mesh, boundary_info
    )
end

struct KOmegaCoefficients{T}
    β⁺::T
    α1::T
    β1::T
    σk::T
    σω::T
end
Adapt.@adapt_structure KOmegaCoefficients

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
Adapt.@adapt_structure KOmegaModel

function initialise_RANS(mdotf, peqn, config, model)
    # unpack turbulent quantities and configuration
    turbulence = model.turbulence
    (; k, omega, nut) = turbulence
    (; solvers, schemes, runtime) = config
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
    ) → eqn

    # Set up preconditioners
    @reset k_eqn.preconditioner = set_preconditioner(
                solvers.k.preconditioner, k_eqn, k.BCs, config)

    @reset ω_eqn.preconditioner = set_preconditioner(
                solvers.omega.preconditioner, ω_eqn, omega.BCs, config)
    
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
    KOmega::KOmegaModel, model, S, S2, prev, config)

    nu = model.nu
    nut = model.turbulence.nut

    (;k_eqn, ω_eqn, kf, ωf, νtf, coeffs, config) = KOmega
    (; solvers, runtime) = config

    k = get_phi(k_eqn)
    omega = get_phi(ω_eqn)

    nueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    Pk = get_source(k_eqn, 1)

    nueffω = get_flux(ω_eqn, 3)
    Dωf = get_flux(ω_eqn, 4)
    Pω = get_source(ω_eqn, 1)

    mesh = k.mesh

    # double_inner_product!(Pk, S, S.gradU)
    # for i ∈ eachindex(Pk)
    #     Pk[i] *= 2.0
    # end

    magnitude2!(Pk, S, scale_factor=2.0) # multiplied by 2 (def of Sij)

    @. Pω.values = coeffs.α1*Pk.values
    @. Dωf.values = coeffs.β1*omega.values
    diffusion_flux!(nueffω, nu, νtf, coeffs.σω)

    # update k fluxes

    @. Dkf.values = coeffs.β⁺*omega.values
    diffusion_flux!(nueffk, nu, νtf, coeffs.σk)
    @. Pk.values = nut.values*Pk.values
    correct_production!(Pk, k.BCs, model)

    # Solve omega equation

    prev .= omega.values
    discretise!(ω_eqn, omega, config)
    apply_boundary_conditions!(ω_eqn, omega.BCs, nothing, config)
    implicit_relaxation!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    constrain_equation!(ω_eqn, omega.BCs, model) # active with WFs only
    update_preconditioner!(ω_eqn.preconditioner, mesh, config)
    run!(ω_eqn, solvers.omega, omega, nothing, config)
   
    constrain_boundary!(omega, omega.BCs, model) # active with WFs only
    bound!(omega, eps())

    # Solve k equation
    
    prev .= k.values
    discretise!(k_eqn, k, config)
    apply_boundary_conditions!(k_eqn, k.BCs, nothing, config)
    implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing, config)
    update_preconditioner!(k_eqn.preconditioner, mesh, config)
    run!(k_eqn, solvers.k, k, nothing, config)
    bound!(k, eps())

    update_eddy_viscosity!(nut, k, omega)
    
    interpolate!(νtf, nut, config)
    correct_boundaries!(νtf, nut, nut.BCs, config)
    correct_eddy_viscosity!(νtf, nut.BCs, model, config)
end

update_eddy_viscosity!(nut::F, k, omega) where F<:AbstractScalarField = begin
    for i ∈ eachindex(nut)
        nut[i] = k[i]/omega[i]
    end
end

diffusion_flux!(nueff, nu, νtf::F, σ) where F<:FaceScalarField = begin
    @inbounds for i ∈ eachindex(nueff)
        nueff[i] = nu[i] + σ*νtf[i]
    end
end

destruction_flux!(Dxf::F, coeff, omega) where F<:ScalarField = begin
    for i ∈ eachindex(Dxf.values)
        Dxf[i] = coeff*omega[i]
    end
end

y_plus_laminar(E, kappa) = begin
    yL = 11.0; for i ∈ 1:10; yL = log(max(yL*E, 1.0))/kappa; end
    yL
end

ω_vis(nu, y, beta1) = 6.0*nu/(beta1*y^2)

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
    # (; faces, cells, boundaries) = mesh
    (; faces, boundary_cellsID, boundaries) = mesh
    (; A, b) = eqn.equation
    # boundary = boundaries[ID]
    IDs_range = boundaries[ID].IDs_range
    # (; cellsID, facesID) = boundary
    ylam = y_plus_laminar(E, kappa)
    ωc = zero(_get_float(mesh))
    # for i ∈ eachindex(cellsID)
    for fID ∈ IDs_range
        # cID = cellsID[i]
        cID = boundary_cellsID[fID]
        # fID = facesID[i]
        face = faces[fID]
        # cell = cells[cID]
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
    # (; faces, cells, boundaries) = mesh
    (; faces, boundary_cellsID, boundaries) = mesh
    # boundary = boundaries[ID]
    # (; cellsID, facesID) = boundary
    IDs_range = boundaries[ID].IDs_range
    ylam = y_plus_laminar(E, kappa)
    ωc = zero(_get_float(mesh))
    for fID ∈ IDs_range
        # cID = cellsID[i]
        cID = boundary_cellsID[fID]
        # fID = facesID[i]
        face = faces[fID]
        # cell = cells[cID]
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
    # (; faces, cells, boundaries) = mesh
    (; faces, boundary_cellsID, boundaries) = mesh
    # boundary = boundaries[ID]
    # (; cellsID, facesID) = boundary
    IDs_range = boundaries[ID].IDs_range
    ylam = y_plus_laminar(E, kappa)
    Uw = SVector{3,_get_float(mesh)}(0.0,0.0,0.0)
    for fID ∈ IDs_range
        # cID = cellsID[i]
        cID = boundary_cellsID[fID]
        # fID = facesID[i]
        face = faces[fID]
        # cell = cells[cID]
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

@generated correct_eddy_viscosity!(νtf, nutBCs, model, config) = begin
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

