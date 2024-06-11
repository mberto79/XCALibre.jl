export KOmega
export initialise_RANS
export turbulence!

# Constructor 

# RANS{KOmega}(; mesh, viscosity) = begin
#     U = VectorField(mesh); F1 = typeof(U)
#     p = ScalarField(mesh); F2 = typeof(p)
#     V = typeof(viscosity)
#     k = ScalarField(mesh); omega = ScalarField(mesh); nut = ScalarField(mesh)
#     turb = (k=k , omega=omega, nut=nut); T = typeof(turb)
#     flag = false; E = typeof(flag)
#     D = typeof(mesh)
#     boundary_info = @time begin boundary_map(mesh) end; BI = typeof(boundary_info)
#     RANS{KOmega,F1,F2,V,T,E,D,BI}(
#         KOmega(), U, p, viscosity, turb, flag, mesh, boundary_info
#     )
# end
struct KOmega <: AbstractTurbulenceModel
    k
    omega
    nut
    coeffs
    kf
    ωf
    νtf
    k_eqn
    ω_eqn
end
Adapt.@adapt_structure KOmega

RANS{KOmega}(mesh) = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    nut = ScalarField(mesh)
    coeffs = (β⁺=0.09, α1=0.52, β1=0.072, σk=0.5, σω=0.5)
    kf = FaceScalarField(mesh)
    ωf = FaceScalarField(mesh)
    νtf = FaceScalarField(mesh)
    KOmega(k, omega, nut, coeffs, kf, ωf, νtf, (), () )
end


# struct KOmegaCoefficients{T}
#     β⁺::T
#     α1::T
#     β1::T
#     σk::T
#     σω::T
# end
# Adapt.@adapt_structure KOmegaCoefficients

# get_coeffs(FloatType) = begin
#     KOmegaCoefficients{FloatType}(
#         0.09,
#         0.52, #5/9,
#         0.072, #3/40,
#         0.5,
#         0.5    
#     )
# end

# struct KOmegaModel{MK,MW,FK,FW,FN,C,S}
#     k_eqn::MK
#     ω_eqn::MW
#     kf::FK
#     ωf::FW
#     νtf::FN
#     coeffs::C
#     config::S
# end
# Adapt.@adapt_structure KOmegaModel

function initialise_RANS(
    model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu<:KOmega,E,D,BI}
    # unpack turbulent quantities and configuration
    turbulence = model.turbulence
    (; k, omega, nut) = turbulence
    (; solvers, schemes, runtime) = config
    mesh = mdotf.mesh
    eqn = peqn.equation

    # kf = FaceScalarField(mesh)
    # ωf = FaceScalarField(mesh)
    # νtf = FaceScalarField(mesh)
    
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
            # Source(Pk) - Source(Dkf)
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

    # update equations in model (originally empty tuple)
    @reset model.turbulence.k_eqn = k_eqn 
    @reset model.turbulence.ω_eqn = ω_eqn

    # float_type = _get_float(mesh)
    # coeffs = get_coeffs(float_type)

    return model
end

function turbulence!( # Sort out dispatch when possible
    model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, config
    ) where {T,F,M,Tu<:KOmega,E,D,BI}

    nu = _nu(model.fluid)
    nut = model.turbulence.nut

    (;k, omega, k_eqn, ω_eqn, kf, ωf, νtf, coeffs) = model.turbulence
    (; solvers, runtime) = config

    # k = get_phi(k_eqn)
    # omega = get_phi(ω_eqn)

    nueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    # Dkf = get_source(k_eqn, 2)
    Pk = get_source(k_eqn, 1)

    nueffω = get_flux(ω_eqn, 3)
    Dωf = get_flux(ω_eqn, 4)
    Pω = get_source(ω_eqn, 1)

    mesh = k.mesh

    # double_inner_product!(Pk, S, S.gradU)
    # for i ∈ eachindex(Pk)
    #     Pk[i] *= 2.0
    # end

    magnitude2!(Pk, S, config, scale_factor=2.0) # multiplied by 2 (def of Sij)

    constrain_boundary!(omega, omega.BCs, model, config) # active with WFs only

    correct_production!(Pk, k.BCs, model, S.gradU, config)
    # @. Pω.values = coeffs.α1*Pk.values/nut.values
    @. Pω.values = coeffs.α1*Pk.values
    @. Pk.values = nut.values*Pk.values
    @. Dωf.values = coeffs.β1*omega.values
    @. nueffω.values = nu.values + coeffs.σω*νtf.values

    # update k fluxes

    @. Dkf.values = coeffs.β⁺*omega.values
    # @. Dkf.values = coeffs.β⁺*omega.values*k.values
    @. nueffk.values = nu.values + coeffs.σk*νtf.values
    # @. Pk.values = nut.values*Pk.values
    # correct_production!(Pk, k.BCs, model, config)

    # Solve omega equation

    prev .= omega.values
    discretise!(ω_eqn, omega, config)
    apply_boundary_conditions!(ω_eqn, omega.BCs, nothing, config)
    constrain_equation!(ω_eqn, omega.BCs, model, config) # active with WFs only
    implicit_relaxation!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    update_preconditioner!(ω_eqn.preconditioner, mesh, config)
    solve_system!(ω_eqn, solvers.omega, omega, nothing, config)
   
    constrain_boundary!(omega, omega.BCs, model, config) # active with WFs only
    bound!(omega, config)

    # Solve k equation
    
    prev .= k.values
    discretise!(k_eqn, k, config)
    apply_boundary_conditions!(k_eqn, k.BCs, nothing, config)
    implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing, config)
    update_preconditioner!(k_eqn.preconditioner, mesh, config)
    solve_system!(k_eqn, solvers.k, k, nothing, config)
    bound!(k, config)

    @. nut.values = k.values/omega.values

    interpolate!(νtf, nut, config)
    correct_boundaries!(νtf, nut, nut.BCs, config)
    correct_eddy_viscosity!(νtf, nut.BCs, model, config)
end

y_plus_laminar(E, kappa) = begin
    yL = 11.0; for i ∈ 1:10; yL = log(max(yL*E, 1.0))/kappa; end
    yL
end

ω_vis(nu, y, beta1) = 6*nu/(beta1*y^2)

ω_log(k, y, cmu, kappa) = sqrt(k)/(cmu^0.25*kappa*y)

y_plus(k, nu, y, cmu) = cmu^0.25*y*sqrt(k)/nu

sngrad(Ui, Uw, delta, normal) = begin
    Udiff = (Ui - Uw)
    Up = Udiff - (Udiff⋅normal)*normal # parallel velocity difference
    grad = Up/delta
    return grad
end

mag(vector) = sqrt(vector[1]^2 + vector[2]^2 + vector[3]^2) 

nut_wall(nu, yplus, kappa, E::T) where T = begin
    max(nu*(yplus*kappa/log(max(E*yplus, 1.0 + 1e-4)) - 1), zero(T))
end

@generated constrain_equation!(eqn, fieldBCs, model, config) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: OmegaWallFunction
            call = quote
                constrain!(eqn, fieldBCs[$i], model, config)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

function constrain!(eqn, BC, model, config)

    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Access equation data and deconstruct sparse array
    A = _A(eqn)
    b = _b(eqn, nothing)
    rowval = _rowval(A)
    colptr = _colptr(A)
    nzval = _nzval(A)
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundaries, boundary_cellsID) = mesh

    facesID_range = get_boundaries(BC, boundaries)
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    kernel! = _constrain!(backend, workgroup)
    kernel!(
        model, BC, faces, start_ID, boundary_cellsID, rowval, colptr, nzval, b, ndrange=length(facesID_range)
    )
end

@kernel function _constrain!(model, BC, faces, start_ID, boundary_cellsID, rowval, colptr, nzval, b)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    @uniform begin
        nu = _nu(model.fluid)
        k = model.turbulence.k
        (; kappa, beta1, cmu, B, E) = BC.value
        ylam = y_plus_laminar(E, kappa)
    end
    ωc = zero(eltype(nzval))
    
    @inbounds begin
        cID = boundary_cellsID[fID]
        face = faces[fID]
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
        # b[cID] += A[cID,cID]*ωc
        # A[cID,cID] += A[cID,cID]
        
        nzIndex = spindex(colptr, rowval, cID, cID)
        Atomix.@atomic b[cID] += nzval[nzIndex]*ωc
        Atomix.@atomic nzval[nzIndex] += nzval[nzIndex] 
    end
end

@generated constrain_boundary!(field, fieldBCs, model, config) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: OmegaWallFunction
            call = quote
                set_cell_value!(field, fieldBCs[$i], model, config)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

function set_cell_value!(field, BC, model, config)
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundaries, boundary_cellsID) = mesh

    facesID_range = get_boundaries(BC, boundaries)
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    kernel! = _set_cell_value!(backend, workgroup)
    kernel!(
        field, model, BC, faces, start_ID, boundary_cellsID, ndrange=length(facesID_range)
    )
end

@kernel function _set_cell_value!(field, model, BC, faces, start_ID, boundary_cellsID)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    @uniform begin
        nu = _nu(model.fluid)
        k = model.turbulence.k
        (; kappa, beta1, cmu, B, E) = BC.value
        (; values) = field
        ylam = y_plus_laminar(E, kappa)
    end
    ωc = zero(eltype(values))

    @inbounds begin
        cID = boundary_cellsID[fID]
        face = faces[fID]
        y = face.delta
        ωvis = ω_vis(nu[cID], y, beta1)
        ωlog = ω_log(k[cID], y, cmu, kappa)
        yplus = y_plus(k[cID], nu[cID], y, cmu) 

        if yplus > ylam 
            ωc = ωlog
        else
            ωc = ωvis
        end

        values[cID] = ωc # needs to be atomic?
    end
end

@generated correct_production!(P, fieldBCs, model, gradU, config) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: KWallFunction
            call = quote
                set_production!(P, fieldBCs[$i], model, gradU, config)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

function set_production!(P, BC, model, gradU, config)
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundary_cellsID, boundaries) = mesh

    facesID_range = get_boundaries(BC, boundaries)
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    kernel! = _set_production!(backend, workgroup)
    kernel!(
        P.values, BC, model, faces, boundary_cellsID, start_ID, gradU, ndrange=length(facesID_range)
    )
end

@kernel function _set_production!(values, BC, model, faces, boundary_cellsID, start_ID, gradU)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    (; kappa, beta1, cmu, B, E) = BC.value
    (; U, nu) = model
    (; k, nut) = model.turbulence

    ylam = y_plus_laminar(E, kappa)
    # Uw = SVector{3,_get_float(mesh)}(0.0,0.0,0.0)
    Uw = SVector{3}(0.0,0.0,0.0)
        cID = boundary_cellsID[fID]
        face = faces[fID]
        nuc = nu[cID]
        (; delta, normal)= face
        uStar = cmu^0.25*sqrt(k[cID])
        dUdy = uStar/(kappa*delta)
        yplus = y_plus(k[cID], nuc, delta, cmu)
        nutw = nut_wall(nuc, yplus, kappa, E)
        # mag_grad_U = mag(sngrad(U[cID], Uw, delta, normal))
        mag_grad_U = mag(gradU[cID]*normal)
        if yplus > ylam
            values[cID] = (nu[cID] + nutw)*mag_grad_U*dUdy
        else
            values[cID] = 0.0
        end
end

@generated correct_eddy_viscosity!(νtf, nutBCs, model, config) = begin
    BCs = nutBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        BC = BCs[i]
        if BC <: NutWallFunction
            call = quote
                correct_nut_wall!(νtf, nutBCs[$i], model, config)
            end
            push!(func_calls, call)
        end
    end
    quote
    $(func_calls...)
    nothing
    end 
end

function correct_nut_wall!(νtf, BC, model, config)
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    # Deconstruct mesh to required fields
    mesh = model.mesh
    (; faces, boundary_cellsID, boundaries) = mesh

    facesID_range = get_boundaries(BC, boundaries)
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    kernel! = _correct_nut_wall!(backend, workgroup)
    kernel!(
        νtf.values, model, BC, faces, boundary_cellsID, start_ID, ndrange=length(facesID_range)
    )
end

@kernel function _correct_nut_wall!(values, model, BC, faces, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    (; kappa, beta1, cmu, B, E) = BC.value
    (; nu) = model
    (; k) = model.turbulence
    
    ylam = y_plus_laminar(E, kappa)
        cID = boundary_cellsID[fID]
        face = faces[fID]
        # nuf = nu[fID]
        (; delta)= face
        # yplus = y_plus(k[cID], nuf, delta, cmu)
        nuc = nu[cID]
        yplus = y_plus(k[cID], nuc, delta, cmu)
        nutw = nut_wall(nuc, yplus, kappa, E)
        if yplus > ylam
            values[fID] = nutw
        else
            values[fID] = 0.0
        end
end

