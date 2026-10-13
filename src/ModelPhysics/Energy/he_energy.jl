export SensibleEnthalpy, InternalEnergy
export temperature_to_energy!, energy_to_temperature!, thermo_Psi!, energy_clamp!

# Model type definitions

"""
    SensibleEnthalpy <: AbstractEnergyModel

Type that represents energy model, coefficients and respective fields.

### Fields
- `he`: Sensible enthalpy ScalarField.
- `T`: Temperature ScalarField.
- `hef`: Sensible enthalpy FaceScalarField.
- `Tf`: Temperature FaceScalarField.
- `K`: Specific kinetic energy ScalarField.
- `S_he`: Energy source term ScalarField (dp/dt for enthalpy).
- `coeffs`: A tuple of model coefficients.

"""
struct SensibleEnthalpy{S,FS,V,C} <: AbstractEnergyModel
    he::S
    T::S
    mueff_cell::S
    hef::FS
    Tf::FS
    K::S
    Kf::FS
    prevP::V 
    prevRhoK::V
    S_he::S
    coeffs::C
end
Adapt.@adapt_structure SensibleEnthalpy

"""
    InternalEnergy <: AbstractEnergyModel

Type that represents the internal energy transport model, coefficients and respective fields.
The solved variable is `e = cv*(T - Tref)` where `cv = cp/gamma`.

### Fields
- `he`: Internal energy ScalarField (e = cv*(T - Tref)).
- `T`: Temperature ScalarField.
- `hef`: Internal energy FaceScalarField.
- `Tf`: Temperature FaceScalarField.
- `K`: Specific kinetic energy ScalarField.
- `S_he`: Energy source term ScalarField (-p*div(U) for internal energy).
- `coeffs`: A tuple of model coefficients.

"""
struct InternalEnergy{S,FS,V,C} <: AbstractEnergyModel
    he::S
    T::S
    mueff_cell::S
    hef::FS
    Tf::FS
    K::S
    Kf::FS
    prevP::V 
    prevRhoK::V
    S_he::S
    coeffs::C
end
Adapt.@adapt_structure InternalEnergy

# Unified equation model container
struct EnergyEquationModel{E1,State}
    energy_eqn::E1
    state::State
end
Adapt.@adapt_structure EnergyEquationModel

# API constructors
Energy{SensibleEnthalpy}(; Tref) = begin
    coeffs = (Tref=Tref, other=nothing)
    ARG = typeof(coeffs)
    Energy{SensibleEnthalpy,ARG}(coeffs)
end

Energy{InternalEnergy}(; Tref) = begin
    coeffs = (Tref=Tref, other=nothing)
    ARG = typeof(coeffs)
    Energy{InternalEnergy,ARG}(coeffs)
end

# Functor constructors
(energy::Energy{EnergyModel, ARG})(mesh, fluid) where {EnergyModel,ARG} = begin
    backend = _get_backend(mesh)
    float_type = _get_float(mesh)
    n_cells = length(mesh.cells)
    he = ScalarField(mesh)
    T = ScalarField(mesh)
    mueff_cell = ScalarField(mesh)
    hef = FaceScalarField(mesh)
    Tf = FaceScalarField(mesh)
    K = ScalarField(mesh)
    Kf = FaceScalarField(mesh)
    prevP = KernelAbstractions.zeros(backend, float_type, n_cells) 
    prevRhoK = KernelAbstractions.zeros(backend, float_type, n_cells)
    S_he = ScalarField(mesh)
    coeffs = energy.args
    EnergyModel(he, T, mueff_cell, hef, Tf, K, Kf, prevP, prevRhoK, S_he, coeffs)
end

# Unified initialise
function initialise(
    energy::Union{SensibleEnthalpy, InternalEnergy},
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, mdotf, rho, peqn, config
    ) where {T1,F,SO,M,Tu,E,D,BI}

    (; he, T, S_he) = energy
    (; solvers, schemes, runtime, boundaries) = config
    mesh = model.domain
    eqn = peqn.equation

    keff = FaceScalarField(mesh)
    divK = ScalarField(mesh)
    dKdt = ScalarField(mesh)
    Phi = ScalarField(mesh)  # viscous heating: ∇·(τeff·U) for h, τeff:∇U for e

    # keep he when it already yields T (a continued run); rebuild it from a changed T
    energy_to_temperature!(model, he, divK) # divK as scratch, set in energy! before use
    divK.values == T.values || temperature_to_energy!(model, T, he)

    energy_eqn = (
        Time{schemes.he.time}(rho, he)
        + Divergence{schemes.he.divergence}(mdotf, he)
        - Laplacian{schemes.he.laplacian}(keff, he)
        ==
        Source(S_he) - Source(divK) - Source(dKdt) + Source(Phi)
    ) → eqn

    @reset energy_eqn.preconditioner = set_preconditioner(solvers.he, energy_eqn)
    @reset energy_eqn.solver = _workspace(solvers.he, energy_eqn)

    init_residual = (:he, 1.0)
    state = ModelState(init_residual, false)
    return EnergyEquationModel(energy_eqn, state)
end

"""
    energy!(energy::EnergyEquationModel, model::Physics{T1,F,SO,M,Tu,E,D,BI}, prevP, prevRhoK, mdotf, gradP, gradU, rho, mueff, time, config
    ) where {T1,F,SO,M,Tu,E,D,BI}

Run energy transport equations (sensible enthalpy or internal energy).

# Input
- `energy`: EnergyEquationModel.
- `model`: Physics model defined by user.
- `prevP`: Previous pressure cell values.
- `prevRhoK`: Previous rho*K values.
- `mdotf`: Face mass flow.
- `gradP`: Pressure gradient.
- `gradU`: Velocity gradient.
- `rho`: Density ScalarField.
- `mueff`: Effective viscosity FaceScalarField.
- `time`: Simulation runtime.
- `config`: Configuration structure defined by user with solvers, schemes, runtime and hardware structures set.

"""
function energy!(
    energy::EnergyEquationModel, model::Physics{T1,F,SO,M,Tu,E,D,BI}, mdotf, gradP, gradU, mueff, time, dt, config
    ) where {T1,F,SO,M,Tu,E,D,BI}

    mesh = model.domain

    (; rho, nu, Pr) = model.fluid
    (; U, p, Uf) = model.momentum
    (; he, hef, T, K, Kf, prevP, prevRhoK, S_he, mueff_cell) = model.energy
    (; energy_eqn, state) = energy
    (; solvers, schemes, runtime, hardware, boundaries) = config
    (; backend) = hardware

    keff = get_flux(energy_eqn, 3)

    S_he = get_source(energy_eqn, 1)
    divK = get_source(energy_eqn, 2)
    dKdt = get_source(energy_eqn, 3)
    Phi = get_source(energy_eqn, 4)

    # Set diffusion coefficient based on energy model type
    if model.energy isa SensibleEnthalpy
        @. keff.values = mueff.values / Pr.values
    else  # InternalEnergy: keff/cv = mueff * gamma / Pr
        gamma = model.fluid.gamma
        @. keff.values = mueff.values * gamma.values / Pr.values
    end

    # Kinetic-energy transport (divK, dKdt) and pressure work depend on the formulation.
    # Enthalpy: solves h_s, needs +div(rho U K)+d(rho K)/dt and dp/dt source.
    # Internal energy: solves e, has NO kinetic-energy terms; pressure work is -p*div(U).
    if model.energy isa SensibleEnthalpy
        # the viscous work multiplies the stress by the current U: take both from the same field
        grad!(gradU, Uf, U, boundaries.U, time, config)
        limit_gradient!(schemes.U.limiter, gradU, U, config)
        interpolate!(Uf, U, config)
        correct_boundaries!(Uf, U, boundaries.U, time, config)
        @. K.values = 0.5*(U.x.values^2 + U.y.values^2 + U.z.values^2)
        # K takes the face interpolation of the he convection term (term 2), so h + K is convected
        # by one discrete operator; boundary faces keep K of the boundary velocity
        @. Kf.values = 0.5*(Uf.x.values^2 + Uf.y.values^2 + Uf.z.values^2)
        interpolate!(Kf, K, mdotf, energy_eqn.model.terms[2].type, config)
        @. Kf.values *= mdotf.values
        div!(divK, Kf, config)
        @. dKdt.values = (rho.values*K.values - prevRhoK)/dt
        @. S_he.values = (p.values - prevP)/dt
    else
        @. divK.values = 0
        @. dKdt.values = 0
        _compute_pdivU!(S_he, p, gradU, config)
        @. S_he.values = -S_he.values
    end

    # h + K form: viscous work ∇·(τ·U); the e form takes the dissipation τ:∇U
    if model.energy isa SensibleEnthalpy
        viscous_work!(Phi, mueff, mueff_cell, U, Uf, gradU, mdotf, boundaries.U, config)
    else
        viscous_dissipation!(Phi, mueff_cell.values, gradU, boundaries.U, config)
    end

    # Set up and solve energy equation
    discretise!(energy_eqn, he, config)
    apply_boundary_conditions!(energy_eqn, boundaries.he, nothing, time, config)
    implicit_relaxation_diagdom!(energy_eqn, he.values, solvers.he.relax, nothing, config)
    update_preconditioner!(energy_eqn.preconditioner, mesh, config)
    he_res = solve_system!(energy_eqn, solvers.he, he, nothing, config)

    if !isnothing(solvers.he.limit)
        Tmin = solvers.he.limit[1]; Tmax = solvers.he.limit[2]
        energy_clamp!(model, he, Tmin, Tmax)
    end

    energy_to_temperature!(model, he, T)

    residuals = (:he, he_res)
    converged = he_res <= solvers.he.convergence
    state.residuals = residuals
    state.converged = converged

    return nothing
end

# Thermodynamic dispatch functions — SensibleEnthalpy

function thermo_Psi!(
    model::Physics{T,F,SO,M,Tu,E,D,BI}, Psi::ScalarField
    ) where {T,F<:AbstractCompressible,SO,M,Tu,E<:SensibleEnthalpy,D,BI}
    (; coeffs, he) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp; R = model.fluid.R
    @. Psi.values = Cp.values/(R.values*(he.values + Cp.values*Tref))
end

function thermo_Psi!(
    model::Physics{T,F,SO,M,Tu,E,D,BI}, Psif::FaceScalarField, time, config
    ) where {T,F<:AbstractCompressible,SO,M,Tu,E<:SensibleEnthalpy,D,BI}
    (; coeffs, hef, he) = model.energy
    interpolate!(hef, he, config)
    correct_boundaries!(hef, he, config.boundaries.he, time, config)
    (; Tref) = coeffs
    Cp = model.fluid.cp; R = model.fluid.R
    @. Psif.values = Cp.values/(R.values*(hef.values + Cp.values*Tref))
end

function temperature_to_energy!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, T::ScalarField, he::ScalarField
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E<:SensibleEnthalpy,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    @. he.values = Cp.values*(T.values-Tref)
end


function energy_to_temperature!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, he::ScalarField, T::ScalarField
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E<:SensibleEnthalpy,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    @. T.values = (he.values/Cp.values) + Tref
end

function energy_clamp!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, he::ScalarField, Tmin, Tmax
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E<:SensibleEnthalpy,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    hmin = Cp.values*(Tmin-Tref)
    hmax = Cp.values*(Tmax-Tref)
    clamp!(he.values, hmin, hmax)
end

function energy_clamp!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, hef::FaceScalarField, Tmin, Tmax
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E<:SensibleEnthalpy,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    hmin = Cp.values*(Tmin-Tref)
    hmax = Cp.values*(Tmax-Tref)
    clamp!(hef.values, hmin, hmax)
end


# Thermodynamic dispatch functions — InternalEnergy

function thermo_Psi!(
    model::Physics{T,F,SO,M,Tu,E,D,BI}, Psi::ScalarField
    ) where {T,F<:AbstractCompressible,SO,M,Tu,E<:InternalEnergy,D,BI}
    (; coeffs, he) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp; R = model.fluid.R; gamma = model.fluid.gamma
    @. Psi.values = (Cp.values/gamma.values) / (R.values*(he.values + (Cp.values/gamma.values)*Tref))
end


function thermo_Psi!(
    model::Physics{T,F,SO,M,Tu,E,D,BI}, Psif::FaceScalarField, time, config
    ) where {T,F<:AbstractCompressible,SO,M,Tu,E<:InternalEnergy,D,BI}
    (; coeffs, hef, he) = model.energy
    interpolate!(hef, he, config)
    correct_boundaries!(hef, he, config.boundaries.he, time, config)
    (; Tref) = coeffs
    Cp = model.fluid.cp; R = model.fluid.R; gamma = model.fluid.gamma
    @. Psif.values = (Cp.values/gamma.values) / (R.values*(hef.values + (Cp.values/gamma.values)*Tref))
end

function temperature_to_energy!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, T::ScalarField, he::ScalarField
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E<:InternalEnergy,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp; gamma = model.fluid.gamma
    @. he.values = (Cp.values/gamma.values)*(T.values - Tref)
end

function energy_to_temperature!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, he::ScalarField, T::ScalarField
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E<:InternalEnergy,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp; gamma = model.fluid.gamma
    @. T.values = he.values * gamma.values / Cp.values + Tref
end

function energy_clamp!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, he::ScalarField, Tmin, Tmax
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E<:InternalEnergy,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp; gamma = model.fluid.gamma
    cv = Cp.values/gamma.values
    emin = cv*(Tmin - Tref)
    emax = cv*(Tmax - Tref)
    clamp!(he.values, emin, emax)
end


function viscous_dissipation!(Phi::ScalarField, mueff_cell, gradU, U_BCs, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = Phi.mesh
    n_cells = length(mesh.cells)

    kernel! = _sized(_viscous_dissipation!, backend, workgroup, n_cells)
    kernel!(Phi.values, mueff_cell, gradU.result)
    KernelAbstractions.synchronize(backend)

    for BC ∈ U_BCs
        zero_viscous_dissipation!(BC, Phi, mesh, backend, workgroup)
    end
end

zero_viscous_dissipation!(BC, Phi, mesh, backend, workgroup) = nothing

function zero_viscous_dissipation!(
    BC::Union{Slip,Symmetry}, Phi, mesh, backend, workgroup)
    (; IDs_range) = BC
    ndrange = length(IDs_range)
    ndrange == 0 && return nothing
    kernel! = _zero_viscous_dissipation!(backend)
    kernel!(Phi.values, mesh.boundary_cellsID, IDs_range;
        _dynamic_setup(backend, workgroup, ndrange)...)
    KernelAbstractions.synchronize(backend)
end

@kernel function _zero_viscous_dissipation!(Phi, boundary_cellsID, IDs_range)
    i = @index(Global)
    fID = IDs_range[i]
    cID = boundary_cellsID[fID]
    Phi[cID] = 0
end

@kernel function _viscous_dissipation!(Phi, mueff, gradU)
    i = @index(Global)

    G = gradU[i]

    divU = G[1,1] + G[2,2] + G[3,3]
    diag_terms_sq = G[1,1]^2 + G[2,2]^2 + G[3,3]^2
    cross_terms_sq = (G[1,2] + G[2,1])^2 +
                     (G[1,3] + G[3,1])^2 +
                     (G[2,3] + G[3,2])^2

    Phi[i] = mueff[i] * (2.0 * diag_terms_sq + cross_terms_sq - (2.0/3.0) * divU^2)
end


# Viscous work ∇·(τ·U) as Σ_f U_f·F_f, with F_f the momentum equation's own viscous face force:
# Laplacian μ_f gDiff (U_N - U_P) plus the transpose-stress flux. U_f is upwinded by the mass flux;
# boundary faces use the boundary-condition gradient, and slip, symmetry and empty faces do no work.
function viscous_work!(Phi::ScalarField, mueff, mueff_cell, U, Uf, gradU, mdotf, U_BCs, config)
    (; backend, workgroup) = config.hardware
    mesh = Phi.mesh
    (; cells, cell_faces, cell_nsign, faces, face_gDiff) = mesh
    Phiv, Uv, Ufv, G = Phi.values, field_values(U), _face_vector_values(Uf), field_values(gradU.result)
    kernel! = _sized(_viscous_work_cells!, backend, workgroup, length(cells))
    kernel!(Phiv, mueff.values, mueff_cell.values, G, Uv, mdotf.values, face_gDiff,
        cells, cell_faces, cell_nsign, faces)
    KernelAbstractions.synchronize(backend)
    for BC ∈ U_BCs
        _viscous_work_boundary!(Phiv, mueff.values, G, Uv, Ufv, BC, cells, faces, backend, workgroup)
    end
    nothing
end

_face_vector_values(Uf::FaceVectorField) = VectorValues(Uf.x.values, Uf.y.values, Uf.z.values)

# stress per unit viscosity, τ/μ_eff, and its transpose part dev2((∇U)ᵀ), from G[i,j] = ∂U_i/∂x_j
@inline _stress(G, ::Type{TF}) where TF = G + G' - (TF(2)/3)*tr(G)*I
@inline _dev2_transpose(G, ::Type{TF}) where TF = G' - (TF(2)/3)*tr(G)*I

@kernel inbounds=true function _viscous_work_cells!(
    Phi, mueff, mueffc, gradU, U, mdotf, gDiff, cells::AbstractArray{Cell{TF,SV,UR}}, cell_faces, cell_nsign, faces
    ) where {TF,SV,UR}
    i = @index(Global)
    volume, faces_range = cells.volume[i], cells.faces_range[i]
    Ti = _dev2_transpose(gradU[i], TF)
    μi = _cell_coefficient(mueffc, i, TF)
    Ui = U[i]
    work = zero(TF)
    for fi ∈ faces_range
        fID = cell_faces[fi]
        nsign = cell_nsign[fi]
        ownerCells, area, normal, weight = faces.ownerCells[fID], faces.area[fID], faces.normal[fID], faces.weight[fID]
        owner = ownerCells[1] == i
        wi = owner ? weight : one(TF) - weight
        nID = owner ? ownerCells[2] : ownerCells[1]
        Un = U[nID]
        Tt = _face_flux_tensor(nothing, mueffc, fID, wi, μi, Ti, nID, _dev2_transpose(gradU[nID], TF))
        force = mueff[fID]*gDiff[fID]*(Un - Ui) + (Tt*normal)*(area*nsign)
        Uup = mdotf[fID] >= zero(TF) ? U[ownerCells[1]] : U[ownerCells[2]]
        work += Uup⋅force
    end
    Phi[i] = work/volume
end

_viscous_work_boundary!(Phi, mueff, gradU, U, Uf, ::Union{Slip,Symmetry,Empty}, cells, faces, backend, workgroup) =
    nothing

function _viscous_work_boundary!(Phi, mueff, gradU, U, Uf, BC, cells, faces, backend, workgroup)
    (; IDs_range) = BC
    isempty(IDs_range) && return nothing
    kernel! = _sized(_viscous_work_boundary_kernel!, backend, workgroup, length(IDs_range))
    kernel!(Phi, mueff, gradU, U, Uf, BC, IDs_range, cells, faces)
    KernelAbstractions.synchronize(backend)
end

@kernel inbounds=true function _viscous_work_boundary_kernel!(Phi, mueff, gradU, U, Uf, BC, IDs_range, cells, faces)
    i = @index(Global)
    fID = IDs_range[i]
    cID = faces.ownerCells[fID][1]
    normal, area, delta = faces.normal[fID], faces.area[fID], faces.delta[fID]
    Mb = _patch_gradient(gradU[cID], _boundary_sngrad(BC, U[cID], delta, normal), normal)
    τb = mueff[fID]*_stress(Mb, typeof(area))
    Atomix.@atomic Phi[cID] += (Uf[fID]⋅(τb*normal))*(area/cells.volume[cID])
end

function _compute_pdivU!(S::ScalarField, p, gradU, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = S.mesh
    n_cells = length(mesh.cells)
    kernel! = _sized(_pdivU_kernel!, backend, workgroup, n_cells)
    kernel!(S.values, p.values, gradU.result)
end

@kernel function _pdivU_kernel!(S, p, gradU)
    i = @index(Global)
    G = gradU[i]
    divU = G[1,1] + G[2,2] + G[3,3]
    S[i] = p[i] * divU
end
