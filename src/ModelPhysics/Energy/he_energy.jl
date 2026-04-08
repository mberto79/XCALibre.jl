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
    mesh = mdotf.mesh
    eqn = peqn.equation

    keff = FaceScalarField(mesh)
    divK = ScalarField(mesh)
    dKdt = ScalarField(mesh)
    Phi = ScalarField(mesh)  # viscous dissipation source τeff:∇U

    temperature_to_energy!(model, T, he)

    energy_eqn = (
        Time{schemes.he.time}(rho, he)
        + Divergence{schemes.he.divergence}(mdotf, he)
        - Laplacian{schemes.he.laplacian}(keff, he)
        ==
        Source(S_he) - Source(divK) - Source(dKdt) + Source(Phi)
    ) → eqn

    @reset energy_eqn.preconditioner = set_preconditioner(solvers.he.preconditioner, energy_eqn)
    @reset energy_eqn.solver = _workspace(solvers.he.solver, _b(energy_eqn))

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

    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, boundaries.U, time, config)
    
    @. K.values = 0.5*(U.x.values^2 + U.y.values^2 + U.z.values^2)
    @. Kf.values = 0.5*(Uf.x.values^2 + Uf.y.values^2 + Uf.z.values^2)

    # interpolate_upwind!(Kf, K, mdotf, config) # only do internal faces
    interpolate!(Kf, K, config) # only do internal faces
    @. Kf.values *= mdotf.values
    div!(divK, Kf, config)

    @. dKdt.values = (rho.values*K.values - prevRhoK)/dt


    # Set source term based on energy model type
    if model.energy isa SensibleEnthalpy
        @. S_he.values = (p.values - prevP)/dt
    else  
        _compute_pdivU!(S_he, p, gradU, config) # InternalEnergy: -p*div(U)
        @. S_he.values = -S_he.values
    end

    viscous_dissipation!(Phi, mueff_cell.values, gradU, config)

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
    model::Physics{T,F,SO,M,Tu,E,D,BI}, Psif::FaceScalarField, config
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
    model::Physics{T,F,SO,M,Tu,E,D,BI}, Psif::FaceScalarField, config
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


function viscous_dissipation!(Phi::ScalarField, mueff_cell, gradU, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = Phi.mesh
    n_cells = length(mesh.cells)

    kernel! = _viscous_dissipation!(_setup(backend, workgroup, n_cells)...)
    kernel!(Phi.values, mueff_cell, gradU.result)
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


function _compute_pdivU!(S::ScalarField, p, gradU, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = S.mesh
    n_cells = length(mesh.cells)
    kernel! = _pdivU_kernel!(_setup(backend, workgroup, n_cells)...)
    kernel!(S.values, p.values, gradU.result)
end

@kernel function _pdivU_kernel!(S, p, gradU)
    i = @index(Global)
    G = gradU[i]
    divU = G[1,1] + G[2,2] + G[3,3]
    S[i] = p[i] * divU
end

# function interpolate_upwind!(phif::FaceScalarField, phi::ScalarField, mdotf::FaceScalarField, config)
#     vals = phi.values
#     fvals = phif.values
#     flux = mdotf.values

#     mesh = phif.mesh
#     (; cells, faces) = mesh

#     nbfaces = length(mesh.boundary_cellsID)
#     internal_faces_count = length(faces) - nbfaces

#     (; hardware) = config
#     (; backend, workgroup) = hardware

#     kernel! = interpolate_upwind_Scalar!(_setup(backend, workgroup, internal_faces_count)...)
#     kernel!(fvals, vals, flux, cells, faces, nbfaces)
# end

# @kernel function interpolate_upwind_Scalar!(fvals, vals, flux, cells, faces, nbfaces)
#     t = @index(Global)
#     i = t + nbfaces

#     @inbounds begin
#         face = faces[i]
#         (; ownerCells) = face

#         owner1 = ownerCells[1]
#         owner2 = ownerCells[2]

#         phi1 = vals[owner1]
#         phi2 = vals[owner2]

#         if flux[i] >= 0.0
#             fvals[i] = phi1
#         else
#             fvals[i] = phi2
#         end
#     end
# end