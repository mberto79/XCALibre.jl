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
struct SensibleEnthalpy{S1,S2,F1,F2,S3,S4,C} <: AbstractEnergyModel
    he::S1
    T::S2
    hef::F1
    Tf::F2
    K::S3
    S_he::S4
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
struct InternalEnergy{S1,S2,F1,F2,S3,S4,C} <: AbstractEnergyModel
    he::S1
    T::S2
    hef::F1
    Tf::F2
    K::S3
    S_he::S4
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
(energy::Energy{EnergyModel, ARG})(mesh, fluid) where {EnergyModel<:SensibleEnthalpy,ARG} = begin
    he = ScalarField(mesh)
    T = ScalarField(mesh)
    hef = FaceScalarField(mesh)
    Tf = FaceScalarField(mesh)
    K = ScalarField(mesh)
    S_he = ScalarField(mesh)
    coeffs = energy.args
    SensibleEnthalpy(he, T, hef, Tf, K, S_he, coeffs)
end

(energy::Energy{EnergyModel, ARG})(mesh, fluid) where {EnergyModel<:InternalEnergy,ARG} = begin
    he = ScalarField(mesh)
    T = ScalarField(mesh)
    hef = FaceScalarField(mesh)
    Tf = FaceScalarField(mesh)
    K = ScalarField(mesh)
    S_he = ScalarField(mesh)
    coeffs = energy.args
    InternalEnergy(he, T, hef, Tf, K, S_he, coeffs)
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

    temperature_to_energy!(model, T, he)

    energy_eqn = (
        Time{schemes.he.time}(rho, he)
        + Divergence{schemes.he.divergence}(mdotf, he)
        - Laplacian{schemes.he.laplacian}(keff, he)
        ==
        Source(S_he) - Source(divK) - Source(dKdt)
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
    energy::EnergyEquationModel, model::Physics{T1,F,SO,M,Tu,E,D,BI}, prevP, prevRhoK, mdotf, gradP, gradU, rho, mueff, time, config
    ) where {T1,F,SO,M,Tu,E,D,BI}

    mesh = model.domain

    (; rho, nu) = model.fluid
    (; U, p) = model.momentum
    (; he, hef, T, K, S_he) = model.energy
    (; energy_eqn, state) = energy
    (; solvers, runtime, hardware, boundaries) = config
    (; backend) = hardware

    keff = get_flux(energy_eqn, 3)

    S_he = get_source(energy_eqn, 1)
    divK = get_source(energy_eqn, 2)
    dKdt = get_source(energy_eqn, 3)

    Uf = FaceVectorField(mesh)
    Kf = FaceScalarField(mesh)
    Pr = model.fluid.Pr

    dt = runtime.dt[1]

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

    interpolate_upwind!(Kf, K, mdotf, config) # only do internal faces

    @. Kf.values *= mdotf.values
    div!(divK, Kf, config)

    @. dKdt.values = (rho.values*K.values - prevRhoK)/dt

    # Set source term based on energy model type
    if model.energy isa SensibleEnthalpy
        @. S_he.values = (p.values - prevP)/dt
    else  # InternalEnergy: -p*div(U)
        _compute_pdivU!(S_he, p, gradU, config)
        @. S_he.values = -S_he.values
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
    interpolate!(hef, he, config)
    correct_boundaries!(hef, he, boundaries.he, time, config)

    @. prevRhoK = rho.values*0.5*(U.x.values^2 + U.y.values^2 + U.z.values^2)
    @. prevP = p.values

    residuals = (:he, he_res)
    converged = he_res <= solvers.he.convergence
    state.residuals = residuals
    state.converged = converged

    return nothing
end


# Thermodynamic dispatch functions — SensibleEnthalpy

"""
    thermo_Psi!(model::Physics{T,F,SO,M,Tu,E,D,BI}, Psi::ScalarField)
    where {T,F<:AbstractCompressible,M,Tu,E<:SensibleEnthalpy,D,BI}

Model updates the value of Psi.

### Input
- `model`  -- Physics model defined by user.
- `Psi`    -- Compressibility factor ScalarField.

### Algorithm
Weakly compressible currently uses the ideal gas equation for establishing the
compressibility factor where ``\\rho = p * \\Psi``. ``\\Psi`` is calculated from the sensible enthalpy, reference temperature and fluid model specified ``C_p`` and ``R`` value where ``R`` is calculated from ``C_p`` and ``\\gamma`` specified in the fluid model.
"""
function thermo_Psi!(
    model::Physics{T,F,SO,M,Tu,E,D,BI}, Psi::ScalarField
    ) where {T,F<:AbstractCompressible,SO,M,Tu,E<:SensibleEnthalpy,D,BI}
    (; coeffs, he) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp; R = model.fluid.R
    @. Psi.values = Cp.values/(R.values*(he.values + Cp.values*Tref))
end

"""
    thermo_Psi!(model::Physics{T,F,SO,M,Tu,E,D,BI}, Psif::FaceScalarField, config)
    where {T,F<:AbstractCompressible,M,Tu,E<:SensibleEnthalpy,D,BI}

Function updates the value of Psi.

### Input
- `model`  -- Physics model defined by user.
- `Psif`    -- Compressibility factor FaceScalarField.

### Algorithm
Weakly compressible currently uses the ideal gas equation for establishing the
compressibility factor where ``\\rho = p * \\Psi``. ``\\Psi`` is calculated from the sensible
enthalpy, reference temperature and fluid model specified ``C_p`` and ``R`` value where
``R`` is calculated from ``C_p`` and ``\\gamma`` specified in the fluid model.
"""
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

"""
    temperature_to_energy!(model::Physics{T1,F,SO,M,Tu,E,D,BI}, T::ScalarField, he::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E<:SensibleEnthalpy,D,BI}

Function converts temperature ScalarField to sensible enthalpy ScalarField.

### Input
- `model`  -- Physics model defined by user.
- `T`      -- Temperature ScalarField.
- `he`     -- Sensible enthalpy ScalarField.
"""
function temperature_to_energy!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, T::ScalarField, he::ScalarField
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E<:SensibleEnthalpy,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    @. he.values = Cp.values*(T.values-Tref)
end

"""
    energy_to_temperature!(model::Physics{T1,F,SO,M,Tu,E,D,BI}, he::ScalarField, T::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E<:SensibleEnthalpy,D,BI}

Function converts sensible enthalpy ScalarField to temperature ScalarField.

### Input
- `model`  -- Physics model defined by user.
- `he`     -- Sensible enthalpy ScalarField.
- `T`      -- Temperature ScalarField.
"""
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

"""
    thermo_Psi!(model::Physics{T,F,SO,M,Tu,E,D,BI}, Psi::ScalarField)
    where {T,F<:AbstractCompressible,M,Tu,E<:InternalEnergy,D,BI}

Update the compressibility factor Psi for the internal energy model.

### Algorithm
For ideal gas: rho = p * Psi. With e = cv*(T - Tref), temperature T = e/cv + Tref.
Thus Psi = cv / (R*(e + cv*Tref)) = 1 / (R*(e/cv + Tref)) = 1 / (R*T).
"""
function thermo_Psi!(
    model::Physics{T,F,SO,M,Tu,E,D,BI}, Psi::ScalarField
    ) where {T,F<:AbstractCompressible,SO,M,Tu,E<:InternalEnergy,D,BI}
    (; coeffs, he) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp; R = model.fluid.R; gamma = model.fluid.gamma
    @. Psi.values = (Cp.values/gamma.values) / (R.values*(he.values + (Cp.values/gamma.values)*Tref))
end

"""
    thermo_Psi!(model::Physics{T,F,SO,M,Tu,E,D,BI}, Psif::FaceScalarField, config)
    where {T,F<:AbstractCompressible,M,Tu,E<:InternalEnergy,D,BI}

Update the face compressibility factor Psif for the internal energy model.
"""
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


# Utility functions

function viscous_dissipation!(Phi::ScalarField, nu, rho, gradU, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = Phi.mesh
    n_cells = length(mesh.cells)

    kernel! = _viscous_dissipation_kernel!(_setup(backend, workgroup, n_cells)...)
    kernel!(Phi.values, nu, rho, gradU.result)
end

@kernel function _viscous_dissipation_kernel!(Phi, nu, rho, gradU)
    i = @index(Global)

    G = gradU[i]

    divU = G[1,1] + G[2,2] + G[3,3]
    diag_terms_sq = G[1,1]^2 + G[2,2]^2 + G[3,3]^2
    cross_terms_sq = (G[1,2] + G[2,1])^2 +
                     (G[1,3] + G[3,1])^2 +
                     (G[2,3] + G[3,2])^2

    Phi[i] = nu[i]*rho[i] * (2.0 * diag_terms_sq + cross_terms_sq - (2.0/3.0) * divU^2)
end

function correct_face_interpolation!(phif::FaceScalarField, phi, Uf::FaceVectorField)
    mesh = phif.mesh
    (; faces, cells) = mesh
    for fID ∈ eachindex(faces)
        face = faces[fID]
        (; ownerCells, area, normal) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        phi1 = phi[cID1]
        phi2 = phi[cID2]
        flux = Uf[fID]
        if flux >= 0.0
            phif.values[fID] = phi1
        else
            phif.values[fID] = phi2
        end
    end
end

function interpolate_upwind!(phif::FaceScalarField, phi::ScalarField, mdotf::FaceScalarField, config)
    vals = phi.values
    fvals = phif.values
    flux = mdotf.values

    mesh = phif.mesh
    (; cells, faces) = mesh

    nbfaces = length(mesh.boundary_cellsID)
    internal_faces_count = length(faces) - nbfaces

    (; hardware) = config
    (; backend, workgroup) = hardware

    kernel! = interpolate_upwind_Scalar!(_setup(backend, workgroup, internal_faces_count)...)
    kernel!(fvals, vals, flux, cells, faces, nbfaces)
end

@kernel function interpolate_upwind_Scalar!(fvals, vals, flux, cells, faces, nbfaces)
    t = @index(Global)
    i = t + nbfaces

    @inbounds begin
        face = faces[i]
        (; ownerCells) = face

        owner1 = ownerCells[1]
        owner2 = ownerCells[2]

        phi1 = vals[owner1]
        phi2 = vals[owner2]

        if flux[i] >= 0.0
            fvals[i] = phi1
        else
            fvals[i] = phi2
        end
    end
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
