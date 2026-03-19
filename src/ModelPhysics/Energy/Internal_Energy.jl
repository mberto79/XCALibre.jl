export InternalEnergy

# Model type definition
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

struct InternalEnergyModel{E1,State}
    energy_eqn::E1
    state::State
end
Adapt.@adapt_structure InternalEnergyModel

# Model API constructor
Energy{InternalEnergy}(; Tref) = begin
    coeffs = (Tref=Tref, other=nothing)
    ARG = typeof(coeffs)
    Energy{InternalEnergy,ARG}(coeffs)
end

# Functor as constructor
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

"""
    initialise(energy::InternalEnergy, model::Physics{T1,F,SO,M,Tu,E,D,BI}, mdotf, rho, peqn, config
    ) where {T1,F,SO,M,Tu,E,D,BI}

Initialisation of internal energy transport equations.

# Input
- `energy`: InternalEnergy model.
- `model`: Physics model defined by user.
- `mdtof`: Face mass flow.
- `rho`: Density ScalarField.
- `peqn`: Pressure equation.
- `config`: Configuration structure defined by user with solvers, schemes, runtime and
              hardware structures set.

# Output
- `InternalEnergyModel`: Energy model struct containing energy equation.

"""
function initialise(
    energy::InternalEnergy, model::Physics{T1,F,SO,M,Tu,E,D,BI}, mdotf, rho, peqn, config
    ) where {T1,F,SO,M,Tu,E,D,BI}

    (; he, T, S_he) = energy
    (; solvers, schemes, runtime, boundaries) = config
    mesh = mdotf.mesh
    eqn = peqn.equation

    # keff/cv = mueff * gamma / Pr  (since cv = cp/gamma and keff = mueff*cp/Pr)
    keff_by_cv = FaceScalarField(mesh)
    divK = ScalarField(mesh)
    dKdt = ScalarField(mesh)

    temperature_to_energy!(model, T, he)

    energy_eqn = (
        Time{schemes.he.time}(rho, he)
        + Divergence{schemes.he.divergence}(mdotf, he)
        - Laplacian{schemes.he.laplacian}(keff_by_cv, he)
        ==
        Source(S_he) - Source(divK) - Source(dKdt)
    ) → eqn

    # Set up preconditioners
    @reset energy_eqn.preconditioner = set_preconditioner(solvers.he.preconditioner, energy_eqn)

    # preallocating solvers
    @reset energy_eqn.solver = _workspace(solvers.he.solver, _b(energy_eqn))

    init_residual = (:he, 1.0)
    init_converged = false
    state = ModelState(init_residual, init_converged)

    return InternalEnergyModel(energy_eqn, state)
end


"""
    energy!(energy::InternalEnergyModel, model::Physics{T1,F,SO,M,Tu,E,D,BI}, prevP, prevRhoK, mdotf, gradP, gradU, rho, mueff, time, config
    ) where {T1,F,SO,M,Tu,E,D,BI}

Run internal energy transport equations.

# Input
- `energy`: InternalEnergyModel.
- `model`: Physics model defined by user.
- `prevP`: Previous pressure cell values.
- `prevRhoK`: Previous rho*K values.
- `mdtof`: Face mass flow.
- `gradP`: Pressure gradient.
- `gradU`: Velocity gradient.
- `rho`: Density ScalarField.
- `mueff`: Effective viscosity FaceScalarField.
- `time`: Simulation runtime.
- `config`: Configuration structure defined by user with solvers, schemes, runtime and hardware structures set.

"""
function energy!(
    energy::InternalEnergyModel, model::Physics{T1,F,SO,M,Tu,E,D,BI}, prevP, prevRhoK, mdotf, gradP, gradU, rho, mueff, time, config
    ) where {T1,F,SO,M,Tu,E,D,BI}

    mesh = model.domain

    (; rho, nu) = model.fluid
    (; U, p) = model.momentum
    (; he, hef, T, K) = model.energy
    (; energy_eqn, state) = energy
    (; solvers, runtime, hardware, boundaries) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware

    # keff/cv = mueff * gamma / Pr  (since cv = cp/gamma => keff/cv = mueff*cp/(Pr*cv) = mueff*gamma/Pr)
    keff_by_cv = get_flux(energy_eqn, 3)

    S_he = get_source(energy_eqn, 1)
    divK = get_source(energy_eqn, 2)
    dKdt = get_source(energy_eqn, 3)

    Uf = FaceVectorField(mesh)
    Kf = FaceScalarField(mesh)
    Pr = model.fluid.Pr
    gamma = model.fluid.gamma

    dt = runtime.dt[1]

    TF = _get_float(mesh)
    n_cells = length(mesh.cells)

    # keff/cv = mueff * gamma / Pr
    @. keff_by_cv.values = mueff.values * gamma.values / Pr.values

    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, boundaries.U, time, config)

    @. K.values = 0.5*(U.x.values^2 + U.y.values^2 + U.z.values^2)
    @. Kf.values = 0.5*(Uf.x.values^2 + Uf.y.values^2 + Uf.z.values^2)

    interpolate_upwind!(Kf, K, mdotf, config) # only internal faces

    @. Kf.values *= mdotf.values
    div!(divK, Kf, config)

    @. dKdt.values = (rho.values*K.values - prevRhoK)/dt

    # Source term for internal energy: -p*div(U) = -p*tr(gradU)
    _compute_pdivU!(S_he, p, gradU, config)
    @. S_he.values = -S_he.values

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

# Compute p*div(U) = p*tr(gradU) into a ScalarField
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
    # cv = Cp/gamma; Psi = cv/(R*(e + cv*Tref))
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

"""
    temperature_to_energy!(model::Physics{T1,F,SO,M,Tu,E,D,BI}, T::ScalarField, he::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E<:InternalEnergy,D,BI}

Convert temperature to internal energy: e = cv*(T - Tref) where cv = cp/gamma.
"""
function temperature_to_energy!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, T::ScalarField, he::ScalarField
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E<:InternalEnergy,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp; gamma = model.fluid.gamma
    @. he.values = (Cp.values/gamma.values)*(T.values - Tref)
end

"""
    energy_to_temperature!(model::Physics{T1,F,SO,M,Tu,E,D,BI}, he::ScalarField, T::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E<:InternalEnergy,D,BI}

Convert internal energy to temperature: T = e/cv + Tref where cv = cp/gamma.
"""
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
