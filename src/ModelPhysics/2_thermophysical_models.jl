export HelmholtzEnergy, HelmholtzEnergyFluid, H2, H2_para, N2
export ConstEos, ConstMu, ConstK, ConstCp, ConstBeta
export IdealGas
export phase_compressibility, phase_betaT, _phase_beta_value, _phase_beta_field, R_UNIVERSAL


abstract type HelmholtzEnergyFluid end

struct N2 <: HelmholtzEnergyFluid end
struct H2 <: HelmholtzEnergyFluid end
struct H2_para <: HelmholtzEnergyFluid end

Base.@kwdef struct HelmholtzEnergy{F<:HelmholtzEnergyFluid}
    name::F
end

Base.@kwdef struct HydrogenViscosity <: AbstractPhysicsProperty end
Base.@kwdef struct NitrogenViscosity <: AbstractPhysicsProperty end


"""
    ConstEos <: AbstractEosModel

Constant density equation of state model.

### Fields
- 'rho' -- Constant density value.
"""
Base.@kwdef struct ConstEos{T<:AbstractFloat} <: AbstractEosModel
    rho::T
end
(eos::ConstEos)(phase, model, config) = begin
    rho_field = phase.rho
    initialise!(rho_field, eos.rho)
end


"""
    ConstMu <: AbstractViscosityModel

Constant dynamic viscosity model.

### Fields
- 'mu' -- Dynamic viscosity value [Pa⋅s].
"""
Base.@kwdef struct ConstMu{T<:AbstractFloat} <: AbstractViscosityModel
    mu::T
end
(mu::ConstMu)(phase, model) = begin
    mu_field = phase.mu
    mu_val = phase.mu[1]
    rho_val = phase.rho[1]

    initialise!(mu_field, mu_val/rho_val)
end


"""
    ConstK <: AbstractConductivityModel

Constant thermal conductivity model.

### Fields
- `k` -- Constant thermal conductivity value [W/m/K].
"""
Base.@kwdef struct ConstK{T<:AbstractFloat} <: AbstractConductivityModel
    k::T
end

"""
    ConstCp <: AbstractHeatCapacityModel

Constant specific heat capacity model.

### Fields
- `cp` -- Constant specific heat capacity at constant pressure [J/kg/K].
"""
Base.@kwdef struct ConstCp{T<:AbstractFloat} <: AbstractHeatCapacityModel
    cp::T
end

"""
    ConstBeta <: AbstractExpansivityModel

Constant thermal expansivity model.

### Fields
- `beta` -- Constant coefficient of thermal expansion [1/K].
"""
Base.@kwdef struct ConstBeta{T<:AbstractFloat} <: AbstractExpansivityModel
    beta::T
end


"""
    R_UNIVERSAL

Universal gas constant, 8.314462618 J/mol/K (CODATA).
"""
const R_UNIVERSAL = 8.314462618

"""
    IdealGas <: AbstractEosModel

Ideal gas equation of state, `rho = p/(R*T)`, with `p` the **absolute** pressure.

Construct with either the specific gas constant or the molar mass:

    IdealGas(R=4124.2)          # [J/kg/K]
    IdealGas(M=2.01588e-3)      # [kg/mol]  -> R = R_UNIVERSAL/M

Fernandes et al. (2026) Sec. 3.2 treat the hydrogen vapour in the K-Site and
MHTB tanks as an ideal gas, so this is the reference behaviour for the LH2 tank
cases (rather than the full Helmholtz EOS).

### Fields
- `R` -- Specific gas constant [J/kg/K].
"""
struct IdealGas{T<:AbstractFloat} <: AbstractEosModel
    R::T
end

function IdealGas(; R=nothing, M=nothing)
    if R !== nothing && M !== nothing
        throw(ArgumentError("Provide either `R` (specific gas constant) or `M` (molar mass), not both"))
    elseif R !== nothing
        return IdealGas(float(R))
    elseif M !== nothing
        return IdealGas(float(R_UNIVERSAL/M))
    else
        throw(ArgumentError("`IdealGas` needs either `R=` [J/kg/K] or `M=` [kg/mol]"))
    end
end

"""
    phase_compressibility(eos, p_abs, T)

Isothermal compressibility `(1/rho)*(d rho/d p)` at constant temperature, which
is the coefficient each phase contributes to the pressure equation's
compressibility term.

For an ideal gas this is exactly `1/p`, independent of `R` and `T`. For a
constant-density phase it is zero.
"""
phase_compressibility(::ConstEos, p_abs, T) = zero(p_abs)
phase_compressibility(::IdealGas, p_abs, T) = one(p_abs)/p_abs

"""
    phase_betaT(eos, beta, T)

The dimensionless group `beta*T` (thermal expansivity times temperature), which
weights a phase's contribution to the pressure-work source of the temperature
equation:

    rho*cp*DT/Dt = div(k grad T) + beta*T*Dp/Dt + Phi

For an ideal gas `beta = 1/T` exactly, so `beta*T = 1` and the *full* `Dp/Dt`
appears — this is the term that makes ullage self-pressurisation heat the gas.
For a constant-density phase `beta` is whatever expansivity was supplied
(`ConstBeta`), or zero if none was.
"""
phase_betaT(::IdealGas, beta, T) = one(T)
phase_betaT(::ConstEos, beta, T) = beta*T

"""
    _phase_beta_value(phase) -> Float64

The phase's expansivity as a plain number, zero when none was supplied. Resolving
`nothing` here keeps it out of kernels.

Only valid for a constant expansivity; use [`_phase_beta_field`](@ref) where the
value may vary per cell.
"""
_phase_beta_value(phase) = phase.beta === nothing ? 0.0 : phase.beta[1]

"""
    _phase_beta_field(phase)

The phase's expansivity as something a kernel can index per cell: the stored
field when there is one, and a `ConstantScalar(0)` when the phase has no
expansivity model at all.

Substituting a `ConstantScalar` for `nothing` is what lets the kernels index
uniformly - `beta[i]` is then correct whether the model is constant, tabulated,
or absent - without any of them having to test for `nothing`.
"""
_phase_beta_field(phase) =
    phase.beta === nothing ? ConstantScalar(0.0) : phase.beta

# Per-cell property update for a variable EOS. The generic no-op lives in
# 2_fluid_models.jl; this method must be here because `IdealGas` is defined in
# this file, which ModelPhysics.jl includes after 2_fluid_models.jl.
#
# `p_abs` must be the ABSOLUTE pressure field (gauge pressure + operating
# pressure), not the gauge or p_rgh field.
function update_phase_property!(field, model::IdealGas, p_abs, T, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(field)
    kernel! = _ideal_gas_density!(_setup(backend, workgroup, ndrange)...)
    kernel!(field, p_abs, T, model.R)
    return nothing
end

@kernel inbounds=true function _ideal_gas_density!(rho, p_abs, T, R)
    i = @index(Global)
    rho[i] = p_abs[i]/(R*T[i])
end

# Storage selection for phase properties, used by `build_phase`. Constant
# models collapse to a ConstantScalar (no per-cell storage), variable models
# get a ScalarField, and an unsupplied property stays `nothing`.
_phase_property_field(::Nothing, mesh) = nothing
_phase_property_field(model::ConstEos, mesh) = ConstantScalar(model.rho)
_phase_property_field(model::ConstMu, mesh) = ConstantScalar(model.mu)
_phase_property_field(model::ConstK, mesh) = ConstantScalar(model.k)
_phase_property_field(model::ConstCp, mesh) = ConstantScalar(model.cp)
_phase_property_field(model::ConstBeta, mesh) = ConstantScalar(model.beta)
_phase_property_field(model, mesh) = ScalarField(mesh)