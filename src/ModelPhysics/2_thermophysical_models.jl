export HelmholtzEnergy, HelmholtzEnergyFluid, H2, H2_para, N2
export ConstEos, ConstMu
export Viscosity, SutherlandViscosity, ConstantViscosity


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


struct Viscosity{T,ARG}
    args::ARG
end

"""
    ConstantViscosity <: AbstractViscosityModel

Constant kinematic viscosity model.

### Fields
- 'nu' -- Kinematic viscosity value [m^2/s].
"""
struct ConstantViscosity{S,FS} <: AbstractViscosityModel
    nu::S
    nuf::FS
end
Adapt.@adapt_structure ConstantViscosity

struct SutherlandViscosity{S,FS,C} <: AbstractViscosityModel
    nu::S
    nuf::FS
    coeffs::C
end 
Adapt.@adapt_structure SutherlandViscosity

Viscosity{ConstantViscosity}(; nu) = begin
    coeffs = (nu=nu, other=nothing)
    ARG = typeof(coeffs)
    Viscosity{ConstantViscosity,ARG}(coeffs)
end

Viscosity{SutherlandViscosity}(; mu_ref, T_ref, S) = begin
    coeffs = (mu_ref=mu_ref, T_ref=T_ref, S=S, other=nothing)
    ARG = typeof(coeffs)
    Viscosity{SutherlandViscosity,ARG}(coeffs)
end

(viscosity::Viscosity{ConstantViscosity, ARG})(mesh) where {ConstantViscosity,ARG} = begin
    backend = _get_backend(mesh)
    float_type = _get_float(mesh)
    n_cells = length(mesh.cells)
    nu = ConstantScalar(viscosity.args.nu)
    nuf = nu
    ConstantViscosity(nu, nuf)
end

(viscosity::Viscosity{SutherlandViscosity, ARG})(mesh) where {SutherlandViscosity,ARG} = begin
    backend = _get_backend(mesh)
    float_type = _get_float(mesh)
    n_cells = length(mesh.cells)
    nu = ScalarField(mesh)
    nuf = FaceScalarField(mesh)
    coeffs = viscosity.args
    SutherlandViscosity(nu, nuf, coeffs)
end


