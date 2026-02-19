export AbstractFluid, AbstractIncompressible, AbstractCompressible
export Fluid
export Incompressible, WeaklyCompressible, Compressible
export Phase, Fluid, Multiphase

abstract type AbstractFluid end
abstract type AbstractIncompressible <: AbstractFluid end
abstract type AbstractCompressible <: AbstractFluid end
abstract type AbstractMultiphase <: AbstractFluid end

Base.show(io::IO, fluid::AbstractFluid) = print(io, typeof(fluid).name.wrapper)


"""
    Fluid <: AbstractFluid

Abstract fluid model type for constructing new fluid models.

### Fields
- 'args' -- Model arguments.

"""
struct Fluid{T,ARG}
    args::ARG
end

"""
    Incompressible <: AbstractIncompressible

Incompressible fluid model containing fluid field parameters for incompressible flows.

### Fields
- 'nu'   -- Fluid kinematic viscosity.
- 'rho'  -- Fluid density.

### Examples
- `Fluid{Incompressible}(nu=0.001, rho=1.0)` - Constructor with default values.
"""
@kwdef struct Incompressible{S1, S2, F1, F2} <: AbstractIncompressible
    nu::S1
    rho::S2
    nuf::F1
    rhof::F2
end
Adapt.@adapt_structure Incompressible

Fluid{Incompressible}(; nu, rho=1.0) = begin
    coeffs = (nu=nu, rho=rho)
    ARG = typeof(coeffs)
    Fluid{Incompressible,ARG}(coeffs)
end

(fluid::Fluid{Incompressible, ARG})(mesh) where ARG = begin
    coeffs = fluid.args
    (; rho, nu) = coeffs
    nu = ConstantScalar(nu)
    nuf = nu
    rho = ConstantScalar(rho)
    rhof = rho
    Incompressible(nu, rho, nuf, rhof)
end

"""
    WeaklyCompressible <: AbstractCompressible

Weakly compressible fluid model containing fluid field parameters for weakly compressible 
    flows with constant parameters - ideal gas with constant viscosity.

### Fields
- 'nu'   -- Fluid kinematic viscosity.
- 'cp'   -- Fluid specific heat capacity.
- `gamma` -- Ratio of specific heats.
- `Pr`   -- Fluid Prandtl number.

### Examples
- `Fluid{WeaklyCompressible}(; nu=1E-5, cp=1005.0, gamma=1.4, Pr=0.7)` - Constructor with 
default values.
"""
struct WeaklyCompressible{S1, S2, F1, F2, T} <: AbstractCompressible
    nu::S1
    rho::S2
    nuf::F1
    rhof::F2
    cp::T
    gamma::T
    Pr::T
    R::T
end
Adapt.@adapt_structure WeaklyCompressible

Fluid{WeaklyCompressible}(; nu, cp, gamma, Pr) = begin
    coeffs = (nu=nu, cp=cp, gamma=gamma, Pr=Pr)
    ARG = typeof(coeffs)
    Fluid{WeaklyCompressible,ARG}(coeffs)
end

(fluid::Fluid{WeaklyCompressible, ARG})(mesh) where ARG = begin
    coeffs = fluid.args
    (; nu, cp, gamma, Pr) = coeffs
    cp = ConstantScalar(cp)
    gamma = ConstantScalar(gamma)
    Pr = ConstantScalar(Pr)
    R = ConstantScalar(cp.values*(1.0 - (1.0/gamma.values)))

    nu = ConstantScalar(nu)
    rho = ScalarField(mesh)
    nuf = nu
    rhof = FaceScalarField(mesh)
    WeaklyCompressible(nu, rho, nuf, rhof, cp, gamma, Pr, R)
end

"""
    Compressible <: AbstractCompressible

Compressible fluid model containing fluid field parameters for compressible flows with 
    constant parameters - ideal gas with constant viscosity.

### Fields
- 'nu'   -- Fluid kinematic viscosity.
- 'cp'   -- Fluid specific heat capacity.
- `gamma` -- Ratio of specific heats.
- `Pr`   -- Fluid Prantl number.

### Examples
- `Fluid{Compressible}(; nu=1E-5, cp=1005.0, gamma=1.4, Pr=0.7)` - Constructur with default values.
"""
@kwdef struct Compressible{S1, S2, F1, F2, T} <: AbstractCompressible
    nu::S1
    rho::S2
    nuf::F1
    rhof::F2
    cp::T
    gamma::T
    Pr::T
    R::T
end
Adapt.@adapt_structure Compressible

Fluid{Compressible}(; nu=1E-5, cp=1005.0, gamma=1.4, Pr=0.7 ) = begin
    coeffs = (nu=nu, cp=cp, gamma=gamma, Pr=Pr)
    ARG = typeof(coeffs)
    Fluid{Compressible,ARG}(coeffs)
end

(fluid::Fluid{Compressible, ARG})(mesh) where ARG = begin
    coeffs = fluid.args
    (; nu, cp, gamma, Pr) = coeffs
    cp = ConstantScalar(cp)
    gamma = ConstantScalar(gamma)
    Pr = ConstantScalar(Pr)
    R = ConstantScalar(cp.values*(1.0 - (1.0/gamma.values)))

    nu = ConstantScalar(nu)
    rho = ScalarField(mesh)
    nuf = nu
    rhof = FaceScalarField(mesh)
    Compressible(nu, rho, nuf, rhof, cp, gamma, Pr, R)
end






########


"""
    Phase <: AbstractPhase

Configuration structure for a single fluid phase.

### Fields
- 'eosModel'       -- Equation of State model for the phase.
- 'mu' -- Viscosity model for the phase.
"""
struct Phase{E<:AbstractEosModel, V<:AbstractViscosityModel} <: AbstractPhase
    density::E
    mu::V
end

function Phase(; density, mu) # Covers all combinations e.g. mu=1.8e-5 or mu=SutherlandModel() etc
    density_type = density isa AbstractFloat ? ConstEos(density) : density
    mu_type = mu isa AbstractFloat ? ConstMu(mu) : mu
    
    return Phase(density_type, mu_type)
end

@kwdef struct PhaseState{E<:AbstractEosModel, V<:AbstractViscosityModel, S1,S2,S3,S4,S5} <: AbstractPhase
    density::E
    mu::V

    rho::S1
    nu::S2
    k::S3
    cp::S4
    beta::S5
end
Adapt.@adapt_structure PhaseState

function build_phase(phase_setup::Phase, mesh)
    rho   = ScalarField(mesh)
    nu    = ScalarField(mesh)
    k     = ScalarField(mesh)
    cp    = ScalarField(mesh)
    beta  = ScalarField(mesh)

    return PhaseState(
        density=phase_setup.density,
        mu=phase_setup.mu,
        rho=rho,
        nu=nu,
        k=k,
        cp=cp,
        beta=beta
    )
end



"""
    Multiphase <: AbstractMultiphase

Multiphase fluid model containing multiple phases and their interaction properties.

### Fields
- 'phases'             -- Tuple of PhaseState structures.
- 'physics_properties' -- NamedTuple of physical models (drag, surface tension, etc.).
- 'alpha'              -- Volume fraction ScalarField.
- 'alphaf'             -- Volume fraction FaceScalarField.
- 'rho'                -- Mixture density ScalarField.
- 'rhof'               -- Mixture density FaceScalarField.
- 'nu'                 -- Mixture kinematic viscosity ScalarField.
- 'nuf'                -- Mixture kinematic viscosity FaceScalarField.
- 'p_rgh'              -- Dynamic pressure ScalarField.
- 'p_rghf'             -- Dynamic pressure FaceScalarField.
"""
@kwdef struct Multiphase{P1,P2,S1,F1,S2,F2,S3,F3,S4,F4} <: AbstractMultiphase
    phases::P1
    physics_properties::P2
    alpha::S1
    alphaf::F1
    rho::S2
    rhof::F2
    nu::S3
    nuf::F3
    p_rgh::S4
    p_rghf::F4
end
Adapt.@adapt_structure Multiphase

Fluid{Multiphase}(; phases::Tuple, kwargs...) = begin
    coeffs = (; phases, kwargs...)
    ARG = typeof(coeffs)
    Fluid{Multiphase, ARG}(coeffs)
end


(fluid::Fluid{Multiphase, ARG})(mesh) where {ARG} = begin
    coeffs = fluid.args

    physics_properties = Base.structdiff(coeffs, (phases = nothing,))

    build_multiphase(coeffs.phases, physics_properties, mesh)
end


build_property(property, mesh) = property
build_property(setup::Gravity, mesh) = build_gravityModel(setup, mesh)

function build_multiphase(phase_setups::Tuple{<:AbstractPhase, <:AbstractPhase}, physics_properties_setup::NamedTuple, mesh)
    phases = map(setup -> build_phase(setup, mesh), phase_setups)

    built_properties = map(prop_setup -> build_property(prop_setup, mesh), physics_properties_setup)

    alpha  = ScalarField(mesh)
    alphaf = FaceScalarField(mesh)

    rho  = ScalarField(mesh)
    rhof = FaceScalarField(mesh)

    nu  = ScalarField(mesh)
    nuf = FaceScalarField(mesh)

    p_rgh  = ScalarField(mesh)
    p_rghf = FaceScalarField(mesh)
    
    Multiphase(phases=phases, physics_properties=built_properties, alpha=alpha, alphaf=alphaf, rho=rho, rhof=rhof, nu=nu, nuf=nuf, p_rgh=p_rgh, p_rghf=p_rghf)
end