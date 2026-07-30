export AbstractFluid, AbstractIncompressible, AbstractCompressible
export Fluid
export Incompressible, Incompressible_MRF, WeaklyCompressible, Compressible
export Phase, Fluid, Multiphase
export AbstractModel, AbstractEosModel, AbstractViscosityModel
export AbstractConductivityModel, AbstractHeatCapacityModel, AbstractExpansivityModel
export AbstractMultiphaseModel, VOF, Mixture
export update_phase_property!
export Incompressible, WeaklyCompressible, Compressible, SupersonicFlow

abstract type AbstractFluid end
abstract type AbstractIncompressible <: AbstractFluid end
abstract type AbstractCompressible <: AbstractFluid end
abstract type AbstractMultiphase <: AbstractFluid end

abstract type AbstractPhase <: AbstractMultiphase end
abstract type AbstractModel end
abstract type AbstractEosModel <: AbstractModel end
abstract type AbstractViscosityModel <: AbstractModel end
abstract type AbstractConductivityModel <: AbstractModel end
abstract type AbstractHeatCapacityModel <: AbstractModel end
abstract type AbstractExpansivityModel <: AbstractModel end
abstract type AbstractMultiphaseModel end


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
    scalar = ScalarFloat(mesh)
    nu = ConstantScalar(scalar(nu))
    nuf = nu
    rho = ConstantScalar(scalar(rho))
    rhof = rho
    Incompressible(nu, rho, nuf, rhof)
end

"""
    Incompressible_MRF <: AbstractIncompressible

Incompressible fluid model containing fluid field parameters for incompressible flows that utilise multiple reference frames (MRF).

### Fields
- 'nu'   -- Fluid kinematic viscosity.
- 'rho'  -- Fluid density.
- 'frames' -- Reference frames information.

### Examples
- `Fluid{Incompressible}(nu=0.001, rho=1.0)` - Constructor with default values.
"""
@kwdef struct Incompressible_MRF{S1, S2, F1, F2, RefFrames} <: AbstractIncompressible
    nu::S1
    rho::S2
    nuf::F1
    rhof::F2
    refFrames::RefFrames
end
Adapt.@adapt_structure Incompressible_MRF

Fluid{Incompressible_MRF}(; nu, rho=1.0, refFrames) = begin
    coeffs = (nu=nu, rho=rho, refFrames=refFrames)
    ARG = typeof(coeffs)
    Fluid{Incompressible_MRF,ARG}(coeffs)
end

(fluid::Fluid{Incompressible_MRF, ARG})(mesh) where ARG = begin
    coeffs = fluid.args
    (; rho, nu, refFrames) = coeffs
    scalar = ScalarFloat(mesh)
    nu = ConstantScalar(scalar(nu))
    nuf = nu
    rho = ConstantScalar(scalar(rho))
    rhof = rho
    Incompressible_MRF(nu, rho, nuf, rhof, refFrames)
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
struct WeaklyCompressible{S1, S2, F1, F2, T, VM} <: AbstractCompressible
    nu::S1
    rho::S2
    nuf::F1
    rhof::F2
    cp::T
    gamma::T
    Pr::T
    R::T
    visc_model::VM
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
    nu, nuf, visc_model = initialise_viscosity(nu, mesh)
    rho = ScalarField(mesh)
    rhof = FaceScalarField(mesh)
    WeaklyCompressible(nu, rho, nuf, rhof, cp, gamma, Pr, R, visc_model)
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
@kwdef struct Compressible{S1, S2, F1, F2, T, VM} <: AbstractCompressible
    nu::S1
    rho::S2
    nuf::F1
    rhof::F2
    cp::T
    gamma::T
    Pr::T
    R::T
    visc_model::VM
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
    nu, nuf, visc_model = initialise_viscosity(nu, mesh)
    rho = ScalarField(mesh)
    rhof = FaceScalarField(mesh)
    Compressible(nu, rho, nuf, rhof, cp, gamma, Pr, R, visc_model)
end


"""
    Phase <: AbstractPhase

Configuration structure for a single fluid phase.

### Fields
- `rho`  -- Density model (Equation of State) for the phase.
- `mu`   -- Viscosity model for the phase.
- `k`    -- Thermal conductivity model (optional, `nothing` if not provided).
- `cp`   -- Specific heat capacity model (optional, `nothing` if not provided).
- `beta` -- Thermal expansivity model (optional, `nothing` if not provided).

Any property given as a plain `AbstractFloat` is promoted to the corresponding
constant model (e.g. `k=0.1` becomes `ConstK(0.1)`).

The thermal properties are only required by energy-aware solvers. They default
to `nothing` so that a phase which is missing a property needed by the selected
energy model fails loudly rather than silently defaulting to zero.

### Examples
- `Phase(rho=1000.0, mu=1.0e-3)` - isothermal use.
- `Phase(rho=70.8, mu=13.2e-6, k=0.1, cp=9660.0, beta=0.0164)` - with energy.
"""
struct Phase{E<:AbstractEosModel, V<:AbstractViscosityModel, K, C, B} <: AbstractPhase
    rho::E
    mu::V
    k::K
    cp::C
    beta::B
end

# Covers all combinations e.g. mu=1.8e-5 or mu=SutherlandModel() etc
function Phase(; rho, mu, k=nothing, cp=nothing, beta=nothing)
    rho_model  = rho  isa AbstractFloat ? ConstEos(rho)   : rho
    mu_model   = mu   isa AbstractFloat ? ConstMu(mu)     : mu
    k_model    = k    isa AbstractFloat ? ConstK(k)       : k
    cp_model   = cp   isa AbstractFloat ? ConstCp(cp)     : cp
    beta_model = beta isa AbstractFloat ? ConstBeta(beta) : beta
    return Phase(rho_model, mu_model, k_model, cp_model, beta_model)
end

@kwdef struct PhaseState{E<:AbstractEosModel, V<:AbstractViscosityModel, K, C, B,
                         S1,S2,S3,S4,S5} <: AbstractPhase
    rho_model::E
    mu_model::V
    k_model::K
    cp_model::C
    beta_model::B

    rho::S1
    mu::S2
    k::S3
    cp::S4
    beta::S5
end
Adapt.@adapt_structure PhaseState

# `_phase_property_field` decides the storage for each property (ConstantScalar
# for constant models, ScalarField for variable ones, `nothing` when the
# property was not supplied). Its methods live in 2_thermophysical_models.jl
# alongside the concrete property models, which are included after this file.

function build_phase(phase_setup::Phase, mesh)
    return PhaseState(
        rho_model  = phase_setup.rho,
        mu_model   = phase_setup.mu,
        k_model    = phase_setup.k,
        cp_model   = phase_setup.cp,
        beta_model = phase_setup.beta,

        rho  = _phase_property_field(phase_setup.rho,  mesh),
        mu   = _phase_property_field(phase_setup.mu,   mesh),
        k    = _phase_property_field(phase_setup.k,    mesh),
        cp   = _phase_property_field(phase_setup.cp,   mesh),
        beta = _phase_property_field(phase_setup.beta, mesh),
    )
end

"""
    update_phase_property!(field, model, p, T, config)

Recompute a single per-cell phase property at the given absolute pressure and
temperature. Constant models and unsupplied (`nothing`) properties fall through
to this no-op, so it is safe to call unconditionally each time step.

Variable-property methods are defined alongside their models, e.g.
`update_phase_property!(field, ::IdealGas, ...)` in 2_thermophysical_models.jl.
"""
update_phase_property!(field, model, p, T, config) = nothing

"""
    VOF(; sigma=0.0, cAlpha=1.0) <: AbstractMultiphaseModel

Volume-of-Fluid interface-capturing settings.

### Fields
- `sigma`  -- Surface tension coefficient [N/m].
- `cAlpha` -- Interface compression coefficient (MULES), default is 1.0.
"""
@kwdef struct VOF{T1,T2} <: AbstractMultiphaseModel
    sigma::T1  = 0.0
    cAlpha::T2 = 1.0
end
Adapt.@adapt_structure VOF

"""
    Mixture(; diameter=1.0e-3) <: AbstractMultiphaseModel

Manninen drift-flux mixture-model settings.

### Fields
- `diameter` -- Dispersed-phase particle/bubble diameter [m].
"""
@kwdef struct Mixture{T1} <: AbstractMultiphaseModel
    diameter::T1 = 1.0e-3
end
Adapt.@adapt_structure Mixture

"""
    Multiphase <: AbstractMultiphase

Multiphase fluid model containing multiple phases and their interaction properties.

### Fields
- 'model'              -- Multiphase model selecting the solver pathway (`VOF` or `Mixture`).
- 'phases'             -- Tuple of PhaseState structures.
- 'physics_properties' -- NamedTuple of physical models (drag, surface tension, etc.).
- 'volume_fraction'    -- Index of the phase tracked by the volume fraction field.
- 'alpha'              -- Volume fraction ScalarField.
- 'alphaf'             -- Volume fraction FaceScalarField.
- 'rho'                -- Mixture density ScalarField.
- 'rhof'               -- Mixture density FaceScalarField.
- 'nu'                 -- Mixture kinematic viscosity ScalarField.
- 'nuf'                -- Mixture kinematic viscosity FaceScalarField.
- 'p_rgh'              -- Dynamic pressure ScalarField.
- 'p_rghf'             -- Dynamic pressure FaceScalarField.
"""
@kwdef struct Multiphase{M,P1,P2,S1,F1,S2,F2,S3,F3,S4,F4} <: AbstractMultiphase
    model::M
    phases::P1
    physics_properties::P2
    volume_fraction::Int
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

Fluid{Multiphase}(; phases::NTuple{2, Phase}, model=nothing, kwargs...) = begin
    @assert model isa AbstractMultiphaseModel "Expected `model = VOF(...)` or `model = Mixture(...)`, got: $(typeof(model))"
    coeffs = (; phases, model, kwargs...)
    ARG = typeof(coeffs)
    Fluid{Multiphase, ARG}(coeffs)
end

(fluid::Fluid{Multiphase, ARG})(mesh) where {ARG} = begin
    coeffs = fluid.args
    physics_properties = Base.structdiff(coeffs, (phases = nothing, model = nothing))

    phase_setups = coeffs.phases
    @assert phase_setups isa Tuple{Phase, Phase} "Phases must be a plain Tuple of exactly two Phase objects, e.g. (Phase(...), Phase(...))"

    volume_fraction = 1  # First phase is always the tracked phase

    build_multiphase(coeffs.model, phase_setups, physics_properties, mesh, volume_fraction)
end

build_property(property, mesh) = property
build_property(setup::Gravity, mesh) = build_gravityModel(setup, mesh)

function build_multiphase(model::AbstractMultiphaseModel, phase_setups::Tuple{<:AbstractPhase, <:AbstractPhase}, physics_properties_setup::NamedTuple, mesh, volume_fraction::Int)
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

    Multiphase(model=model, phases=phases, physics_properties=built_properties, volume_fraction=volume_fraction, alpha=alpha, alphaf=alphaf, rho=rho, rhof=rhof, nu=nu, nuf=nuf, p_rgh=p_rgh, p_rghf=p_rghf)
end

"""
    SupersonicFlow <: AbstractCompressible

Fluid model for density-based (explicit) supersonic flow solver.
Uses Rusanov (Local Lax-Friedrichs) flux with Forward Euler time integration.

### Fields
- `nu`    -- Kinematic viscosity (ConstantScalar).
- `rho`   -- Density field (ScalarField, updated each iteration).
- `nuf`   -- Face kinematic viscosity.
- `rhof`  -- Face density field.
- `cp`    -- Specific heat at constant pressure (ConstantScalar).
- `gamma` -- Ratio of specific heats (ConstantScalar).
- `Pr`    -- Prandtl number (ConstantScalar).
- `R`     -- Specific gas constant cp*(1 - 1/gamma) (ConstantScalar).

### Examples
- `Fluid{SupersonicFlow}(nu=1E-5, cp=1005.0, gamma=1.4, Pr=0.7)` - Constructor with default values.
"""
struct SupersonicFlow{S1, S2, F1, F2, T} <: AbstractCompressible
    nu::S1
    rho::S2
    nuf::F1
    rhof::F2
    cp::T
    gamma::T
    Pr::T
    R::T
end
Adapt.@adapt_structure SupersonicFlow

Fluid{SupersonicFlow}(; nu=1E-5, cp=1005.0, gamma=1.4, Pr=0.7) = begin
    coeffs = (nu=nu, cp=cp, gamma=gamma, Pr=Pr)
    ARG = typeof(coeffs)
    Fluid{SupersonicFlow,ARG}(coeffs)
end

(fluid::Fluid{SupersonicFlow, ARG})(mesh) where ARG = begin
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
    SupersonicFlow(nu, rho, nuf, rhof, cp, gamma, Pr, R)
end
