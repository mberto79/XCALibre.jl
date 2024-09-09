export AbstractFluid, AbstractIncompressible, AbstractCompressible
export Fluid
export Incompressible, WeaklyCompressible, Compressible
export _nu, _nuf, _rho
export _R, _Cp, _mu, _Pr

abstract type AbstractFluid end
abstract type AbstractIncompressible <: AbstractFluid end
abstract type AbstractCompressible <: AbstractFluid end

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