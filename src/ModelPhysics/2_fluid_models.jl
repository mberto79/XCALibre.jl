export AbstractFluid, AbstractIncompressible, AbstractCompressible
export FLUID
export Incompressible, WeaklyCompressible, Compressible
export _nu, _nuf, _rho
export _R, _Cp, _mu, _Pr

abstract type AbstractFluid end
abstract type AbstractIncompressible <: AbstractFluid end
abstract type AbstractCompressible <: AbstractFluid end

struct FLUID{T,ARG} <: AbstractFluid
    args::ARG
end

struct Incompressible{S1, S2, F1, F2} <: AbstractFluid
    nu::S1
    rho::S2
    nuf::F1
    rhof::F2
end
Adapt.@adapt_structure Incompressible

FLUID{Incompressible}(; nu=0.001, rho=1.0) = begin
    coeffs = (nu=nu, rho=rho)
    ARG = typeof(coeffs)
    FLUID{Incompressible,ARG}(coeffs)
end

(fluid::FLUID{Incompressible, ARG})(mesh) where ARG = begin
    coeffs = fluid.args
    (; rho, nu) = coeffs
    nu = ConstantScalar(nu)
    nuf = nu
    rho = ConstantScalar(rho)
    rhof = rho
    Incompressible(nu, rho, nuf, rhof)
end

_nu(fluid::AbstractIncompressible) = fluid.nu
_rho(fluid::AbstractIncompressible) = fluid.rho
_nuf(fluid::AbstractIncompressible) = fluid.nuf

@kwdef struct WeaklyCompressible{T, S, FS} <: AbstractCompressible
    nu::T
    cp::T
    gamma::T
    Pr::T
    rho::S
    rhof::FS
end
Adapt.@adapt_structure WeaklyCompressible

@kwdef struct Compressible{T, S, FS} <: AbstractCompressible
    nu::T
    cp::T
    gamma::T
    Pr::T
    rho::S
    rhof::FS
end
Adapt.@adapt_structure Compressible

_R(fluid::AbstractCompressible) = ConstantScalar(fluid.cp.values*(1.0 - (1.0/fluid.gamma.values)))
_Cp(fluid::AbstractCompressible) = fluid.cp
_nu(fluid::AbstractCompressible) = fluid.nu
_Pr(fluid::AbstractCompressible) = fluid.Pr
_rho(fluis::AbstractCompressible) = fluid.rho

_nu(fluid::AbstractCompressible) = fluid.nu
_nuf(fluid::AbstractCompressible) = fluid.nu