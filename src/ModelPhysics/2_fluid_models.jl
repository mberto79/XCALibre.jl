export AbstractFluid, AbstractIncompressible, AbstractCompressible
export Incompressible, WeaklyCompressible, Compressible
export _nu, _rho
export _R, _Cp, _mu, _Pr

abstract type AbstractFluid end
abstract type AbstractIncompressible <: AbstractFluid end
abstract type AbstractCompressible <: AbstractFluid end

@kwdef struct Incompressible{T, F} <: AbstractIncompressible
    nu::T
    rho::F
end
Adapt.@adapt_structure Incompressible

_nu(fluid::AbstractIncompressible) = fluid.nu
_rho(fluid::AbstractIncompressible) = fluid.rho

@kwdef struct WeaklyCompressible{T, F} <: AbstractCompressible
    mu::T
    cp::T
    gamma::T
    Pr::T
    rho::F
end
Adapt.@adapt_structure WeaklyCompressible

@kwdef struct Compressible{T, F} <: AbstractCompressible
    mu::T
    cp::T
    gamma::T
    Pr::T
    rho::F
end
Adapt.@adapt_structure Compressible

_R(fluid::AbstractCompressible) = ConstantScalar(fluid.cp.values*(1.0 - (1.0/fluid.gamma.values)))
_Cp(fluid::AbstractCompressible) = fluid.cp
_mu(fluid::AbstractCompressible) = fluid.mu
_Pr(fluid::AbstractCompressible) = fluid.Pr
_rho(fluis::AbstractCompressible) = fluid.rho