export AbstractFluid, AbstractIncompressible, AbstractCompressible
export Incompressible, WeaklyCompressible, Compressible
export _nu
export _R, _Cp, _mu, _Pr

abstract type AbstractFluid end
abstract type AbstractIncompressible <: AbstractFluid end
abstract type AbstractCompressible <: AbstractFluid end

@kwdef struct Incompressible{T} <: AbstractIncompressible
    nu::T
end
Adapt.@adapt_structure Incompressible

_nu(fluid::AbstractIncompressible) = fluid.nu
# rho(fluid::Incompressible) = fluid.nu

@kwdef struct WeaklyCompressible{T} <: AbstractCompressible
    mu::T
    cp::T
    gamma::T
    Pr::T
end
Adapt.@adapt_structure WeaklyCompressible

@kwdef struct Compressible{T} <: AbstractCompressible
    mu::T
    cp::T
    gamma::T
    Pr::T
end
Adapt.@adapt_structure Compressible

_R(fluid::AbstractCompressible) = ConstantScalar(fluid.cp.values*(1.0 - (1.0/fluid.gamma.values)))
_Cp(fluid::AbstractCompressible) = fluid.cp
_mu(fluid::AbstractCompressible) = fluid.mu
_Pr(fluid::AbstractCompressible) = fluid.Pr