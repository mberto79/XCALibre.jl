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

@kwdef struct Incompressible{S1, S2, F1, F2} <: AbstractIncompressible
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

# _nu(fluid::AbstractIncompressible) = fluid.nu
# _rho(fluid::AbstractIncompressible) = fluid.rho
# _nuf(fluid::AbstractIncompressible) = fluid.nuf

@kwdef struct WeaklyCompressible{S1, S2, F1, F2, T} <: AbstractCompressible
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

FLUID{WeaklyCompressible}(; nu=1E-5, cp=1005.0, gamma=1.4, Pr=0.7 ) = begin
    coeffs = (nu=nu, cp=cp, gamma=gamma, Pr=Pr)
    ARG = typeof(coeffs)
    FLUID{WeaklyCompressible,ARG}(coeffs)
end

(fluid::FLUID{WeaklyCompressible, ARG})(mesh) where ARG = begin
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

FLUID{Compressible}(; nu=1E-5, cp=1005.0, gamma=1.4, Pr=0.7 ) = begin
    coeffs = (nu=nu, cp=cp, gamma=gamma, Pr=Pr)
    ARG = typeof(coeffs)
    FLUID{Compressible,ARG}(coeffs)
end

(fluid::FLUID{Compressible, ARG})(mesh) where ARG = begin
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


# _R(fluid::AbstractCompressible) = ConstantScalar(fluid.cp.values*(1.0 - (1.0/fluid.gamma.values)))
# _Cp(fluid::AbstractCompressible) = fluid.cp
# _nu(fluid::AbstractCompressible) = fluid.nu
# _Pr(fluid::AbstractCompressible) = fluid.Pr
# _rho(fluis::AbstractCompressible) = fluid.rho

# _nu(fluid::AbstractCompressible) = fluid.nu
# _nuf(fluid::AbstractCompressible) = fluid.nu