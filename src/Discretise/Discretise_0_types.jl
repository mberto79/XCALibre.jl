export AbstractScheme, AbstractBoundary
export AbstractDirichlet, AbstractNeumann, AbstractPhysicalConstraint, AbstractWallFunction
export KWallFunction, OmegaWallFunction, NutWallFunction, NutMixingLengthWallFunction
# export Constant, Linear, Upwind, LUST
export Linear, Upwind, LUST, LUSTDC, LinearUpwind, LimitedLinearUpwind
export BoundedUpwind
export SteadyState, Euler, CrankNicolson
export Gauss, Midpoint

# SUPPORTED DISCRETISATION SCHEMES 

abstract type AbstractScheme end
# struct Constant <: AbstractScheme end
struct Linear <: AbstractScheme end
struct Upwind <: AbstractScheme end
struct LUST <: AbstractScheme end

@doc raw"""
    LUSTDC <: AbstractScheme

`LUST` in deferred-correction form. The matrix holds the upwind coefficients, so it stays
diagonally dominant, and the face correction ``\frac{3}{4}(\phi_{lin} - \phi_U)`` is added
explicitly from the current field, with ``\phi_{lin}`` the linearly interpolated and ``\phi_U``
the upwind face value. A converged steady solution is that of `LUST`,
``\phi_f = \frac{3}{4}\phi_{lin} + \frac{1}{4}\phi_U``; boundary faces use the `LUST`
boundary coefficients. No gradient is needed. In transient runs the correction lags by one
momentum assembly (one time step in `piso!`/`cpiso!`), so results differ from `LUST` by
``O(\Delta t)``. This is not OpenFOAM's `LUST`, which keeps the linear part implicit and adds
an explicit `linearUpwind` correction to the remaining quarter.
"""
struct LUSTDC <: AbstractScheme end

@doc raw"""
    LinearUpwind <: AbstractScheme

Second-order upwind divergence scheme. The matrix holds the upwind coefficients and the face
value is corrected explicitly (deferred correction) with the gradient of the upwind cell,
``\phi_f = \phi_U + (\nabla\phi)_U \cdot (\mathbf{x}_f - \mathbf{x}_U)``. The gradient is
supplied through the equation's operator, `Divergence{LinearUpwind}(flux, phi, grad)`; the
solvers pass the gradient of `U` (limited with `Schemes(limiter=...)`). Boundary faces use
upwind values. Without a gradient the scheme reduces to `Upwind`.
"""
struct LinearUpwind <: AbstractScheme end

@doc raw"""
    LimitedLinearUpwind <: AbstractScheme

Bounded form of `LinearUpwind`. The face correction ``\delta`` is limited per face with the
downwind value, ``0 \le \delta/\Delta \le \min(r, 1)``, where ``\Delta = \phi_D - \phi_U`` and
``r = 2 (\nabla\phi)_U \cdot (\mathbf{x}_D - \mathbf{x}_U)/\Delta - 1``. The face value then lies
between the upwind and downwind values and local extrema are convected with upwind values. Steady
solutions stay bounded on regular meshes, and over- and undershoots are much reduced on irregular
ones. The limiter is inactive for linear fields on uniform meshes (``r = 1``) and reduces the
scheme to first order at extrema and discontinuities.
"""
struct LimitedLinearUpwind <: AbstractScheme end
struct BoundedUpwind <: AbstractScheme end
struct Gauss <: AbstractScheme end
struct Midpoint <: AbstractScheme end
struct SteadyState <: AbstractScheme end 
struct Euler <: AbstractScheme end 
struct CrankNicolson <: AbstractScheme end # not implemented yet


# SUPPORTED BOUNDARY CONDITIONS 

abstract type AbstractBoundary end
abstract type AbstractDirichlet <: AbstractBoundary end
abstract type AbstractNeumann <: AbstractBoundary end
abstract type AbstractWallFunction <: AbstractBoundary end
abstract type AbstractPhysicalConstraint <: AbstractBoundary end

y_plus_laminar(E, kappa) = begin
    yL = 11.0; for i ∈ 1:10; yL = log(max(yL*E, 1.0))/kappa; end
    yL
end

# Kwall function structure and constructor
struct KWallFunction{I,V,R<:UnitRange} <: AbstractWallFunction
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure KWallFunction

@kwdef struct KWallFunctionValue{F<:AbstractFloat}
    kappa::F
    beta1::F
    cmu::F
    B::F
    E::F
    yPlusLam::F
end
Adapt.@adapt_structure KWallFunctionValue

adapt_value(value::KWallFunctionValue, mesh) = begin
    F = _get_float(mesh)
    (; kappa, beta1, cmu, B, E, yPlusLam) = value
    KWallFunctionValue{F}(
            kappa=kappa, beta1=beta1, cmu=cmu, B=B, E=E, yPlusLam=yPlusLam
            )
end
KWallFunction(name::Symbol; kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8) = begin
    yPlusLam = y_plus_laminar(E, kappa)
    KWallFunction(name, KWallFunctionValue(
        kappa=kappa, beta1=beta1, cmu=cmu, B=B, E=E, yPlusLam=yPlusLam))
end
# Omega wall function structure and constructor
struct OmegaWallFunction{I,V,R<:UnitRange} <: AbstractWallFunction
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure OmegaWallFunction

@kwdef struct OmegaWallFunctionValue{F<:AbstractFloat}
    kappa::F
    beta1::F
    cmu::F
    B::F
    E::F
    yPlusLam::F
end
Adapt.@adapt_structure OmegaWallFunctionValue

adapt_value(value::OmegaWallFunctionValue, mesh) = begin
    F = _get_float(mesh)
    (; kappa, beta1, cmu, B, E, yPlusLam) = value
    OmegaWallFunctionValue{F}(
            kappa=kappa, beta1=beta1, cmu=cmu, B=B, E=E, yPlusLam=yPlusLam
            )
end

OmegaWallFunction(name::Symbol; kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8) = begin
    yPlusLam = y_plus_laminar(E, kappa)
    OmegaWallFunction(
        name, OmegaWallFunctionValue(
            kappa=kappa, beta1=beta1, cmu=cmu, B=B, E=E, yPlusLam=yPlusLam)
        )
end

# Nut wall function structure and constructor
struct NutWallFunction{I,V,R<:UnitRange} <: AbstractWallFunction 
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure NutWallFunction

@kwdef struct NutWallFunctionValue{F<:AbstractFloat}
    kappa::F
    beta1::F
    cmu::F
    B::F
    E::F
    yPlusLam::F
end
Adapt.Adapt.@adapt_structure NutWallFunctionValue

adapt_value(value::NutWallFunctionValue, mesh) = begin
    F = _get_float(mesh)
    (; kappa, beta1, cmu, B, E, yPlusLam) = value
    NutWallFunctionValue{F}(
            kappa=kappa, beta1=beta1, cmu=cmu, B=B, E=E, yPlusLam=yPlusLam
            )
end

NutWallFunction(name::Symbol; kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8) = begin
    yPlusLam = y_plus_laminar(E, kappa)
    NutWallFunction(name, NutWallFunctionValue(kappa=kappa, beta1=beta1, cmu=cmu, B=B, E=E, yPlusLam=yPlusLam))
end
# Nut mixing-length wall function (LES wall model, no TKE required)
struct NutMixingLengthWallFunction{I,V,R<:UnitRange} <: AbstractWallFunction
    ID::I
    value::V
    IDs_range::R
end
Adapt.@adapt_structure NutMixingLengthWallFunction

@kwdef struct NutMixingLengthWallFunctionValue{F<:AbstractFloat}
    kappa::F
    E::F
    yPlusLam::F
end
Adapt.@adapt_structure NutMixingLengthWallFunctionValue

adapt_value(value::NutMixingLengthWallFunctionValue, mesh) = begin
    F = _get_float(mesh)
    (; kappa, E, yPlusLam) = value
    NutMixingLengthWallFunctionValue{F}(kappa=kappa, E=E, yPlusLam=yPlusLam)
end

NutMixingLengthWallFunction(name::Symbol; kappa=0.41, E=9.8) = begin
    yPlusLam = y_plus_laminar(E, kappa)
    NutMixingLengthWallFunction(
        name, NutMixingLengthWallFunctionValue(kappa=kappa, E=E, yPlusLam=yPlusLam)
    )
end