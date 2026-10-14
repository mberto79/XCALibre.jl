export AbstractScheme, AbstractBoundary
export AbstractDirichlet, AbstractNeumann, AbstractPhysicalConstraint, AbstractWallFunction
export KWallFunction, OmegaWallFunction, NutWallFunction, NutMixingLengthWallFunction
# export Constant, Linear, Upwind, LUST
export Linear, Upwind, LUST, LUSTDC, LUSTLU, LinearUpwind, LimitedLinearUpwind
export BoundedUpwind
export SteadyState, Euler, CrankNicolson
export Gauss, Midpoint, LeastSquares, WeightedLeastSquares

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
the upwind face value. A converged steady solution has the face values of `LUST`,
``\phi_f = \frac{3}{4}\phi_{lin} + \frac{1}{4}\phi_U``. Boundary faces use the `LUST`
boundary coefficients (periodic patches keep its implicit coupling). No gradient is needed.
Intended for steady runs: in transient runs the correction lags one time step, so results
differ from `LUST` by ``O(\Delta t)`` and the stable time step is smaller; use `LUST` for
transient and LES runs. This is not OpenFOAM's `LUST`, which keeps the linear part implicit and
corrects the remaining quarter explicitly with the upwind-cell gradient, as `LinearUpwind` does.
"""
struct LUSTDC <: AbstractScheme end

@doc raw"""
    LUSTLU <: AbstractScheme

`LUST` plus an explicit second-order correction of its upwind share, for transient and LES runs.
The matrix holds the `LUST` coefficients (75 % linear, 25 % upwind), and the upwind quarter of the
face value is corrected explicitly with the gradient of the upwind cell, as in `LinearUpwind`:

``\phi_f = \frac{3}{4}\phi_{lin} + \frac{1}{4}\left[\phi_U + (\nabla\phi)_U \cdot (\mathbf{x}_f - \mathbf{x}_U)\right]``.

The face value is second order on any mesh and `LUST`'s first-order dissipation is removed. The
gradient is supplied through the equation's operator, `Divergence{LUSTLU}(flux, phi, grad)`; the
solvers pass the gradient of `U` (limited with `Schemes(limiter=...)`). Boundary faces take the
`LUST` coefficients; periodic patches also get the correction. Without a gradient the scheme
reduces to `LUST` (as in the film and multiphase solvers). It is not available for `he`.
"""
struct LUSTLU <: AbstractScheme end

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

@doc raw"""
    LeastSquares <: AbstractScheme

Gradient scheme, selected with `Schemes(gradient=LeastSquares)`. The gradient of a cell is the
unweighted least-squares fit ``\min \sum_f (\phi_f - \phi_P - \nabla\phi\cdot\mathbf{d}_f)^2``
over its faces, with ``\mathbf{d}_f`` the vector from the cell centre to the neighbour centre
(internal faces) or to the face centre (boundary faces, which use the boundary value). It is
exact for linear fields on any mesh, skewed or stretched, where `Gauss` is not, and is computed
cell by cell with no extra storage. See also `WeightedLeastSquares`.
"""
struct LeastSquares <: AbstractScheme end

@doc raw"""
    WeightedLeastSquares <: AbstractScheme

`LeastSquares` with each face weighted by the inverse square of its distance,
``w_f = 1/|\mathbf{d}_f|^2`` (the weighting of OpenFOAM's `leastSquares`), so near neighbours
dominate on stretched meshes. Exact for linear fields.
"""
struct WeightedLeastSquares <: AbstractScheme end
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