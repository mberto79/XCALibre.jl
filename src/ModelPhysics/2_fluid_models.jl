export AbstractFluid, AbstractIncompressible, AbstractCompressible
export Fluid
export AbstractWallLubrication, Antal, Frank, wall_lubrication_coefficient
export AbstractLift, TomiyamaLift, ConstantLift, lift_coefficient
export Incompressible, Incompressible_MRF, WeaklyCompressible, Compressible
export Phase, Fluid, Multiphase
export AbstractModel, AbstractEosModel, AbstractViscosityModel
export AbstractConductivityModel, AbstractHeatCapacityModel, AbstractExpansivityModel
export AbstractMultiphaseModel, VOF, Mixture, implicit_alpha_transport
export update_phase_property!, update_phase_properties!
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
    # Property models are read INSIDE kernels (e.g. `phase_compressibility` is
    # called on `phase.rho_model` per cell), so any array a model carries - the
    # lookup tables of the `Tabulated*` models - has to live on the same backend
    # as the mesh. The `Const*` models hold no arrays and pass through unchanged.
    backend = _get_backend(mesh)
    to_device(m) = adapt(backend, m)

    return PhaseState(
        rho_model  = to_device(phase_setup.rho),
        mu_model   = to_device(phase_setup.mu),
        k_model    = to_device(phase_setup.k),
        cp_model   = to_device(phase_setup.cp),
        beta_model = to_device(phase_setup.beta),

        rho  = _phase_property_field(phase_setup.rho,  mesh),
        mu   = _phase_property_field(phase_setup.mu,   mesh),
        k    = _phase_property_field(phase_setup.k,    mesh),
        cp   = _phase_property_field(phase_setup.cp,   mesh),
        beta = _phase_property_field(phase_setup.beta, mesh),
    )
end

"""
    update_phase_properties!(phase, p_abs, T, config)

Refresh every variable property of a phase at the current absolute pressure and
temperature: density, viscosity, conductivity, heat capacity and expansivity.

Each call dispatches on that property's model, so constant models and properties
that were never supplied (`nothing`) fall through to the no-op
[`update_phase_property!`](@ref). Safe - and intended - to call unconditionally
every time step.
"""
function update_phase_properties!(phase, p_abs, T, config)
    update_phase_property!(phase.rho,  phase.rho_model,  p_abs, T, config)
    update_phase_property!(phase.mu,   phase.mu_model,   p_abs, T, config)
    update_phase_property!(phase.k,    phase.k_model,    p_abs, T, config)
    update_phase_property!(phase.cp,   phase.cp_model,   p_abs, T, config)
    update_phase_property!(phase.beta, phase.beta_model, p_abs, T, config)
    return nothing
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
    Mixture(; diameter=1.0e-3, alpha_transport=:mules) <: AbstractMultiphaseModel

Manninen drift-flux mixture-model settings.

### Fields
- `diameter` -- Dispersed-phase particle/bubble diameter [m].
- `alpha_transport` -- How the volume fraction is advanced, `:mules` (default)
  or `:implicit`.

### `alpha_transport`

`:mules` is the explicit flux-corrected update shared with `VOF`. It is bounded
by construction, but the limiter's boundedness argument fixes the update form to
the advective rearrangement, which is exact only when `div(u) = 0`. That makes
the compressibility and phase-change terms awkward to include correctly, and it
imposes an alpha-Courant time step limit.

`:implicit` solves a conservative transport equation as a linear system instead,
using `solvers.alpha`. Sources enter the matrix rather than being applied
afterwards, and the Courant limit disappears. Boundedness is enforced by a clamp
rather than guaranteed.

!!! warning "Measured worse on the LH2 pipe case"
    `:implicit` is the more defensible formulation on paper, but on the
    forced-convection boiling case it made the discrete vapour mass balance
    *worse* (residual 3-30x the phase change rate against ~1x for `:mules`), and
    `max|U|` and the near-wall cooling both degraded. The default is therefore
    unchanged. See `dev_notes_LH2_pipe_boiling.md` before selecting it.
"""
@kwdef struct Mixture{T1,S} <: AbstractMultiphaseModel
    diameter::T1 = 1.0e-3
    alpha_transport::S = :mules
end
Adapt.@adapt_structure Mixture

# =============================================================================
#  Wall lubrication force
# =============================================================================

"""
    AbstractWallLubrication

Lateral force pushing dispersed bubbles AWAY from a wall.

### Why it is needed

A bubble approaching a wall must drain the liquid film between the two, and that
drainage resists - a lubrication effect. The resulting force is short range and
repulsive, and it is what stops the near-wall void fraction pinning at 1 in
bubbly wall flows.

It is ORIENTATION INDEPENDENT: it depends on the wall normal and on the
wall-PARALLEL relative velocity, not on gravity. That matters for a vertical pipe,
where buoyancy drift is purely axial and therefore cannot move vapour off the
wall at all.

### Why not lift instead

The lift force is the other lateral mechanism, but its sign is set by bubble
size. Tomiyama's `C_L` changes sign at `Eo_d ~ 4`, i.e. `d ~ 2.58 mm` in LH2 at
0.4 MPa. Departure diameters here are 107-429 um, some 24x smaller, so `C_L` is
POSITIVE and lift drives bubbles TOWARD the wall - it would deepen the wall peak
rather than relieve it. That is also why real bubbly upflow is wall peaked.

The general form is

    F_WL = C_w * rho_c * alpha_d * |U_r,parallel|^2 * n_wall        [N/m^3]

with `C_w` [1/m] supplied by the concrete model.
"""
abstract type AbstractWallLubrication end

"""
    Antal(; Cw1 = -0.104, Cw2 = 0.147)

Antal, Lahey & Flaherty (1991):

    C_w = max(0, (Cw1 - 0.06*|U_r,par|)/d_b + Cw2/y)

Grows as `1/y` towards the wall and cuts off at `y = -Cw2*d/Cw1 ~ 1.41*d_b`.

SHORT RANGE by construction - about 2.7 cells on the LH2 pipe's near-wall mesh.
That is the intent rather than a limitation: the job is to stop the wall cell
saturating, not to flatten the profile, and bubbly upflow genuinely is wall
peaked. MEASURED for that case (d = 107 um, |U_r,par| = 0.0412 m/s): the balance
against Stokes drag gives a wall-normal velocity of 0.033 m/s at the first cell
centre against the 0.0094 m/s needed to clear the cell within its 5.9 ms fill
time - a 3.5x margin, falling to break-even by the second cell.
"""
struct Antal{F} <: AbstractWallLubrication
    Cw1::F
    Cw2::F
end
Antal(; Cw1 = -0.104, Cw2 = 0.147) = Antal(float(Cw1), float(Cw2))
Adapt.@adapt_structure Antal

"""
    Frank(; Cwd = 6.8, Cwc = 10.0, p = 1.7, Cw3 = 1.0)

Frank et al. (2008) generalisation, with a LONGER and tunable range:

    C_w = Cw3 * max(0, 1 - y/(Cwc*d_b)) / (Cwd * d_b * (y/d_b)^p)

Cuts off at `y = Cwc*d_b`, i.e. ~10 bubble diameters by default - about 19 cells,
or 36% of the pipe radius, on the LH2 pipe. Use when the void needs spreading
further than [`Antal`](@ref) reaches; `Cwc` is the knob for that range.
"""
struct Frank{F} <: AbstractWallLubrication
    Cwd::F
    Cwc::F
    p::F
    Cw3::F
end
Frank(; Cwd = 6.8, Cwc = 10.0, p = 1.7, Cw3 = 1.0) =
    Frank(float(Cwd), float(Cwc), float(p), float(Cw3))
Adapt.@adapt_structure Frank

"""
    wall_lubrication_coefficient(model, d_b, y, Ur_par) -> C_w  [1/m]

`C_w` in `F_WL = C_w rho_c alpha_d |U_r,par|^2 n_wall`. Zero beyond the model's
range, so the force switches itself off away from walls.
"""
# WALL DISTANCE IS FLOORED AT THE BUBBLE RADIUS, and this is load bearing.
#
# Both correlations carry a `1/y`-type singularity, and both are derived for a
# sphere standing OFF the wall by `y`. Below `y = d_b/2` the bubble centre would
# be closer to the wall than its own radius - it would intersect the wall - so
# the premise fails and the coefficient diverges for a purely geometric reason.
# Worse, it diverges WITH MESH REFINEMENT: halve the first cell and the force on
# it doubles, which makes the whole model resolution dependent.
#
# MEASURED on the LH2 pipe (d_b = 107.3 um, first cell centre y = 27.85 um, so
# y/d = 0.26 - already inside the bubble):
#
#   y used        y/d    C_w [1/m]   U_wl [m/s]
#   27.85 um     0.26         4286      0.0331    <- unfloored, over-empties
#   53.67 um     0.50         1747      0.0135    <- floored at d_b/2
#  107.3  um     1.00          378      0.0029    <- too weak
#
# against 0.0094 m/s needed to clear the wall cell within its fill time. Without
# the floor the first cell was emptied to alpha = 0.030 against 0.099 in the next
# cell out, with 54% azimuthal scatter and some cells clamped at exactly zero -
# i.e. the `clamp!(alpha, 0, 1)` firing and destroying vapour mass.
@inline _wl_y(d_b::F, y) where F = max(y, F(0.5)*d_b)

@inline function wall_lubrication_coefficient(m::Antal, d_b::F, y, Ur_par) where F
    y <= zero(F) && return zero(F)
    ye = _wl_y(d_b, y)
    Cw1 = F(m.Cw1) - F(0.06)*Ur_par
    return max(zero(F), Cw1/d_b + F(m.Cw2)/ye)
end

@inline function wall_lubrication_coefficient(m::Frank, d_b::F, y, Ur_par) where F
    y <= zero(F) && return zero(F)
    ycut = F(m.Cwc)*d_b
    y >= ycut && return zero(F)
    ye = _wl_y(d_b, y)
    return F(m.Cw3)*(one(F) - ye/ycut)/(F(m.Cwd)*d_b*(ye/d_b)^F(m.p))
end

@inline wall_lubrication_coefficient(::Nothing, d_b::F, y, Ur_par) where F = zero(F)

# =============================================================================
#  Lift force
# =============================================================================

"""
    AbstractLift

Lateral force on a dispersed bubble in a SHEARED continuous phase,

    F_L = -C_L * rho_c * alpha_d * (U_r x curl(U_c))            [N/m^3]

Orientation independent: it is set by the local vorticity, not by gravity.

### Sign, and why it matters here

`C_L > 0` drives bubbles toward the wall in upflow; `C_L < 0` drives them toward
the core. Tomiyama's correlation changes sign at `Eo_d ~ 4`, which in LH2 at
0.4 MPa is `d ~ 2.58 mm`. Departure diameters on the pipe case are 107-429 um -
some 24x smaller - so `C_L ~ +0.29` and lift acts TOWARD the wall.

That is not a reason to leave it out. It is the physical counterpart of
[`AbstractWallLubrication`](@ref), which acts away from the wall, and the two
together set the void profile of bubbly pipe flow: wall-peaked, but not
saturated. With lubrication alone the near-wall layer is over-evacuated wherever
no evaporative source refills it - MEASURED downstream of the heater on the LH2
pipe, void 0.085 at the wall against a 0.273 peak 300 um out, a 3.2x depletion
where the real profile is wall PEAKED.
"""
abstract type AbstractLift end

"""
    TomiyamaLift(; C_max = 0.288, C_deformed = -0.27)

Tomiyama et al. (2002), through the modified Eotvos number of the DEFORMED
bubble:

    Eo   = g*drho*d^2/sigma
    d_H  = d*(1 + 0.163*Eo^0.757)^(1/3)
    Eo_d = g*drho*d_H^2/sigma
    f(E) = 0.00105E^3 - 0.0159E^2 - 0.0204E + 0.474

    C_L = min(C_max*tanh(0.121*Re_p), f(Eo_d))   Eo_d < 4
        = f(Eo_d)                                4 <= Eo_d <= 10
        = C_deformed                             Eo_d > 10

Small, near-spherical bubbles get `C_L > 0` (toward the wall); large deformed
ones get `C_L < 0` (toward the core). `d_H` is what makes that switch depend on
DEFORMATION rather than raw size.
"""
struct TomiyamaLift{F} <: AbstractLift
    C_max::F
    C_deformed::F
end
TomiyamaLift(; C_max = 0.288, C_deformed = -0.27) =
    TomiyamaLift(float(C_max), float(C_deformed))
Adapt.@adapt_structure TomiyamaLift

"""
    ConstantLift(; C_L)

Fixed lift coefficient. For isolating the SIGN and MAGNITUDE of the lift response
without the Tomiyama correlation's size dependence in the way - set `C_L = 0` to
disable lift while keeping the code path, or a negative value to force
core-peaking irrespective of bubble size.
"""
struct ConstantLift{F} <: AbstractLift
    C_L::F
end
ConstantLift(; C_L) = ConstantLift(float(C_L))
Adapt.@adapt_structure ConstantLift

"""
    lift_coefficient(model, d_b, drho, sigma, g, Re_p) -> C_L

Sign convention: POSITIVE drives the dispersed phase toward the wall in upflow.
"""
@inline lift_coefficient(m::ConstantLift, d_b::F, drho, sigma, g, Re_p) where F =
    F(m.C_L)

@inline function lift_coefficient(m::TomiyamaLift, d_b::F, drho, sigma, g, Re_p) where F
    (sigma <= zero(F) || drho <= zero(F)) && return zero(F)
    Eo   = g*drho*d_b*d_b/sigma
    d_H  = d_b*cbrt(one(F) + F(0.163)*Eo^F(0.757))
    Eo_d = g*drho*d_H*d_H/sigma
    fE   = F(0.00105)*Eo_d^3 - F(0.0159)*Eo_d^2 - F(0.0204)*Eo_d + F(0.474)
    return Eo_d < F(4)  ? min(F(m.C_max)*tanh(F(0.121)*Re_p), fE) :
           Eo_d <= F(10) ? fE : F(m.C_deformed)
end

@inline lift_coefficient(::Nothing, d_b::F, drho, sigma, g, Re_p) where F = zero(F)

"""True when the mixture model advances the volume fraction implicitly."""
implicit_alpha_transport(m::Mixture) = m.alpha_transport === :implicit
implicit_alpha_transport(m) = false

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
