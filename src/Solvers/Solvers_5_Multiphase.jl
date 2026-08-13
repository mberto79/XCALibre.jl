export multiphase!
export relax_source!

function multiphase!(
    model, config;
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=2)

    residuals = setup_multiphase_solvers(
        MULTIPHASE, model, config;
        output=output,
        pref=pref,
        ncorrectors=ncorrectors,
        inner_loops=inner_loops
        )

    return residuals
end

"""
    is_compressible_multiphase(phases) -> Bool

True when any phase has a non-constant equation of state, in which case the
pressure equation needs a compressibility term.
"""
is_compressible_multiphase(phases) = any(ph -> !(ph.rho_model isa ConstEos), phases)

"""
    multiphase_p_operating(fluid)

Operating (datum) pressure of the multiphase fluid, from the optional
`p_operating` keyword of `Fluid{Multiphase}`. Defaults to zero, i.e. `p` is a
gauge pressure, which is what the incompressible cases assume.

A compressible phase needs the ABSOLUTE pressure for its equation of state, so
`p_operating` must be set for those: an ideal gas at `p_abs = 0` is meaningless.
"""
multiphase_p_operating(fluid) = get(fluid.physics_properties, :p_operating, 0.0)

"""Optional `phase_change = Schrage(...)` / `Lee(...)` / `ModifiedEnergyJump(...)`."""
multiphase_phase_change(fluid) = get(fluid.physics_properties, :phase_change, nothing)

"""Optional `saturation = Antoine(...)`; defaults to the hydrogen fit."""
multiphase_saturation(fluid) = get(fluid.physics_properties, :saturation, Antoine())

"""Latent heat of vaporisation `h_fg` [J/kg]."""
multiphase_h_fg(fluid) = get(fluid.physics_properties, :h_fg, 0.0)


"""
    multiphase_rho_ref(fluid, phases, main)

Reference density for the `p_rgh` split, from the optional `rho_ref` keyword of
`Fluid{Multiphase}`. Defaults to the tracked (continuous) phase's density.

### Why a reference density rather than the local one

`p_rgh` splits the pressure into a dynamic part and a hydrostatic part. Which
density that hydrostatic part uses changes the numerics substantially:

    p_rgh = p - rho(x) g.h   ->  momentum source  -grad(p_rgh) - g.h grad(rho)
    p_rgh = p - rho_ref g.h  ->  momentum source  -grad(p_rgh) + (rho - rho_ref) g

Both are exact. But the first expresses buoyancy as a GRADIENT of density
multiplied by `g.h`, so it is divided by the cell size and scaled by the domain
height; the second is a body force proportional to the density excess and
involves no gradient at all.

On the LH2 pipe with wall boiling - `alpha` departing from 1 by only ~1e-3 across
a 34 um wall cell, `g.h` up to 3.04 - the two evaluate to:

    g.h grad(rho)      ~ 4283 N/m^3   (about 8x gravity itself)
    (rho - rho_ref) g  ~    0.47 N/m^3

a factor of ~9000. The first manufactures a large spurious force from a small
density gradient across a thin cell; the second does not. This is also what
section 5.4 of dev_notes_LH2_implementation_plan.md warned about - that the
buoyancy kernels are well balanced by construction only while `rho` is piecewise
constant.

Choose `rho_ref` as the phase that occupies most of the domain, so that
`(rho - rho_ref)` is near zero there: the liquid for a mostly-liquid pipe (the
default), the vapour for a large ullage.
"""
multiphase_rho_ref(fluid, phases, main) = get(fluid.physics_properties, :rho_ref, nothing)

"""Optional `wall_boiling = RPI(...)`, the wall nucleate boiling model."""
multiphase_wall_boiling(fluid) = get(fluid.physics_properties, :wall_boiling, nothing)

"""
    multiphase_interfacial_area(fluid, mp_model) -> AbstractInterfacialArea

Interfacial area closure for the BULK phase change source, from the optional
`interfacial_area` keyword of `Fluid{Multiphase}`.

The default is taken from the multiphase model, which is the only setting that
is self-consistent:

- `VOF`     -> [`ResolvedInterface`](@ref), `a_i = |grad(alpha)|`. The interface
               is tracked, so its area is where `alpha` changes.
- `Mixture` -> [`DispersedBubbles`](@ref) at the model's own `diameter`. Drift
               flux has no resolved interface; it has bubbles, and it has already
               committed to their size in the slip closure
               (`tau_d = rho_d d^2/(18 mu_c)`). Closing mass transfer with the
               same `d` keeps the two consistent and adds no free parameter.

Override only to reproduce the previous behaviour or to A/B the two closures -
`interfacial_area = ResolvedInterface()` on a `Mixture` restores the VOF form,
including its checkerboard feedback (see [`ResolvedInterface`](@ref)).
"""
function multiphase_interfacial_area(fluid, mp_model)
    area = get(fluid.physics_properties, :interfacial_area, nothing)
    area === nothing || return area
    return mp_model isa Mixture ?
        DispersedBubbles(diameter = mp_model.diameter) : ResolvedInterface()
end

"""
Under-relaxation factor for the BULK interfacial phase change rate, from the
optional `phase_change_relax` keyword of `Fluid{Multiphase}`. Defaults to 1
(no relaxation), which reproduces the unrelaxed behaviour exactly.
"""
multiphase_phase_change_relax(fluid) = get(fluid.physics_properties, :phase_change_relax, 1.0)

"""
Under-relaxation factor for `dp/dt`, the driver of the energy equation's
pressure-work source `S_T = beta*T*Dp/Dt`, from the optional
`pressure_work_relax` keyword of `Fluid{Multiphase}`.

**Defaults to 0.5, i.e. relaxation is ON.** Set to `1.0` to disable it.

Unlike the phase-change factors this is damped by default, because `Dp/Dt` is the
one source whose *raw* value is routinely dominated by numerics rather than
physics. A segregated pressure solve establishes a flow's pressure field in a
single step, whereas the physical transient it represents propagates
acoustically over many steps. For a heated pipe at `dt = 2e-6` s the first solve
creates ~618 Pa of frictional drop in one step, against an acoustic transit time
of ~7e-4 s - so the raw `Dp/Dt` is some two orders of magnitude too large. With
`beta*T ~ 1.6` even a 1 Pa adjustment per step gives `S_T ~ 8e5` W/m^3.

Because [`relax_source!`](@ref) blends rather than scales, the converged value is
unchanged: at steady state the relaxed and unrelaxed sources are identical, and a
slowly-varying `Dp/Dt` (a self-pressurising tank, say) is reproduced to within
`(1-relax)^n` after `n` steps - about 0.1% after ten steps at the default.

The precedent is direct: STAR-CCM+ blends its slip body force 50/50 between the
old and new value (Mixture Multiphase user guide, Eq. 2923) for the same reason.
"""
multiphase_pressure_work_relax(fluid) =
    get(fluid.physics_properties, :pressure_work_relax, 0.5)

"""
Under-relaxation factor for the WALL nucleate boiling rate, from the optional
`wall_boiling_relax` keyword of `Fluid{Multiphase}`. Defaults to 1.

Kept separate from `phase_change_relax` because the two sources are stiff for
different reasons and at different places: the bulk model responds to
`(T - T_sat)` throughout the interface region, while the wall model responds to
the wall superheat through `N_a ~ dT_sup^1.805`, which is far steeper. Damping
one usually does not call for damping the other by the same amount.
"""
multiphase_wall_boiling_relax(fluid) = get(fluid.physics_properties, :wall_boiling_relax, 1.0)

"""
Under-relaxation factor for the pressure equation's THERMAL EXPANSION source
`beta*DT/Dt`, from the optional `expansion_relax` keyword of
`Fluid{Multiphase}`. Defaults to 1 (no relaxation).

Applied to the thermal part only, before the phase-change volume creation is
summed in, so setting it to zero suppresses thermo-acoustic coupling **without**
removing the volume that boiling actually creates.

See [`multiphase_pressure_work_relax`](@ref) for why these two are usually set
together.
"""
multiphase_expansion_relax(fluid) = get(fluid.physics_properties, :expansion_relax, 1.0)

"""
    multiphase_dispersion_Sc(fluid) -> Float64 or nothing

Turbulent Schmidt number for TURBULENT DISPERSION of the volume fraction, from
the optional `dispersion_Sc` keyword of `Fluid{Multiphase}`. `nothing` (default)
switches the term off, preserving existing behaviour.

Adds `- div(D_t grad(alpha))` to the volume-fraction equation with
`D_t = nu_t/Sc_t`, the standard bubbly-flow closure (Lopez de Bertodano; Burns
et al.), present in both STAR-CCM+ and Fluent for Eulerian and mixture models.

### Why the alpha equation needs it

Without this term the volume-fraction equation is PURE ADVECTION,

    d(alpha)/dt + div(alpha*u) = S

with no Laplacian anywhere. Upwind supplies numerical diffusion only ALONG the
flow, so in a developed pipe - where the radial velocity is essentially zero -
there is nothing at all smoothing `alpha` in the wall-normal direction: no
physical diffusion, no numerical diffusion. Any radial flux noise imprints
directly onto `alpha` and stays there. Measured on the LH2 pipe: a radial
cell-to-cell oscillation of 0.36 (normalised) persisted with an upwind implicit
scheme, a smooth source, no drift flux and a clean pressure field, because
nothing in the equation could damp it.

That absence is also unphysical. Vapour generated at a heated wall has no
mechanism to move into the bulk except mean convection, when in reality
turbulent eddies disperse it down the concentration gradient. So this is missing
physics rather than added dissipation.

`Sc_t` around 0.9 is the usual choice; smaller disperses more strongly.
"""
function multiphase_dispersion_Sc(fluid)
    Sc = get(fluid.physics_properties, :dispersion_Sc, nothing)
    Sc === nothing && return nothing
    (Sc isa Real && Sc > 0) || throw(ArgumentError(
        "`dispersion_Sc` must be a positive turbulent Schmidt number, got $Sc"))
    return float(Sc)
end

"""
    multiphase_pressure_form(fluid) -> Symbol

Which conservation statement the pressure equation enforces, from the optional
`pressure_form` keyword of `Fluid{Multiphase}`.

Applies to compressible AND constant-density runs. A mixture of two
incompressible phases still has a varying density wherever `alpha` varies, so
`div(u) = 0` does not imply mass conservation and the two forms differ.

- `:volume` (default) — the low-Mach VOLUME constraint,

      psi*dp/dt - div(rDf grad p_rgh) = -div(u*) + expansion

  i.e. the pressure correction makes the VOLUMETRIC flux satisfy the dilatation
  budget. This is the historical XCALibre form and what OpenFOAM's
  `compressibleInterFoam` does.

- `:mass` — the MASS constraint `d(rho_m)/dt + div(rho_m u) = 0`, which is the
  same equation multiplied through by `rho_m`, with `u.grad(rho_m)` absorbed
  because `Drho_m/Dt + rho_m div(u) = 0` carries it:

      drho_m/dp*dp/dt - div(rho_f rDf grad p_rgh) = -div(rho_f u*) + rho_m*expansion

  This is the STAR-CCM+ mixture form. Every term picks up the density of the
  quantity it belongs to — that is the check that the conversion is right.

**Why it can matter.** Momentum and energy convect with `rhoPhi`, a MASS flux,
but under `:volume` nothing ever enforces `div(rho u) = -drho/dt`; the pressure
correction only constrains the volumetric flux. Where density varies by ~57x
(LH2/GH2) the two are far apart, and the discrete mass residual is then O(1) —
see `dev_notes_LH2_pipe_boiling.md`. `:mass` closes that gap directly and, as a
side effect, weights each cell's equation by its density, which conditions the
system better across an interface.

With constant density the two forms differ only by a uniform scale factor, so
they agree to solver tolerance. **Opt-in**, so existing cases are untouched.
"""
function multiphase_pressure_form(fluid)
    form = get(fluid.physics_properties, :pressure_form, :volume)
    form in (:volume, :mass) || throw(ArgumentError(
        "`pressure_form` must be :volume or :mass, got :$form"))
    return form
end

"""
    _validate_relax(name, value) -> Float64

Check an under-relaxation factor lies in `[0, 1]`.

Zero is allowed and means the source is switched **off**: `relax_source!` blends
against a `prev` field that starts at zero, so `relax = 0` leaves it at zero for
the whole run. That makes "disable this term" and "damp this term" one keyword
rather than two.
"""
function _validate_relax(name, value)
    (value isa Real && 0 <= value <= 1) || throw(ArgumentError(
        "`$name` must be in [0, 1], got $value. 1 means no relaxation, 0 disables the term."))
    return float(value)
end

"""
    relax_source!(field, prev, relax, config)

Under-relax a source term in TIME:

    field = (1 - relax)*prev + relax*field

then store the result in `prev` for the next step. `relax = 1` is a no-op beyond
the copy.

**Blending, not scaling.** Multiplying the rate by a factor would be the obvious
reading of "under-relaxation", but it is wrong here: it would permanently
evaporate less mass than the model asks for, biasing the vapour generation and
hence the mass balance. Blending against the previous step damps how fast the
source can *change* while leaving its converged value untouched — at steady state
`field == prev`, so the relaxed and unrelaxed answers coincide.

That distinction is what makes this safe to use as a stability aid rather than a
silent modification of the physics.
"""
function relax_source!(field, prev, relax, config)
    if relax >= 1
        @. prev.values = field.values
        return nothing
    end
    @. field.values = (1 - relax)*prev.values + relax*field.values
    @. prev.values = field.values
    return nothing
end

"""
Surface tension [N/m], from the optional `sigma` keyword of `Fluid{Multiphase}`.

Distinct from `VOF(sigma=...)`, which is the interface-capturing surface tension
force. This one is a *material property* read by the wall boiling sub-models that
need it, and it is therefore meaningful for the `Mixture` model too, which has no
surface tension force of its own.
"""
multiphase_sigma(fluid) = get(fluid.physics_properties, :sigma, 0.0)

"""Gravitational acceleration magnitude, for the bubble departure correlations."""
multiphase_g_magnitude(fluid) = norm(fluid.physics_properties.gravity.g)

"""
    validate_wall_boiling_setup(fluid, phases, wall_boiling)

Check that a wall boiling model has what it needs. Same intent as
[`validate_phase_change_setup`](@ref): fail at setup, naming what is missing,
rather than producing a silently zero source.
"""
validate_wall_boiling_setup(fluid, phases, ::Nothing) = nothing

function validate_wall_boiling_setup(fluid, phases, wb::AbstractWallBoilingModel)
    multiphase_h_fg(fluid) > 0 || throw(ArgumentError(
        """`wall_boiling` needs the latent heat. Pass it to the fluid, e.g.

    Fluid{Multiphase}(..., h_fg = 446.0e3)

The evaporative flux is converted to a vapour mass source by dividing by it, and
the energy equation removes that same latent heat again; zero makes both
meaningless."""))

    # Vapour created at the wall has to go somewhere, but it no longer needs a
    # compressible phase to get there: the net volume change travels through the
    # `expansion` source, which both branches of the pressure equation now carry.
    # Same reasoning as the bulk phase-change models above.

    # Two of the optional sub-models divide by, or take the square root of, the
    # surface tension. Zero is the default and is harmless for the models that
    # do not use it, so this only complains when it actually matters.
    needs_sigma = wb isa RPI &&
        (wb.site_density isa HibikiIshii ||
         wb.departure_diameter isa KocamustafaogullariIshii)

    if needs_sigma && multiphase_sigma(fluid) <= 0
        throw(ArgumentError(
            """The selected wall boiling sub-models need the surface tension, but `sigma` \
is $(multiphase_sigma(fluid)). Pass it to the fluid, e.g.

    Fluid{Multiphase}(..., sigma = 1.9e-3)

(`LemmertChawla` + `TolubinskyKostanchuk`, the defaults, do not need it.)"""))
    end
    return nothing
end

"""
    validate_phase_change_setup(fluid, phases, secondary)

Check that a phase change model has everything it needs, naming what is missing.
Returns the vapour specific gas constant (or `0` when the model does not use it).
"""
function validate_phase_change_setup(fluid, phases, secondary)
    pc = multiphase_phase_change(fluid)
    pc === nothing && return 0.0

    multiphase_h_fg(fluid) > 0 || throw(ArgumentError(
        """`phase_change` needs the latent heat. Pass it to the fluid, e.g.

    Fluid{Multiphase}(..., h_fg = 446.0e3)

Every model divides by or multiplies by L; zero makes the source meaningless."""))

    # A compressible phase is NO LONGER REQUIRED. The volume created by phase
    # change, mdot*(1/rho_v - 1/rho_l), is carried by the `expansion` source,
    # which both branches of the pressure equation now include. Two phases of
    # different density exchange volume when one becomes the other whether or not
    # either is compressible.
    #
    # Constant-density phase change is a genuinely useful configuration rather
    # than just a convenience: it is the only way to separate "alpha varies
    # across a density ratio" from "the vapour is compressible" when diagnosing a
    # two-phase pressure failure.

    # Schrage and Lee carry a kinetic-theory prefactor sqrt(1/(2 pi R_sp T_sat)).
    # `R_sp` is a property of the substance, not of the equation of state, so any
    # EOS that knows its own specific gas constant can supply it - `IdealGas`
    # from its defining constant, `TabulatedEos` from the molar mass of the fluid
    # it was tabulated from. This is what allows a real-fluid vapour to be used
    # with the Lee model, which the ideal-gas-only check previously forbade.
    if pc isa Union{Schrage,Lee}
        eos = phases[secondary].rho_model
        # Prefer the EOS's own value; fall back to one supplied on the model.
        # `R_sp` is a property of the SUBSTANCE, not of the equation of state, so
        # a constant-density vapour has a perfectly well-defined specific gas
        # constant even though `ConstEos` has nowhere to store it. Requiring a
        # compressible EOS just to obtain a physical constant would rule out the
        # constant-density control for no reason.
        R_sp = something(specific_gas_constant(eos), pc.R, Some(nothing))
        R_sp === nothing && throw(ArgumentError(
            """`$(typeof(pc).name.wrapper)` needs the vapour specific gas constant for its \
kinetic prefactor, but the vapour equation of state is $(typeof(eos).name.wrapper), \
which does not carry one.

Either give the vapour an EOS that does - `IdealGas(M=...)`, `IdealGas(R=...)`, or a \
tabulated `RealFluid(...)` - or pass the constant directly to the model:

    $(typeof(pc).name.wrapper)(..., R = 4124.5)     # [J/kg/K], = R_universal/M

(`ModifiedEnergyJump` does not need it and works with any vapour EOS.)"""))
        return R_sp
    end
    return 0.0
end

"""
    phase_density_ref(rho) -> scalar

A single representative density for a phase, used only to seed the initial
property blend. Exact for a `ConstantScalar`; the mean for a seeded field.
"""
phase_density_ref(rho::ConstantScalar) = rho.values
phase_density_ref(rho) = sum(rho.values)/length(rho.values)

"""
    seed_phase_properties!(model, p_operating, config)

Fill every variable property field of every phase from a uniform absolute
pressure `p_operating` and the current temperature field, before the first
property blend.

Without this a variable property starts at zero, which for density propagates an
`Inf` straight into the `mu/rho` blend, and for `cp` makes the energy equation's
time term singular on the first step.
"""
function seed_phase_properties!(model, p_operating, config)
    mesh = model.domain
    p_abs = ScalarField(mesh)
    initialise!(p_abs, p_operating)
    T = model.energy.T
    for phase in model.fluid.phases
        update_phase_properties!(phase, p_abs, T, config)
    end
    return nothing
end

"""
    update_phase_state!(model, p_abs, config)

Recompute the per-cell variable properties (density, viscosity, conductivity,
heat capacity, expansivity) of every phase at the current absolute pressure and
temperature. A no-op for properties held constant.
"""
function update_phase_state!(model, p_abs, config)
    T = model.energy.T
    for phase in model.fluid.phases
        update_phase_properties!(phase, p_abs, T, config)
    end
    return nothing
end

"""
    has_variable_properties(phases) -> Bool

True when any phase has a property that must be refreshed each step - i.e. any
model that is not one of the `Const*` family. Used to decide whether the
property-update pass is needed at all, so the all-constant case keeps exactly
the work it did before.
"""
function has_variable_properties(phases)
    is_const(m) = m === nothing || m isa Union{ConstEos,ConstMu,ConstK,ConstCp,ConstBeta}
    return any(phases) do ph
        !(is_const(ph.rho_model) && is_const(ph.mu_model) && is_const(ph.k_model) &&
          is_const(ph.cp_model) && is_const(ph.beta_model))
    end
end

"""
    absolute_pressure!(p_abs, p_rgh, rho, rho_ref, gh, p_operating, config)

`p_abs = p_rgh + rho*gh + p_operating`, the pressure a phase equation of state
must be evaluated at.
"""
absolute_pressure!(p_abs, p_rgh, rho, ::Nothing, gh, p_operating, config) =
    (@. p_abs.values = p_rgh.values + rho.values*gh.values + p_operating; nothing)

function absolute_pressure!(p_abs, p_rgh, rho, rho_ref, gh, p_operating, config)
    @. p_abs.values = p_rgh.values + rho_ref*gh.values + p_operating
    return nothing
end

"""
    multiphase_rD_ref_density(fluid) -> Float64 or nothing

Reference density for the momentum diagonal used by the PRESSURE equation, from
the optional `rD_ref_density` keyword of `Fluid{Multiphase}`. `nothing` (default)
uses the local mixture density, i.e. existing behaviour.

### The loop this breaks

`rD = 1/a_P` and `a_P` contains the mixture density, so

    alpha -> rho_m -> a_P -> rD -> rDf -> Laplacian coefficient -> p -> flux -> alpha

closes. A checkerboard in `alpha` becomes a checkerboard in `rho_m` BY
CONSTRUCTION (`rho_m` is linear in `alpha`), and once it reaches the Laplacian's
COEFFICIENT the checkerboarded pressure is the operator's genuine solution - not
a mode it failed to damp. That is why neither implicit alpha transport, a larger
time step, nor any Rhie-Chow damping removes it: all three act on the operator or
the time integration, while the mode enters through the coefficient. It is also
why temporal under-relaxation cannot help - a static checkerboard is a FIXED
POINT of the loop, and blending against the previous step converges to the same
fixed point.

Freezing the density in `rD` removes the `alpha` dependence and opens the loop.

### The approximation, and when it is small

`a_P = rho*V/dt + (convective and diffusive contributions)`. Where the transient
term dominates - the usual case at small `dt` - `rD ~ dt/(rho*V)`, so scaling by
`rho/rho_ref` recovers the reference-density diagonal to leading order. It is
EXACT only in that limit; with strong convection some `alpha` dependence remains.

The physical density is untouched everywhere it matters: buoyancy, `rhoPhi`,
momentum inertia and the energy equation all still use the true mixture value.
Only the pressure equation's coefficient is frozen - so this changes the PATH to
the solution, not the solution itself, in the same sense as a preconditioner.

Choose `rho_ref` near the working mixture density (the liquid value for a
bubbly flow at high `alpha`). At `alpha ~ 0.998` with LH2/GH2, `rho_m` spans
56.75 to 56.65 - a 0.2% variation - so the approximation is tight and the loop
is fully opened.
"""
function multiphase_rD_ref_density(fluid)
    rho_ref = get(fluid.physics_properties, :rD_ref_density, nothing)
    rho_ref === nothing && return nothing
    (rho_ref isa Real && rho_ref > 0) || throw(ArgumentError(
        "`rD_ref_density` must be a positive density [kg/m^3], got $rho_ref"))
    return float(rho_ref)
end

"""
    multiphase_p_abs_limit(fluid) -> Tuple or nothing

Bounds `(p_min, p_max)` for the ABSOLUTE pressure, from the optional
`p_abs_limit` keyword of `Fluid{Multiphase}`. `nothing` (default) leaves it
unbounded.

`p_abs` is what every property lookup and the saturation curve are evaluated at,
and those CLAMP silently at their table edges - a kernel cannot throw. So an
excursion in `p_abs` does not announce itself, it quietly returns edge values:
at 0.7 MPa operating with a curve starting at 0.25 MPa, a cell that dips below
gets `T_sat = 23.86 K` instead of 29.15 K, and saturated liquid then looks 5 K
superheated.

Bounding `p_abs` explicitly is preferable to letting the tables do it implicitly,
because it is visible, deliberate and REPORTED - see `clamp_absolute_pressure!`.

Note this is NOT `solvers.p_rgh.limit`, which the multiphase solver does not read.
"""
function multiphase_p_abs_limit(fluid)
    lim = get(fluid.physics_properties, :p_abs_limit, nothing)
    lim === nothing && return nothing
    (lim isa Tuple && length(lim) == 2 && lim[1] < lim[2]) || throw(ArgumentError(
        "`p_abs_limit` must be a (p_min, p_max) tuple with p_min < p_max, got $lim"))
    return (float(lim[1]), float(lim[2]))
end

"""
    clamp_absolute_pressure!(p_abs, limit, iteration) -> Int

Clamp `p_abs` into `limit`, returning how many cells were affected.

The count is the point. A silent clamp is what made the saturation-curve
excursion invisible for so long, so this one reports itself the first time it
bites and then every 100 iterations - enough to notice, not enough to spam.
"""
clamp_absolute_pressure!(p_abs, ::Nothing, iteration) = 0

function clamp_absolute_pressure!(p_abs, limit, iteration)
    lo, hi = limit
    n = count(v -> v < lo || v > hi, p_abs.values)
    n == 0 && return 0
    clamp!(p_abs.values, lo, hi)
    if n == 1 || iteration % 100 == 0
        @warn "Absolute pressure clamped" iteration cells=n limit=limit
    end
    return n
end

"""
    add_phase_change_volume!(expansion, mdot, rho_l, rho_v, config)

Add the net volume creation of phase change to the pressure-equation source:

    expansion += mdot*(1/rho_v - 1/rho_l)

Evaporating liquid into a much lighter vapour creates volume, which in a rigid
sealed tank raises the pressure. `1/rho_v >> 1/rho_l` for LH2/GH2 (about 58x at
20 K), so this term dominates the phase-change contribution to pressurisation.

`mdot === nothing` adds nothing.
"""
add_phase_change_volume!(expansion, ::Nothing, rho_l, rho_v, config) = nothing

function add_phase_change_volume!(expansion, mdot, rho_l, rho_v, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(expansion)
    kernel! = _add_phase_change_volume!(_setup(backend, workgroup, ndrange)...)
    kernel!(expansion, mdot, rho_l, rho_v)
    return nothing
end

@kernel inbounds=true function _add_phase_change_volume!(expansion, mdot, rho_l, rho_v)
    i = @index(Global)
    TF = eltype(expansion.values)
    expansion[i] += mdot[i]*(one(TF)/rho_v[i] - one(TF)/rho_l[i])
end

"""
    apply_phase_change_alpha!(alpha, mdot, rho_l, dt, config)

Apply the phase change sink to the volume fraction after the MULES update:

    alpha -= dt*mdot/rho_l

and clamp to `[0, 1]`.

**Known simplification.** Strictly this source should enter before the MULES
limiter computes its bounds, so that boundedness is guaranteed by construction
rather than restored by clamping. It is applied afterwards here because the
phase-change volume rate is minute for these cases (boil-off over hours, so
`dt*mdot/rho_l` is ~1e-12 per step) and the clamp is effectively never active.
That assumption breaks down for vigorous boiling, where the limiter would need
to account for the source properly.
"""
apply_phase_change_alpha!(alpha, ::Nothing, rho_l, dt, config) = nothing

function apply_phase_change_alpha!(alpha, mdot, rho_l, dt, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(alpha)
    kernel! = _apply_phase_change_alpha!(_setup(backend, workgroup, ndrange)...)
    kernel!(alpha, mdot, rho_l, dt)
    return nothing
end

@kernel inbounds=true function _apply_phase_change_alpha!(alpha, mdot, rho_l, dt)
    i = @index(Global)
    TF = eltype(alpha.values)
    a = alpha[i] - dt*mdot[i]/rho_l[i]
    alpha[i] = clamp(a, zero(TF), one(TF))
end

"""
    solve_pressure_compressible!(p_eqn, p_rgh, p_rgh_start, BCs, solversetup, config; ref, time)

Pressure solve for a compressible multiphase run, with the time term's reference
pressure held at its value from the START of the time step.

`solve_equation!` passes the solved field itself as `prev`, so inside the PISO
corrector loop the reference would move with every corrector and each one would
advance the pressure by another full `psi*dp/dt` increment — the pressure rise
would then scale with `inner_loops` and grow without bound instead of converging.
Freezing `prev` makes the correctors iterate towards a single fixed point for the
step, which is the intended PISO behaviour.

This matters only when the pressure equation carries a time term; the
incompressible path keeps using `solve_equation!` unchanged.
"""
function solve_pressure_compressible!(
    p_eqn, p_rgh, p_rgh_start, BCs, solversetup, config;
    ref=nothing, time=nothing, sealed=true)

    # `sealed` selects which field the Time term differences against.
    #
    # SEALED (no boundary fixes the pressure level): freeze at the step start, or
    # each PISO corrector adds another full psi*dp/dt increment and the pressure
    # rise scales with `inner_loops` instead of converging.
    #
    # FLOW-THROUGH (a Dirichlet on p_rgh pins the level): use the solved field,
    # as `solve_equation!` and CPISO both do. Here the outlet already fixes the
    # pressure, so freezing does NOT stabilise anything - it holds the reference
    # away from the value the boundary is imposing, and the difference between
    # them enters every corrector as a source that never relaxes. That is the
    # "stiff spurious source" the `p_ref` docstring in `property_tables.jl`
    # describes, and it is why `p_ref` locking (psi = 0) appeared to be the only
    # way to make such a case run.
    discretise!(p_eqn, sealed ? p_rgh_start : p_rgh, config)
    apply_boundary_conditions!(p_eqn, BCs, nothing, time, config)
    setReference!(p_eqn, ref, 1, config)
    update_preconditioner!(p_eqn.preconditioner, p_rgh.mesh, config)
    return solve_system!(p_eqn, solversetup, p_rgh, nothing, config)
end

"""
    update_expansion!(expansion, alpha, phases, T, T_prev, dt, config)

Thermal-expansion source of the pressure equation,

    expansion = sum_i alpha_i * beta_i * DT/Dt

with `beta_i` the thermal expansivity of each phase (exactly `1/T` for an ideal
gas). Reuses `phase_betaT`, dividing by `T` to recover `beta` itself.

This is the driver of self-pressurisation: heat raises `T`, the gas tries to
expand, and a rigid sealed volume converts that into a pressure rise. `DT/Dt` is
taken from the temperature solve just completed.
"""
function update_expansion!(expansion, alpha, phases, T, T_prev, dt, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Indexable rather than scalar, so a tabulated expansivity varies per cell.
    beta_l = _phase_beta_field(phases[1])
    beta_v = _phase_beta_field(phases[2])

    ndrange = length(expansion)
    kernel! = _update_expansion!(_setup(backend, workgroup, ndrange)...)
    kernel!(expansion, alpha, T, T_prev, dt,
            phases[1].rho_model, phases[2].rho_model, beta_l, beta_v)
    return nothing
end

@kernel inbounds=true function _update_expansion!(
    expansion, alpha, T, T_prev, dt, eos_l, eos_v, beta_l, beta_v)
    i = @index(Global)
    TF = eltype(expansion.values)
    a = alpha[i]
    t = T[i]
    betaT = a*phase_betaT(eos_l, beta_l[i], t) + (one(TF) - a)*phase_betaT(eos_v, beta_v[i], t)
    dTdt = (t - T_prev[i])/dt
    expansion[i] = betaT*dTdt/t
end

"""
    update_psi!(psi, alpha, phases, p_abs, T, config; mass_form=false)

Pressure-equation compressibility coefficient.

Volume form (`mass_form = false`):

    psi = sum_i alpha_i * (1/rho_i) * (d rho_i/dp)

Mass form (`mass_form = true`), the derivative of the MIXTURE density:

    drho_m/dp = sum_i alpha_i * (d rho_i/dp) = sum_i alpha_i * rho_i * psi_i

evaluated per cell, with `alpha_1 = alpha` (the tracked phase) and
`alpha_2 = 1 - alpha`. For an incompressible liquid plus an ideal-gas vapour the
volume form reduces to `psi = (1 - alpha)/p_abs`.

Note the mass coefficient is `sum_i alpha_i rho_i psi_i` and **not**
`rho_m * sum_i alpha_i psi_i`: the scale-by-`rho_m` reading of the mass form is
exact for the fluxes and sources, but the time term's coefficient is a genuine
derivative and each phase must be weighted by its own density. The two coincide
when one phase dominates and differ by ~2x at `alpha = 0.5` for LH2/GH2.
"""
function update_psi!(psi, alpha, phases, p_abs, T, config; mass_form=false)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(psi)
    kernel! = _update_psi!(_setup(backend, workgroup, ndrange)...)
    kernel!(psi, alpha, p_abs, T, phases[1].rho_model, phases[2].rho_model,
            phases[1].rho, phases[2].rho, Val(mass_form))
    return nothing
end

@kernel inbounds=true function _update_psi!(
    psi, alpha, p_abs, T, eos1, eos2, rho1, rho2, mass_form)
    i = @index(Global)
    TF = eltype(psi.values)
    a = alpha[i]
    p = p_abs[i]
    t = T[i]
    # `Val` so the branch is resolved at compile time and the kernel stays
    # GPU-safe (no runtime divergence, no boxed Bool).
    w1, w2 = _psi_weights(mass_form, rho1[i], rho2[i], TF)
    psi[i] = a*w1*phase_compressibility(eos1, p, t) +
             (one(TF) - a)*w2*phase_compressibility(eos2, p, t)
end

@inline _psi_weights(::Val{false}, r1, r2, ::Type{TF}) where {TF} = (one(TF), one(TF))
@inline _psi_weights(::Val{true}, r1, r2, ::Type{TF}) where {TF} = (r1, r2)

"""
    phase_property_faces!(prop_f, prop_cell, config)

Face values of a single phase's property. A `ConstantScalar` is index-independent
so the face field is simply filled; a cell `ScalarField` must be interpolated,
because indexing a cell-sized array by face ID would be wrong.

Used for density, viscosity, conductivity and heat capacity alike — every
property the face kernels read.
"""
phase_property_faces!(prop_f, prop_cell::ConstantScalar, config) =
    initialise!(prop_f, prop_cell.values)
phase_property_faces!(prop_f, prop_cell, config) =
    interpolate!(prop_f, prop_cell, config)

# Retained name for the density-specific call sites.
const phase_density_faces! = phase_property_faces!

"""
    PhaseFaceProperties(mesh, phases)

Per-phase FACE property fields (`rho1f`, `cp1f`, `k1f`, `mu1f` and the phase-2
counterparts), refreshed each step by `update_mixture_properties!`.

They exist because a variable property is stored as a CELL field, and the face
kernels — the mixture blends, the energy equation's `keff` and `rho_cp_phi` —
index by face ID. Interpolating once per step into these fields is both correct
and cheaper than interpolating at each use.

Constant properties are filled rather than interpolated, so an all-constant case
carries the same values it always did.
"""
struct PhaseFaceProperties{F}
    rho1f::F; rho2f::F
    cp1f::F;  cp2f::F
    k1f::F;   k2f::F
    mu1f::F;  mu2f::F
end
Adapt.@adapt_structure PhaseFaceProperties

PhaseFaceProperties(mesh) = PhaseFaceProperties(
    FaceScalarField(mesh), FaceScalarField(mesh),
    FaceScalarField(mesh), FaceScalarField(mesh),
    FaceScalarField(mesh), FaceScalarField(mesh),
    FaceScalarField(mesh), FaceScalarField(mesh))

"""
    update_phase_face_properties!(pf, phase_1, phase_2, config)

Refresh every per-phase face field from the corresponding cell field.

`cp` and `k` may be `nothing` when the energy model does not need them (an
isothermal run); those are skipped rather than defaulted, so a missing property
stays missing instead of silently becoming zero.
"""
function update_phase_face_properties!(pf::PhaseFaceProperties, phase_1, phase_2, config)
    phase_property_faces!(pf.rho1f, phase_1.rho, config)
    phase_property_faces!(pf.rho2f, phase_2.rho, config)

    phase_1.mu === nothing || phase_property_faces!(pf.mu1f, phase_1.mu, config)
    phase_2.mu === nothing || phase_property_faces!(pf.mu2f, phase_2.mu, config)

    phase_1.cp === nothing || phase_property_faces!(pf.cp1f, phase_1.cp, config)
    phase_2.cp === nothing || phase_property_faces!(pf.cp2f, phase_2.cp, config)

    phase_1.k === nothing || phase_property_faces!(pf.k1f, phase_1.k, config)
    phase_2.k === nothing || phase_property_faces!(pf.k2f, phase_2.k, config)
    return nothing
end

multiphase_extras(::VOF, mesh) = ()

function multiphase_extras(::Mixture, mesh)
    div_slip_momentum = VectorField(mesh)
    return (div_slip_momentum,)
end

function setup_multiphase_solvers(
    solver_variant, model, config;
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    )

    (; solvers, schemes, runtime, hardware, boundaries) = config

    @info "Extracting configuration and input fields..."

    (; U, p) = model.momentum
    (; alpha, alphaf, rho, rhof, nu, nuf, p_rgh, p_rghf) = model.fluid

    mp_model = model.fluid.model

    phases = model.fluid.phases
    volume_fraction = model.fluid.volume_fraction
    main = volume_fraction
    secondary = 3 - volume_fraction

    mesh = model.domain

    @info "Pre-allocating fields..."

    TF = _get_float(mesh)
    time = zero(TF)

    ∇p = Grad{schemes.p_rgh.gradient}(p)

    ∇p_rgh = Grad{schemes.p_rgh.gradient}(p_rgh)
    grad!(∇p_rgh, p_rghf, p_rgh, boundaries.p_rgh, time, config)
    limit_gradient!(schemes.p_rgh.limiter, ∇p_rgh, p_rgh, config)

    mdotf = FaceScalarField(mesh)
    rhoPhi = FaceScalarField(mesh)
    rDf = FaceScalarField(mesh)
    initialise!(rDf, 1.0)
    nueff = FaceScalarField(mesh)
    mueff = FaceScalarField(mesh)
    divHv = ScalarField(mesh)

    phi_g = VectorField(mesh)
    phi_gf = FaceScalarField(mesh)

    extra_models = multiphase_extras(mp_model, mesh)

    mules = (
        alpha_prev    = ScalarField(mesh),

        div_alpha     = ScalarField(mesh),
        div_mdotf     = ScalarField(mesh),

        alpha_fluxf   = FaceScalarField(mesh),
        alphaf_upwind = FaceScalarField(mesh),
        alphaf_HO     = FaceScalarField(mesh),

        phiLf         = FaceScalarField(mesh),
        phiHf         = FaceScalarField(mesh),
        phiAf         = FaceScalarField(mesh),

        Pplus         = ScalarField(mesh),
        Pminus        = ScalarField(mesh),

        Qplus         = ScalarField(mesh),
        Qminus        = ScalarField(mesh),

        Rplus         = ScalarField(mesh),
        Rminus        = ScalarField(mesh),

        alphaMaxLocal = ScalarField(mesh),
        alphaMinLocal = ScalarField(mesh),
    )

    @info "Computing fluid properties..."

    # Seed any variable property before the first blend: the fields are
    # allocated as zeros, which would otherwise propagate a zero density (and an
    # Inf in the mu/rho blend below) or a zero cp (a singular energy time term).
    #
    # Seeding is driven by `has_variable_properties`, not by compressibility
    # alone: a phase may have a constant density but a tabulated viscosity or
    # heat capacity, and those need seeding just as much.
    if has_variable_properties(phases) || is_compressible_multiphase(phases)
        p_op = multiphase_p_operating(model.fluid)
        p_op > 0 || throw(ArgumentError(
            """A phase with a variable equation of state needs an absolute pressure datum, \
but `p_operating` is $(p_op). Pass it to the fluid, e.g.

    Fluid{Multiphase}(..., p_operating = 103.0e3)

An ideal gas evaluated at zero absolute pressure has zero density."""))

        model.energy isa TwoPhaseTemperature || throw(ArgumentError(
            """A phase with a variable equation of state needs a temperature field, but the \
energy model is $(model.energy === nothing ? "Energy{Isothermal}" : typeof(model.energy).name.wrapper). \
Use `Energy{TwoPhaseTemperature}(Tref=...)`."""))

        seed_phase_properties!(model, p_op, config)
    end

    # Representative scalar properties for the INITIAL blend only; the first
    # solver iteration replaces these with per-cell/per-face values in
    # `update_mixture_properties!`. For a ConstantScalar this is exactly the old
    # `rho[1]`/`mu[1]`, so the incompressible path is unchanged.
    rho1_0 = phase_density_ref(phases[main].rho)
    rho2_0 = phase_density_ref(phases[secondary].rho)
    mu1_0 = phase_density_ref(phases[main].mu)
    mu2_0 = phase_density_ref(phases[secondary].mu)

    blend_properties!(rho, alpha, rho1_0, rho2_0)
    blend_properties!(rhof, alphaf, rho1_0, rho2_0)
    blend_properties!(nuf, alphaf, mu1_0/rho1_0, mu2_0/rho2_0)
    @. mueff.values = rhof.values * nueff.values

    gh = model.fluid.physics_properties.gravity.gh
    ghf = model.fluid.physics_properties.gravity.ghf
    g = model.fluid.physics_properties.gravity.g

    compute_gh!(gh, g, config)
    compute_ghf!(ghf, g, config)

    @info "Defining models..."

    if typeof(mp_model) <: VOF

        U_eqn = (
            Time{schemes.U.time}(rho, U)
            + Divergence{schemes.U.divergence}(rhoPhi, U)
            - Laplacian{schemes.U.laplacian}(mueff, U)
            ==
            - Source(∇p_rgh.result)
        ) → VectorEquation(U, boundaries.U)

    elseif typeof(mp_model) <: Mixture

        div_slip_momentum = extra_models[1]

        U_eqn = (
            Time{schemes.U.time}(rho, U)
            + Divergence{schemes.U.divergence}(rhoPhi, U)
            - Laplacian{schemes.U.laplacian}(mueff, U)
            ==
            - Source(∇p_rgh.result)
            - Source(div_slip_momentum)
        ) → VectorEquation(U, boundaries.U)

    end

    # Compressibility of the mixture. `psi` holds the pressure-equation
    # coefficient sum_i alpha_i * (1/rho_i)(d rho_i/dp); see `update_psi!`.
    #
    # The Time term is only added when a phase actually has a variable EOS, so
    # the incompressible path keeps exactly the equation (and the single-term
    # `make_symmetric!` treatment in `solve_equation!`) that it had before.
    psi = ScalarField(mesh)
    # Thermal-expansion driver: sum_i alpha_i * beta_i * DT/Dt. This is the term
    # that actually makes a sealed tank pressurise — heating the gas makes it
    # want to expand, the rigid volume forbids it, so the pressure rises.
    expansion = ScalarField(mesh)
    compressible = is_compressible_multiphase(phases)

    # Face flux of the compressibility coefficient, `psi_f * (u_f . S_f)`. This
    # is the coefficient of the IMPLICIT pressure-convection term that completes
    # the material derivative - see the p_eqn comment below. Built from the same
    # `psi` the Time term uses, so the two halves of Dp/Dt cannot disagree:
    # under `:volume` that is sum_i alpha_i psi_i, and under `:mass` it is
    # d(rho_m)/dp, and in each case `psi_f*(u.S)*p` carries exactly the units of
    # the flux the equation is written in.
    pconv = FaceScalarField(mesh)

    if compressible
        # sum_i alpha_i/rho_i * Drho_i/Dt balances div(u), giving
        #     psi * dp/dt - div(rDf grad(p_rgh)) = -div(Hv flux)
        # The Time term must be terms[1] (see `solve_equation!`), and its default
        # `rho_prev` is the term's own flux, so this discretises
        # psi*(p^n - p^{n-1})/dt, i.e. a frozen-psi d/dt rather than d(psi*p)/dt.
        #
        # dp/dt is approximated by dp_rgh/dt: the hydrostatic part rho*gh is
        # quasi-steady here. Same approximation as OpenFOAM's
        # compressibleInterFoam.
        # Full low-Mach constraint:
        #     div(u) + psi*Dp/Dt - sum_i alpha_i*beta_i*DT/Dt = 0
        # With u = Hv - rD*grad(p_rgh) this becomes
        #     psi*Dp/Dt - div(rDf grad p_rgh) = -div(Hv) + expansion
        #
        # NOTE THE MATERIAL DERIVATIVE. Dp/Dt = dp/dt + u.grad(p), and for a long
        # time only the dp/dt half was implemented - the `Divergence(pconv,...)`
        # term below is the other half. Dropping it does not merely lose accuracy:
        # it removes the IMPLICIT coupling between pressure and the flow that
        # carries it, so the acoustic mode is left to be resolved explicitly
        # through the PISO correctors. In a sealed tank that costs nothing (u ~ 0,
        # so u.grad(p) ~ 0, which is why every tank regression passed). In a
        # flow-through domain at 5.33 m/s with a compressible phase it is the
        # difference between an unconditionally stable pressure equation and one
        # that explodes as soon as the second phase appears - measured at 1e21 Pa
        # from 0.6% vapour.
        #
        # This mirrors what `Solvers_2_CPISO.jl` has always done for the
        # single-phase compressible solver, which carries exactly this term.
        # Omitting `expansion` leaves nothing to drive the pressure: heating the
        # ullage would then have no effect on tank pressure at all.
        #
        # The SAME assembled equation serves both the volume and the mass form
        # (see `multiphase_pressure_form`) — only what the solver loop writes
        # into the four coefficient/source arrays differs:
        #
        #   term        :volume            :mass
        #   Time flux   sum_i a_i psi_i    sum_i a_i rho_i psi_i  (= drho_m/dp)
        #   Lapl. flux  rDf                rho_f*rDf
        #   divHv       div(u*)            div(rho_f u*)
        #   expansion   [1/s]              rho_m*[1/s]            [kg/m3/s]
        p_eqn = (
            Time{schemes.p_rgh.time}(psi, p_rgh)
            - Laplacian{schemes.p.laplacian}(rDf, p_rgh)
            + Divergence{schemes.p_rgh.divergence}(pconv, p_rgh)
            ==
            - Source(divHv)
            + Source(expansion)
        ) → ScalarEquation(p_rgh, boundaries.p_rgh)
    else
        # `expansion` is carried here too, even though an incompressible mixture
        # has no psi and no thermal-expansion path. Phase change still creates
        # volume - mdot*(1/rho_v - 1/rho_l) is non-zero whenever the two phases
        # differ in density, compressible or not - and that volume has to go
        # somewhere. Without this source the constraint would be div(u) = 0,
        # which silently discards it.
        #
        # This also makes a CONSTANT-DENSITY two-phase flow with phase change a
        # usable configuration, which is the control that isolates "alpha varies
        # across a density ratio" from "the vapour is compressible" - the two
        # were previously impossible to separate.
        p_eqn = (
            - Laplacian{schemes.p.laplacian}(rDf, p_rgh)
            ==
            - Source(divHv)
            + Source(expansion)
        ) → ScalarEquation(p_rgh, boundaries.p_rgh)
    end

    @info "Initialising preconditioners..."

    @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    @reset p_eqn.preconditioner = set_preconditioner(solvers.p_rgh.preconditioner, p_eqn)

    @info "Pre-allocating solvers..."

    @reset U_eqn.solver = _workspace(solvers.U.solver, _b(U_eqn, XDir()))
    @reset p_eqn.solver = _workspace(solvers.p_rgh.solver, _b(p_eqn))

    @info "Initialising turbulence model..."
    turbulenceModel, config = initialise(model.turbulence, model, mdotf, p_eqn, config)

    @info "Initialising energy model..."
    energyModel = initialise_multiphase_energy(model.energy, model, mdotf, config)

    # A compressible run determines its own pressure level through the psi dp/dt
    # term, so pinning a reference cell would suppress exactly the
    # self-pressurisation being solved for.
    if compressible && pref !== nothing
        throw(ArgumentError(
            """`pref` must not be set for a compressible multiphase run: the compressibility \
term already fixes the pressure level, and pinning a reference cell would suppress the \
pressure rise. Pass `pref=nothing` and set `p_operating` on the fluid instead."""))
    end

    residuals = solver_variant(
        model, turbulenceModel, energyModel, ∇p, ∇p_rgh, U_eqn, p_eqn,
        mdotf, rhoPhi, gh, ghf, phi_g, phi_gf, extra_models, mules, config;
        output=output, pref=pref,
        ncorrectors=ncorrectors, inner_loops=inner_loops)

    return residuals
end

# `Energy{Isothermal}` builds to `nothing`, so that is the no-energy case.
initialise_multiphase_energy(::Nothing, model, mdotf, config) = nothing
initialise_multiphase_energy(energy::TwoPhaseTemperature, model, mdotf, config) =
    initialise(energy, model, mdotf, config)

multiphase_energy!(::Nothing, model, alpha_fluxf, mdotf, phase_faces, nueff,
                   dpdt, mdot_pc, L, time, dt, config) = nothing
multiphase_energy!(energyModel, model, alpha_fluxf, mdotf, phase_faces, nueff,
                   dpdt, mdot_pc, L, time, dt, config) =
    energy!(energyModel, model, alpha_fluxf, mdotf, phase_faces, nueff,
            dpdt, mdot_pc, L, time, dt, config)

energy_residuals(::Nothing) = ()
energy_residuals(energyModel) = (energyModel.state.residuals,)



function MULTIPHASE(
    model, turbulenceModel, energyModel, ∇p, ∇p_rgh, U_eqn, p_eqn,
    mdotf, rhoPhi, gh, ghf, phi_g, phi_gf,
    extra_models, mules, config;
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=3
    )

    (; alpha_prev, div_alpha, div_mdotf, alpha_fluxf,
       alphaf_upwind, alphaf_HO, phiLf, phiHf, phiAf,
       Pplus, Pminus, Qplus, Qminus, Rplus, Rminus,
       alphaMaxLocal, alphaMinLocal) = mules

    (; U, p) = model.momentum
    (; nu, nuf, rho, rhof, alpha, alphaf, p_rgh, p_rghf) = model.fluid
    (; solvers, schemes, runtime, hardware, boundaries, postprocess) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware

    mesh      = model.domain
    mp_model  = model.fluid.model
    phases    = model.fluid.phases
    main      = model.fluid.volume_fraction
    secondary = 3 - main

    TF      = _get_float(mesh)
    TI      = _get_int(mesh)
    n_cells = length(mesh.cells)

    @info "Allocating working memory..."

    dt_cpu = zeros(TF, 1)
    copyto!(dt_cpu, runtime.dt)
    postprocess = convert_time_to_iterations(postprocess, model, dt_cpu[1], iterations)

    mueff = get_flux(U_eqn, 3)

    # When any phase is compressible the pressure equation carries a leading
    # Time term (see `setup_multiphase_solvers`), which shifts the Laplacian
    # flux from index 1 to index 2.
    compressible = is_compressible_multiphase(phases)
    p_flux = get_flux(p_eqn, compressible ? 2 : 1)
    psi = compressible ? get_flux(p_eqn, 1) : nothing

    # Volume or mass form of the pressure equation - see `multiphase_pressure_form`.
    # NOT gated on `compressible`. That gate conflated "a phase has an equation
    # of state" with "the mixture density varies", which are different things: at
    # CONSTANT per-phase densities, rho_m = alpha*rho_l + (1-alpha)*rho_v still
    # varies wherever alpha does, so the volumetric and mass fluxes are not
    # proportional and mass conservation is not implied by div(u) = 0. For
    # LH2/GH2 at 0.7 MPa the ratio is 6.4, so the two fluxes differ by that much
    # across the interface.
    #
    # Phase change makes this concrete: it puts a volume source into the pressure
    # equation, which imposes div(u) != 0 while energy and momentum convect with
    # `rhoPhi`. The incompressible branch carries that source too (see the p_eqn
    # construction), so it needs the mass form for exactly the same reason the
    # compressible branch does.
    #
    # Nothing else here has to change: the Time term and `pconv` exist only on the
    # compressible branch, and both are correctly absent from the mass form when
    # d(rho_m)/dp = 0 - a mixture of incompressible phases has no compressibility
    # to carry, only a composition dependence, which travels through the flux and
    # the expansion source.
    mass_form = multiphase_pressure_form(model.fluid) === :mass

    # Does any boundary FIX the pressure level? A `Dirichlet` on p_rgh (an outlet,
    # typically) pins it; a domain with only zero-gradient and wall conditions is
    # sealed and the level is set by the compressibility term alone.
    #
    # This is the same distinction `pref` is validated against a few lines below,
    # and it selects the Time term's reference in
    # `solve_pressure_compressible!` - the two situations genuinely need
    # different treatment and neither choice is right for both.
    sealed_pressure = !any(bc -> bc isa Dirichlet, boundaries.p_rgh)
    if compressible
        @info "Compressible pressure level: " *
              (sealed_pressure ? "SEALED (set by the compressibility term)" :
                                 "FIXED by a Dirichlet boundary")
    end

    # In the VOLUME form the Laplacian coefficient IS `rDf`, so `rD` is
    # interpolated straight into the equation's flux array. In the MASS form the
    # coefficient is `rho_f*rDf`, so `rDf` needs storage of its own: the velocity
    # correction and the buoyancy flux both still want the plain `rDf`, because
    # the momentum equation is unchanged by how the pressure equation is scaled.
    rDf = mass_form ? FaceScalarField(mesh) : p_flux
    mass_form && initialise!(rDf, 1.0)

    # Flux of the implicit pressure-convection term, retrieved from the equation
    # itself (term 3, after Time and Laplacian) so the array the solver fills is
    # the same one the discretisation reads. `psif` is loop-local working storage.
    pconv = compressible ? get_flux(p_eqn, 3) : nothing
    psif = FaceScalarField(mesh)

    p_operating = multiphase_p_operating(model.fluid)
    # Reference density for the p_rgh split - see `multiphase_rho_ref`.
    rho_ref = multiphase_rho_ref(model.fluid, phases, main)
    p_abs_limit = multiphase_p_abs_limit(model.fluid)
    rD_ref_density = multiphase_rD_ref_density(model.fluid)
    g_vector = model.fluid.physics_properties.gravity.g
    p_abs = ScalarField(mesh)
    initialise!(p_abs, p_operating)

    # For the pressure-work source of the energy equation. `dpdt` stays `nothing`
    # for an all-incompressible mixture, which zeroes that source exactly.
    #
    # `p_abs_prev` MUST be seeded from the actual initial absolute pressure, not
    # from the uniform `p_operating` datum. `absolute_pressure!` includes the
    # hydrostatic term `rho*gh` (and any non-zero initial `p_rgh`), so seeding a
    # uniform value makes the FIRST step see
    #
    #     dp/dt = (rho*gh + p_rgh_0)/dt
    #
    # which is the entire hydrostatic head divided by one time step - a purely
    # numerical transient that nothing physical produced. It then enters the
    # energy equation as `S_T = beta*T*dp/dt` and, through the temperature it
    # creates, the pressure equation's `expansion` source.
    #
    # The resulting temperature error `beta*T*rho*gh/(rho*cp)` is independent of
    # `dt`, so it does not vanish by refining the time step. On a slow, sealed
    # tank it damps out; on a stiff through-flow case it does not.
    p_abs_prev = ScalarField(mesh)
    absolute_pressure!(p_abs, p_rgh, rho, rho_ref, gh, p_operating, config)
    clamp_absolute_pressure!(p_abs, p_abs_limit, 0)
    @. p_abs_prev.values = p_abs.values

    # Under-relaxation of dp/dt. ON by default - see
    # `multiphase_pressure_work_relax` for why this source in particular is
    # damped out of the box while the phase-change ones are not.
    relax_pressure_work = _validate_relax(:pressure_work_relax,
                                          multiphase_pressure_work_relax(model.fluid))
    dpdt_prev = compressible ? ScalarField(mesh) : nothing

    # Thermal-expansion source of the pressure equation. Together with the
    # pressure work above this forms an explicitly-evaluated THERMO-ACOUSTIC
    # loop, and neither term can be damped usefully on its own:
    #
    #   dT -> expansion = beta*dT/dt -> dp -> dp/dt -> S_T = beta*T*dp/dt -> dT
    #
    # Each traversal divides by dt twice, so treating it explicitly imposes an
    # acoustic CFL limit, dt < dx/c with c = 1/sqrt(rho*psi). For a mesh with
    # 34 um wall cells in liquid hydrogen (c ~ 420 m/s) that is dt < 8e-8 s.
    # A sealed tank never excites it - dp/dt there is order 1 Pa/s - but a
    # through-flow case with a startup pressure transient does.
    #
    # Both terms vanish at steady state, so setting either to zero costs nothing
    # for a steady-state case while removing the loop entirely.
    relax_expansion = _validate_relax(:expansion_relax,
                                      multiphase_expansion_relax(model.fluid))
    expansion_prev = compressible ? ScalarField(mesh) : nothing
    dpdt = compressible ? ScalarField(mesh) : nothing
    T_prev = ScalarField(mesh)
    expansion = get_source(p_eqn, 2)
    # Time-step-start pressure, held fixed across the PISO correctors so the
    # compressibility term advances once per step (see
    # `solve_pressure_compressible!`).
    p_rgh_start = ScalarField(mesh)

    # --- phase change -------------------------------------------------------
    phase_change = multiphase_phase_change(model.fluid)
    saturation   = multiphase_saturation(model.fluid)
    h_fg         = multiphase_h_fg(model.fluid)
    R_vapour     = validate_phase_change_setup(model.fluid, phases, secondary)

    # How the interfacial mass flux becomes a volumetric rate. Defaulted from the
    # multiphase model rather than fixed, because the two models have genuinely
    # different interfaces - see `multiphase_interfacial_area`.
    interfacial_area = multiphase_interfacial_area(model.fluid, mp_model)
    if phase_change !== nothing
        @info "Bulk phase change interfacial area" closure=typeof(interfacial_area).name.wrapper
    end

    # --- wall nucleate boiling ----------------------------------------------
    # A second, independent source of vapour: the bulk models above act on the
    # liquid/vapour interface, this one on the heated wall. Both write into the
    # same `mdot_pc`, so either can be used alone or the two together.
    wall_boiling = multiphase_wall_boiling(model.fluid)
    validate_wall_boiling_setup(model.fluid, phases, wall_boiling)
    wallBoiling = initialise_wall_boiling(wall_boiling, model, config)
    sigma_material = multiphase_sigma(model.fluid)
    g_magnitude = multiphase_g_magnitude(model.fluid)

    # Volumetric phase change rate [kg/m^3/s], positive for evaporation, and the
    # interfacial area density |grad(alpha)| it is built from. Allocated when
    # EITHER mechanism is active, since they share the field.
    any_phase_change = phase_change !== nothing || wallBoiling !== nothing
    mdot_pc = any_phase_change ? ScalarField(mesh) : nothing
    gradAlphaMag_pc = phase_change === nothing ? nothing : ScalarField(mesh)

    # Independent temporal under-relaxation of the two vapour sources. Both are
    # stiff, but for different reasons, so they get separate factors: the bulk
    # models respond to (T - T_sat) across the interface, the wall model to the
    # wall superheat through N_a ~ dT_sup^1.805, which is far steeper.
    #
    # `relax_source!` BLENDS against the previous step rather than scaling the
    # rate, so the converged answer is unchanged - see its docstring.
    relax_bulk = _validate_relax(:phase_change_relax,
                                 multiphase_phase_change_relax(model.fluid))
    relax_wall = _validate_relax(:wall_boiling_relax,
                                 multiphase_wall_boiling_relax(model.fluid))

    mdot_bulk_prev = phase_change === nothing ? nothing : ScalarField(mesh)
    mdot_wall_prev = wallBoiling === nothing ? nothing : ScalarField(mesh)

    if relax_bulk < 1 || relax_wall < 1
        @info "Phase change under-relaxation" bulk=relax_bulk wall=relax_wall
    end

    divHv = get_source(p_eqn, 1)
    nueff = FaceScalarField(mesh)

    outputWriter = initialise_writer(output, mesh)

    gradU  = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    Uf = FaceVectorField(mesh)
    S  = StrainRate(gradU, gradUT, U, Uf)

    # Aux fields for discretisation consistency
    ∇p_rghf_deconstructed = FaceScalarField(mesh)
    ∇p_rghf_reconstructed = VectorField(mesh)
    pressure_force_face   = FaceScalarField(mesh)

    # Viscosity may be constant or tabulated. What it may NOT be is a model the
    # solver has no way to refresh, since that would leave the field at its
    # seeded value for the whole run without any indication.
    for (i, phase) in enumerate(phases)
        phase.mu_model isa Union{ConstMu,TabulatedMu} || throw(ArgumentError(
            """Phase $i has an unsupported viscosity model \
($(typeof(phase.mu_model).name.wrapper)) for the multiphase solver. Use a constant \
(`mu = <value>`) or a tabulated model from `RealFluid(...)`."""))
    end

    variable_properties = has_variable_properties(phases)

    # Representative scalars, used only where a single number is genuinely
    # wanted (the drift-flux relaxation time below) and for the initial fill of
    # the face fields.
    rho1_val = phase_density_ref(phases[main].rho)
    rho2_val = phase_density_ref(phases[secondary].rho)
    mu1_val  = phase_density_ref(phases[main].mu)
    mu2_val  = phase_density_ref(phases[secondary].mu)

    # Per-phase face properties (density, viscosity, cp, k), refreshed each step
    # by `update_mixture_properties!` (a no-op refill for constant phases).
    phase_faces = PhaseFaceProperties(mesh)
    update_phase_face_properties!(phase_faces, phases[main], phases[secondary], config)

    ∇alpha  = Grad{schemes.alpha.gradient}(alpha)
    ∇alphaf = FaceVectorField(mesh)

    if typeof(mp_model) <: Mixture
        div_slip_momentum, = extra_models
        Sc_t     = 0.7
        C_alpha  = 0.0
        g_vec    = model.fluid.physics_properties.gravity.g
        diameter = mp_model.diameter
        tau_d    = (rho2_val * diameter^2) / (18.0 * mu1_val + eps())
        tau_d_field = ConstantScalar(tau_d)
        # Assumes constant values for now

        U_prev = VectorField(mesh)
        DUmDt  = VectorField(mesh)
        Ur     = VectorField(mesh)
        Urf    = FaceVectorField(mesh)
        # Face slip velocity for the MOMENTUM diffusion stress, built from the
        # force-balance drift velocity ALONE. Kept separate from `Urf` because
        # `Urf` may additionally carry turbulent dispersion, which must not reach
        # the momentum equation - see the note at `turbulent_dispersion!` below.
        Urf_slip = FaceVectorField(mesh)
        ∇U     = Grad{schemes.U.gradient}(U)

        # Implicit volume-fraction transport. See `build_alpha_equation` for why
        # the Mixture path leaves MULES behind and the VOF path does not.
        S_alpha     = ScalarField(mesh)
        drift_flux  = FaceScalarField(mesh)
        # Phase change rate carried over from the previous step, so the sink
        # enters the alpha EQUATION rather than being applied afterwards.
        mdot_lagged = ScalarField(mesh)
        implicit_alpha = implicit_alpha_transport(mp_model)

        # Turbulent dispersion coefficient D_t = nu_t/Sc_t on faces. The term is
        # ALWAYS assembled so the equation keeps a fixed shape; when dispersion
        # is off the field stays zero and the Laplacian contributes nothing.
        dispersion_Sc = multiphase_dispersion_Sc(model.fluid)
        Dtf = FaceScalarField(mesh)

        alpha_eqn   = implicit_alpha ?
            build_alpha_equation(model, mdotf, Dtf, S_alpha, config) : nothing

        # WHERE turbulent dispersion is applied. Two routes exist and they model
        # the SAME physics, so exactly one must be active:
        #
        #   Laplacian route  -div(D_t grad(alpha)) in the alpha equation. Needs
        #                    the implicit transport, and is the better of the two
        #                    - it is a diffusion term discretised as one.
        #   drift-flux route dispersion folded into `Ur` and carried by the drift
        #                    flux. The only option on the MULES path, and it
        #                    carries the hardcoded `Sc_t` above rather than the
        #                    user's `dispersion_Sc`.
        #
        # Running both double-counts dispersion (with two different Schmidt
        # numbers), which is what happened when the Laplacian was added alongside
        # the pre-existing drift-flux route.
        dispersion_in_Ur = !(implicit_alpha && dispersion_Sc !== nothing)
        if dispersion_Sc !== nothing
            @info "Turbulent dispersion of alpha" route=(dispersion_in_Ur ?
                "drift flux (Sc_t = $Sc_t)" : "Laplacian (Sc_t = $dispersion_Sc)")
        end
    end

    if typeof(mp_model) <: VOF
        sigma   = mp_model.sigma
        C_alpha = mp_model.cAlpha

        nhatf_prep     = FaceVectorField(mesh)
        kappa          = ScalarField(mesh)
        kappaf         = FaceScalarField(mesh)
        grad_alpha_mag = ScalarField(mesh)
    else
        sigma  = zero(TF)
        kappaf = ConstantScalar(zero(TF))
    end

    phirf    = FaceScalarField(mesh)
    Urdotf   = FaceScalarField(mesh)

    Hv       = VectorField(mesh)
    rD       = ScalarField(mesh)
    rho_prev = ScalarField(mesh)
    prev     = KernelAbstractions.zeros(backend, TF, n_cells)

    # Scratch for the boundary-face part of the least-squares reconstruction.
    # Allocated once: reconstruct! runs three times per time step.
    reconstruct_ws = ReconstructWorkspace(mesh, backend)

    R_ux    = ones(TF, iterations)
    R_uy    = ones(TF, iterations)
    R_uz    = ones(TF, iterations)
    R_p     = ones(TF, iterations)
    R_alpha = ones(TF, iterations)
    cellsCourant      = KernelAbstractions.zeros(backend, TF, n_cells)
    cellsAlphaCourant = KernelAbstractions.zeros(backend, TF, n_cells)

    time = zero(TF)
    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, boundaries.U, time, config)
    flux!(mdotf, Uf, config)
    @. rhoPhi.values = mdotf.values * rhof.values
    update_nueff!(nueff, nuf, model.turbulence, config)
    @. mueff.values  = rhof.values * nueff.values

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    @info "Starting multiphase solver..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations

        copyto!(dt_cpu, config.runtime.dt)
        time += dt_cpu[1]

        @. rho_prev.values = rho.values

        if typeof(mp_model) <: Mixture
            @. U_prev.x.values = U.x.values
            @. U_prev.y.values = U.y.values
            @. U_prev.z.values = U.z.values

            grad!(∇U, Uf, U, boundaries.U, time, config)
        end

        if typeof(mp_model) <: Mixture
            grad!(∇alpha, alphaf, alpha, boundaries.alpha, time, config)
            limit_gradient!(schemes.alpha.limiter, ∇alpha, alpha, config)

            compute_DUmDt!(DUmDt, U, U_prev, ∇U, dt_cpu[1], config)
            compute_Ur!(Ur, alpha, rho, g_vec, DUmDt,
                        phases[main].rho, phases[secondary].rho, phases[main].mu,
                        diameter, tau_d_field, config)

            # MOMENTUM slip stress uses the force-balance drift velocity only.
            # Snapshot it to faces BEFORE any dispersion is added.
            #
            # `div_slip_outer!` forms  alpha*(1-alpha)*(rho1+rho2)/rho_m * Ur (x) Ur,
            # which is derived from the MEAN slip between the phases. Turbulent
            # dispersion is not a mean slip - it closes the fluctuation
            # correlation <alpha' u'>, i.e. a DIFFUSIVE flux - so feeding it into
            # that stress and squaring it is a category error, and an expensive
            # one. Substituting the dispersion velocity D_t*grad(alpha)/(a(1-a))
            # into the stress gives
            #
            #     D_t^2 * |grad(alpha)|^2 / (alpha*(1 - alpha))
            #
            # which is QUADRATIC in the volume-fraction gradient - so a
            # cell-to-cell oscillation, the field that maximises that gradient,
            # is its preferred mode - and divided by alpha*(1 - alpha), which is
            # 0.002 in a nearly pure liquid: a 500x amplification exactly where
            # the flow is least two-phase. Measured on the LH2 pipe at
            # alpha_l = 0.998, a 1% oscillation across a 43 um wall cell
            # generates a spurious relative velocity of 0.33 m/s, LARGER than the
            # physical buoyant slip, and it then enters the momentum equation
            # squared.
            interpolate_vanleer!(Urf_slip, Ur, mdotf, config)
            zero_wall_drift_velocity!(Urf_slip, config)

            # VOLUME FRACTION drift flux. Turbulent dispersion belongs here and
            # only here - but only when the alpha equation is not ALREADY
            # carrying it as an explicit Laplacian, or it would be counted twice
            # (and with two different Schmidt numbers, the hardcoded `Sc_t` here
            # and the user's `dispersion_Sc`).
            dispersion_in_Ur || turbulent_dispersion!(
                Ur, alpha, ∇alpha, model.turbulence, Sc_t, config)

            interpolate_vanleer!(Urf, Ur, mdotf, config)
            zero_wall_drift_velocity!(Urf, config)
            face_dot_Sf!(Urdotf, Urf, config)
        end

        # Bounded alpha eqn. transport via MULES
        ralpha = zero(TF)
        if typeof(mp_model) <: Mixture && implicit_alpha
            # Turbulent dispersion coefficient for this step: D_t = nu_t/Sc_t on
            # faces. Left at zero when `dispersion_Sc` is unset, in which case the
            # Laplacian term is present but contributes nothing.
            if dispersion_Sc !== nothing
                interpolate!(Dtf, model.turbulence.nut, config)
                @. Dtf.values /= dispersion_Sc
            end
            @. alpha_prev.values = alpha.values
            ralpha = advance_alpha_implicit!(
                alpha_eqn, model, ∇alpha, ∇alphaf, mdotf, Urdotf,
                S_alpha, drift_flux, mdot_lagged, phases[main].rho,
                dt_cpu[1], time, config)
            # The limited alpha flux is still needed by the energy equation and
            # `blend_rhoPhi!`; rebuild it from the solved field so the two stay
            # consistent with the transport that actually happened.
            @. alpha_fluxf.values = mdotf.values*alphaf.values
        else
            advance_alpha!(model, mp_model, ∇alpha, ∇alphaf, mdotf,
                           alpha_prev, alphaf_upwind, alphaf_HO, phirf, Urdotf, phiLf, phiHf, phiAf,
                           alpha_fluxf, div_alpha, div_mdotf,
                           Pplus, Pminus, Qplus, Qminus, Rplus, Rminus,
                           alphaMaxLocal, alphaMinLocal, C_alpha, dt_cpu[1], time,
                           compressible, config)
        end

        # Vapour generation. Two independent mechanisms write into `mdot_pc`:
        #
        #   1. bulk interfacial phase change, evaluated from the alpha field just
        #      advanced, using |grad(alpha)| as the interfacial area density;
        #   2. wall nucleate boiling, evaluated on the heated wall faces.
        #
        # The combined rate then feeds three sources: the alpha sink here, the
        # volume creation in the pressure equation, and the latent heat in the
        # energy equation. Sharing one field is what lets either mechanism be
        # used alone, or both together, with no further branching downstream.
        if mdot_pc !== nothing
            absolute_pressure!(p_abs, p_rgh, rho, rho_ref, gh, p_operating, config)
            clamp_absolute_pressure!(p_abs, p_abs_limit, iteration)

            # Checked HERE, immediately before anything consults the saturation
            # curve. `saturation_temperature` clamps at the table edge because it
            # is a kernel function and cannot throw, so a pressure excursion
            # would otherwise pass silently into `T_sat` - and a wrong `T_sat`
            # makes saturated liquid look superheated, which an RPI site density
            # (~dT_sup^1.805) turns into an evaporation rate the wall cannot
            # supply. That failure is invisible from every field the solver
            # writes, so it has to be caught at the point of use.
            check_saturation_range(saturation, p_abs)

            # `phase_change_rate!` zeroes the field when the model is `nothing`,
            # which is exactly what the wall-boiling-only case needs.
            # `|grad(alpha)|` is only needed by `ResolvedInterface`; the
            # dispersed closure is algebraic in `alpha` alone. Guarding it makes
            # that independence explicit as well as saving the gradient pass.
            if phase_change !== nothing && interfacial_area isa ResolvedInterface
                cell_grad_magnitude!(gradAlphaMag_pc, ∇alpha, config)
            end
            phase_change_rate!(
                mdot_pc, phase_change, interfacial_area, alpha, gradAlphaMag_pc,
                model.energy.T, p_abs,
                phases[main].rho, phases[secondary].rho,
                saturation, h_fg, R_vapour, config)

            # Relax the BULK rate while `mdot_pc` still holds it alone, i.e.
            # before the wall contribution is summed in. Relaxing afterwards
            # would apply the bulk factor to both sources.
            mdot_bulk_prev === nothing ||
                relax_source!(mdot_pc, mdot_bulk_prev, relax_bulk, config)

            # Wall boiling can be held off until the base flow is established -
            # see `wall_boiling_active`. Before `start_iteration` the rate field
            # is ZEROED rather than left untouched, so no stale source survives
            # into the alpha equation, the pressure equation or the energy sink.
            wb_on = wallBoiling === nothing ? false :
                    wall_boiling_active(wallBoiling.model, iteration)
            if wb_on
                wall_boiling_source!(wallBoiling, model, p_abs, saturation, h_fg,
                                     g_magnitude, sigma_material, dt_cpu[1], config)
                relax_source!(wallBoiling.mdot_wall, mdot_wall_prev, relax_wall, config)
            elseif wallBoiling !== nothing
                fill!(wallBoiling.mdot_wall.values,
                      zero(eltype(wallBoiling.mdot_wall.values)))
            end
            add_wall_boiling_rate!(
                mdot_pc, wallBoiling === nothing ? nothing : wallBoiling.mdot_wall, config)

            # VOF keeps the after-the-fact application (with its documented
            # limitation). Mixture instead carries the rate forward so it enters
            # the alpha EQUATION on the next step - the whole point of the
            # implicit path, and what removes the MULES ordering constraint.
            if typeof(mp_model) <: Mixture && implicit_alpha
                @. mdot_lagged.values = mdot_pc.values
            else
                apply_phase_change_alpha!(alpha, mdot_pc, phases[main].rho,
                                          dt_cpu[1], config)
            end
            # alpha has moved, so refresh its face values before the blend below
            interpolate_vanleer!(alphaf, alpha, ∇alpha, mdotf, config)
            correct_boundaries!(alphaf, alpha, boundaries.alpha, time, config)
        end

        # Variable properties: refresh the absolute pressure first, then every
        # per-cell phase property, then the pressure-equation compressibility
        # coefficient. T lags by one step here (the energy equation runs below),
        # which is a first-order coupling consistent with the rest of the
        # segregated loop.
        if variable_properties && !compressible
            # A phase with constant density but tabulated cp/k/mu still needs its
            # properties refreshed; without a compressible phase there is no
            # `p_abs` update above to do it.
            absolute_pressure!(p_abs, p_rgh, rho, rho_ref, gh, p_operating, config)
            clamp_absolute_pressure!(p_abs, p_abs_limit, iteration)
            update_phase_state!(model, p_abs, config)
        end

        if compressible
            absolute_pressure!(p_abs, p_rgh, rho, rho_ref, gh, p_operating, config)
            clamp_absolute_pressure!(p_abs, p_abs_limit, iteration)
            # dp/dt over the PREVIOUS step, used for the energy equation's
            # pressure work. One term of the pressure/temperature coupling has to
            # lag in a segregated loop; the expansion driver below is the one kept
            # current, since it is what actually sets the pressurisation rate.
            @. dpdt.values = (p_abs.values - p_abs_prev.values)/dt_cpu[1]
            @. p_abs_prev.values = p_abs.values

            # Damp how fast dp/dt may CHANGE, without altering its converged
            # value (`relax_source!` blends against the previous step). A no-op
            # when `pressure_work_relax = 1.0`.
            relax_source!(dpdt, dpdt_prev, relax_pressure_work, config)
            @. T_prev.values = model.energy.T.values
            @. p_rgh_start.values = p_rgh.values

            update_phase_state!(model, p_abs, config)
            update_psi!(psi, alpha, phases, p_abs, model.energy.T, config;
                        mass_form=mass_form)
        end

        # Mixture property update from the new alpha
        update_mixture_properties!(model, alpha_fluxf, mdotf, rhoPhi, nueff, mueff,
                                   phase_faces, config)

        # Two-phase energy transport, BEFORE the pressure solve so the expansion
        # driver below sees this step's dT/dt. Reuses the limited `alpha_fluxf`
        # from advance_alpha! so energy and mass advection agree at the
        # interface. No-op when the energy model is Isothermal.
        multiphase_energy!(energyModel, model, alpha_fluxf, mdotf, phase_faces, nueff,
                           dpdt, mdot_pc, h_fg, time, dt_cpu[1], config)

        if compressible
            update_expansion!(expansion, alpha, phases, model.energy.T, T_prev,
                              dt_cpu[1], config)

            # Relax the THERMAL part only, while `expansion` still holds it
            # alone. Doing it after the phase-change volume below would damp the
            # vapour that boiling actually creates, which is not the intent -
            # that source is physical and must survive at any relaxation.
            relax_source!(expansion, expansion_prev, relax_expansion, config)
        else
            # No EOS, so no thermal-expansion term to compute - but `expansion`
            # is still the pressure equation's source and must be reset, since
            # `update_expansion!` (which normally overwrites it) did not run.
            @. expansion.values = 0
        end

        begin
            # Net volume created by phase change. OUTSIDE the compressibility
            # branch: two phases of different density exchange volume when one
            # becomes the other whether or not either is compressible, and the
            # incompressible pressure equation now carries this source too.
            #
            # Added after `update_expansion!` because that call overwrites the
            # field, and after `relax_source!` because this source is physical
            # and must survive at any relaxation setting.
            add_phase_change_volume!(expansion, mdot_pc,
                                     phases[main].rho, phases[secondary].rho, config)

            # MASS FORM: both sources are volume production rates [1/s]; the mass
            # equation wants mass production rates [kg/m3/s], so each scales by
            # the mixture density. For the phase-change term this is exactly the
            # STAR-CCM+ source,
            #
            #     b_cell = mdot_lv*(rho_m/rho_v - rho_m/rho_l)*V_cell
            #
            # Unlike the psi coefficient above, `rho_m` (not the per-phase
            # density) is correct here: these terms come from scaling the volume
            # constraint through by the mixture density, not from differentiating
            # rho_m. The volume created is shared by the mixture occupying the
            # cell, so it is the mixture's inertia that resists it.
            mass_form && @. expansion.values *= rho.values
        end

        # Interface curvature for surface tension (VOF only)
        if typeof(mp_model) <: VOF
            update_curvature!(model, ∇alpha, ∇alphaf, nhatf_prep, kappa, kappaf,
                              grad_alpha_mag, time, config)
        end
        
        # Drift flux divergence (Mixture only). `Urf_slip`, NOT `Urf`: the
        # momentum stress takes the force-balance slip alone.
        if typeof(mp_model) <: Mixture
            div_slip_outer!(div_slip_momentum, alphaf, rhof,
                            phase_faces.rho1f, phase_faces.rho2f, Urf_slip, config)
        end

        well_balanced_pressure_grad!(
            ∇p_rgh.result, pressure_force_face,
            p_rgh, rho, rhof, ghf, g_vector, rho_ref, mesh, config, reconstruct_ws;
            sigma=sigma, kappaf=kappaf, alpha=alpha)

        rx, ry, rz = solve_equation!(
            U_eqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config; rho_prev=rho_prev, time=time)

        inverse_diagonal!(rD, U_eqn, config)

        # Freeze the density in the pressure equation's diagonal - see
        # `multiphase_rD_ref_density`. To leading order a_P ~ rho*V/dt, so
        # scaling rD by rho/rho_ref gives the reference-density diagonal and
        # removes alpha from the Laplacian coefficient, opening the
        # alpha -> rho_m -> rD -> p -> flux -> alpha loop.
        if rD_ref_density !== nothing
            @. rD.values *= rho.values/rD_ref_density
        end

        interpolate!(rDf, rD, config)
        # Mass form: the pressure equation's Laplacian coefficient is rho_f*rDf.
        # `rhof` was refreshed by `update_mixture_properties!` above.
        mass_form && @. p_flux.values = rhof.values * rDf.values

        remove_pressure_source!(U_eqn, ∇p_rgh, config)

        rp = 0.0
        for i ∈ 1:inner_loops
            H!(Hv, U, U_eqn, config)

            interpolate!(Uf, Hv, config)
            correct_boundaries!(Uf, Hv, boundaries.U, time, config)

            flux!(mdotf, Uf, config)

            phi_gf!(phi_gf, rho, rhof, ghf, g_vector, rho_ref, rDf, model, config)

            if typeof(mp_model) <: VOF
                surface_tension_flux!(rDf, sigma, kappaf, alpha, phi_gf, config)
            end

            reconstruct!(phi_g, phi_gf, config, reconstruct_ws)

            @. mdotf.values += phi_gf.values

            # MASS FORM: switch `mdotf` from a volumetric to a mass flux for the
            # duration of the pressure solve, and switch it back afterwards.
            #
            # This scaling is what keeps the correction consistent. The
            # correction `correct_mass_flux_mp!` reads off the assembled matrix
            # is built from the Laplacian coefficient, which is now `rho_f*rDf` —
            # so it is a MASS flux correction. Adding it to a volumetric `mdotf`
            # would be wrong by a factor of ~57 for LH2/GH2. Scaling the field
            # instead of the correction also means the shared SIMPLE kernel needs
            # no change, and the boundary correction (which SETS rather than adds
            # on some patches) lands on the right quantity either way.
            #
            # Everything downstream of the unscaling — the alpha equation,
            # `alpha_fluxf`, the Courant numbers — still sees a volumetric flux.
            mass_form && @. mdotf.values *= rhof.values

            # IMPLICIT PRESSURE CONVECTION - the second half of psi*Dp/Dt.
            #
            # `Divergence(pconv, p_rgh)` in the equation carries div(psi*u*p)
            # implicitly, so the matching EXPLICIT part must come off the
            # right-hand side or the term is counted twice. Same split as
            # `Solvers_2_CPISO.jl`: subtract it here, add it back after the solve
            # at the NEW pressure, which is what makes the corrected flux
            # consistent with the equation that produced it.
            if compressible
                interpolate!(psif, psi, config)
                flux!(pconv, Uf, config)
                @. pconv.values *= psif.values
                interpolate!(p_rghf, p_rgh, config)
                correct_boundaries!(p_rghf, p_rgh, boundaries.p_rgh, time, config)
                @. mdotf.values -= pconv.values*p_rghf.values
            end

            div!(divHv, mdotf, config)

            @. prev = p_rgh.values
            rp = if compressible
                solve_pressure_compressible!(
                    p_eqn, p_rgh, p_rgh_start, boundaries.p_rgh, solvers.p_rgh,
                    config; ref=pref, time=time, sealed=sealed_pressure)
            else
                solve_equation!(p_eqn, p_rgh, boundaries.p_rgh, solvers.p_rgh,
                                config; ref=pref, time=time)
            end

            grad!(∇p_rgh, p_rghf, p_rgh, boundaries.p_rgh, time, config)
            limit_gradient!(schemes.p_rgh.limiter, ∇p_rgh, p_rgh, config)

            # Restore the explicitly-removed pressure convection, at the NEW
            # pressure (`p_rghf` was refreshed by `grad!` above). Before
            # `correct_mass_flux_mp!`, as CPISO does, so the matrix correction is
            # applied to the complete flux.
            compressible && @. mdotf.values += pconv.values*p_rghf.values

            correct_mass_flux_mp!(mdotf, p_eqn, config)

            # Back to a volumetric flux (see the scaling above).
            mass_form && @. mdotf.values /= rhof.values

            # `rDf` here is the plain 1/a_P interpolation in both forms: the
            # velocity correction is -rD*grad(p_rgh) regardless of how the
            # pressure equation itself was scaled.
            pressure_grad!(p_rgh, ∇p_rghf_deconstructed, phi_gf, rDf, config)
            reconstruct!(∇p_rghf_reconstructed, ∇p_rghf_deconstructed, config, reconstruct_ws)

            correct_velocity_rgh!(U, Hv, ∇p_rghf_reconstructed, rD, config)
        end

        # `p` carries the operating datum so it is the absolute pressure for a
        # compressible run, and unchanged (gauge) when p_operating is zero.
        @. p.values = p_rgh.values + ((rho_ref === nothing ? rho.values : rho_ref) .* gh.values) + p_operating

        turbulence!(turbulenceModel, model, S, prev, time, config)
        update_nueff!(nueff, nuf, model.turbulence, config)
        @. mueff.values = rhof.values * nueff.values

        courant      = max_courant_number!(cellsCourant, model, config)
        alphaCourant = max_alpha_courant_number!(cellsAlphaCourant, alpha, mdotf, model, config, dt_cpu[1])
        update_dt!(config.runtime, courant, alphaCourant)

        R_ux[iteration]    = rx
        R_uy[iteration]    = ry
        R_uz[iteration]    = rz
        R_p[iteration]     = rp
        R_alpha[iteration] = ralpha

        ProgressMeter.next!(
            progress, showvalues = [
                (:dt, dt_cpu[1]),
                (:time, time),
                (:Courant, courant),
                (:AlphaCourant, alphaCourant),
                (:Ux, R_ux[iteration]),
                (:Uy, R_uy[iteration]),
                (:Uz, R_uz[iteration]),
                (:p_rgh, R_p[iteration]),
                (:alpha, R_alpha[iteration]),
                turbulenceModel.state.residuals...,
                energy_residuals(energyModel)...
                ]
            )

        runtime_postprocessing!(postprocess,iteration,iterations,S,time,config)

        if iteration % write_interval + signbit(write_interval) == 0
            save_output(model, outputWriter, iteration, time, config)
            save_postprocessing(postprocess, iteration, time, mesh, outputWriter, config.boundaries)
            # Heated wall patches as a separate SURFACE file: the RPI partition
            # lives on boundary faces and has no cell-centred counterpart.
            # A no-op when wall boiling is not active.
            write_wall_boiling_surface(wallBoiling, mesh, iteration, time)
            # Area-averaged RPI partition to the log: the boiling curve in one
            # line, and `closure`/`evap_frac` flag a bad wall-temperature solve
            # without having to open the surface file.
            report_wall_boiling(wallBoiling, mesh; iteration=iteration)
        end
    end

    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p, alpha=R_alpha)
end





"""
    build_alpha_equation(model, mdotf, S_alpha, config)

Assemble the IMPLICIT volume-fraction transport equation used by the `Mixture`
model,

    d(alpha)/dt + div(alpha*u) = S_alpha

in conservative form, solved as a linear system rather than by the explicit
MULES flux-corrected update.

### Why implicit here, and only for `Mixture`

MULES exists to keep a **VOF interface** sharp, and the price it pays is that
the update form is fixed: the limiter guarantees boundedness only for the
advective rearrangement `d(alpha)/dt + u.grad(alpha)`, which is exact solely
when `div(u) = 0`. For a compressible mixture that is false — with wall boiling
`div(u)` reaches ~84 1/s — and the compressibility and phase-change terms cannot
be added without entering the limiter's bounds calculation first. Attempting the
correction outside the limiter makes the solution worse, not better (measured:
vapour mass residual rose from ~1x to 15-60x `mdot`).

A drift-flux mixture has no interface to keep sharp — `alpha` is smooth by
construction and `cAlpha` is already zero for this model — so nothing is lost by
dropping MULES here, and three things are gained:

  - every source term goes into the matrix, so there is no ordering constraint;
  - the alpha-Courant limit disappears, which is the dominant cost in a case
    needing O(1e5-1e6) steps;
  - a stiff source (wall boiling) is damped by implicit treatment rather than
    amplified.

The `VOF` path keeps MULES untouched, so the interface-capturing cases are
unaffected.

### Boundedness

Weaker than MULES, which is bounded by construction. With `Upwind` the
convection operator gives diagonal dominance `V/dt + sum(flux) = V/dt + div(u)*V`,
so the system is *more* dominant when `div(u) > 0` — the boiling case. It can
degrade under net compression, so the solution is clamped to `[0, 1]` afterwards
as a backstop and the clamp activity is worth monitoring.
"""
function build_alpha_equation(model, mdotf, Dtf, S_alpha, config)
    (; alpha) = model.fluid
    (; solvers, schemes, boundaries) = config

    hasproperty(solvers, :alpha) || throw(ArgumentError(
        """The `Mixture` multiphase model now solves an implicit volume-fraction \
equation and needs a solver for it, but `solvers` has no `alpha` entry.

Add one, e.g.

    alpha = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                        convergence=1e-7, relax=1.0, rtol=1e-2, atol=1e-10)

(Note this entry was previously accepted and silently ignored.)"""))

    TF = _get_float(model.domain)
    alpha_eqn = (
        Time{schemes.alpha.time}(ConstantScalar(one(TF)), alpha)
        + Divergence{schemes.alpha.divergence}(mdotf, alpha)
        - Laplacian{schemes.alpha.laplacian}(Dtf, alpha)
        ==
        Source(S_alpha)
    ) → ScalarEquation(alpha, boundaries.alpha)

    @reset alpha_eqn.preconditioner = set_preconditioner(solvers.alpha.preconditioner, alpha_eqn)
    @reset alpha_eqn.solver = _workspace(solvers.alpha.solver, _b(alpha_eqn))
    return alpha_eqn
end

"""
    advance_alpha_implicit!(alpha_eqn, model, ∇alpha, ∇alphaf, mdotf, Urdotf,
                            S_alpha, drift_flux, mdot_lagged, dt, time, config)

Advance the volume fraction by solving [`build_alpha_equation`](@ref).

The source carries everything the conservative liquid volume equation needs:

    S_alpha = -Gamma/rho_l  +  div(Urdotf * alpha*(1-alpha))

the first term being phase change, the second the drift flux written as a
deferred correction (it is non-linear in `alpha`, so it cannot be implicit).

`Gamma` is **lagged by one step**: the phase change rate is evaluated after this
call in the solver loop, because the bulk models need the freshly advected
`alpha` for their interfacial area. A one-step lag is consistent with the rest of
the segregated loop, and - the point of the exercise - the source now enters the
EQUATION rather than being applied to `alpha` after the fact.

The liquid compressibility term `-(alpha/rho_l) Drho_l/Dt` is **not** included.
It is identically zero for a constant-density liquid, which is the configuration
this path was built for; for a tabulated liquid it is a real omission and is
recorded as such rather than approximated.
"""
function advance_alpha_implicit!(alpha_eqn, model, ∇alpha, ∇alphaf, mdotf, Urdotf,
                                 S_alpha, drift_flux, mdot_lagged, rho_l, dt, time, config)
    (; alpha, alphaf) = model.fluid
    (; solvers, schemes, boundaries) = config
    mesh = model.domain

    grad!(∇alpha, alphaf, alpha, boundaries.alpha, time, config)
    limit_gradient!(schemes.alpha.limiter, ∇alpha, alpha, config)

    # Drift flux, explicit: Urdotf * alpha*(1-alpha), upwinded in alpha to match
    # what the MULES path used.
    build_drift_flux!(drift_flux, Urdotf, alpha, mdotf, boundaries, time, config)
    div!(S_alpha, drift_flux, config)

    # ...plus the (lagged) phase change sink. Sign: positive `mdot` is
    # evaporation, which destroys liquid.
    add_alpha_phase_change!(S_alpha, mdot_lagged, rho_l, config)

    discretise!(alpha_eqn, alpha, config)
    apply_boundary_conditions!(alpha_eqn, boundaries.alpha, nothing, time, config)
    implicit_relaxation_diagdom!(alpha_eqn, alpha.values, solvers.alpha.relax, nothing, config)
    update_preconditioner!(alpha_eqn.preconditioner, mesh, config)
    residual = solve_system!(alpha_eqn, solvers.alpha, alpha, nothing, config)

    # Backstop only - see the boundedness note in `build_alpha_equation`.
    clamp!(alpha.values, zero(eltype(alpha.values)), one(eltype(alpha.values)))

    interpolate_vanleer!(alphaf, alpha, ∇alpha, mdotf, config)
    correct_boundaries!(alphaf, alpha, boundaries.alpha, time, config)
    return residual
end

"""Face flux of the drift term, `Urdotf * alpha_up*(1 - alpha_up)`."""
function build_drift_flux!(drift_flux, Urdotf, alpha, mdotf, boundaries, time, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    alphaf_up = drift_flux    # reuse as scratch for the upwind interpolation
    interpolate_upwind!(alphaf_up, alpha, mdotf, config)
    correct_boundaries!(alphaf_up, alpha, boundaries.alpha, time, config)

    ndrange = length(drift_flux)
    kernel! = _drift_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(drift_flux, Urdotf)
    return nothing
end

@kernel inbounds=true function _drift_flux!(drift_flux, Urdotf)
    i = @index(Global)
    TF = eltype(drift_flux.values)
    af = drift_flux[i]                       # holds alphaf_upwind on entry
    drift_flux[i] = Urdotf[i]*af*(one(TF) - af)
end

add_alpha_phase_change!(S_alpha, ::Nothing, rho_l, config) = nothing

function add_alpha_phase_change!(S_alpha, mdot, rho_l, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    ndrange = length(S_alpha)
    kernel! = _add_alpha_phase_change!(_setup(backend, workgroup, ndrange)...)
    kernel!(S_alpha, mdot, rho_l)
    return nothing
end

@kernel inbounds=true function _add_alpha_phase_change!(S_alpha, mdot, rho_l)
    i = @index(Global)
    S_alpha[i] -= mdot[i]/rho_l[i]
end


function advance_alpha!(model, mp_model, ∇alpha, ∇alphaf, mdotf,
                        alpha_prev, alphaf_upwind, alphaf_HO, phirf, Urdotf, phiLf, phiHf, phiAf,
                        alpha_fluxf, div_alpha, div_mdotf,
                        Pplus, Pminus, Qplus, Qminus, Rplus, Rminus,
                        alphaMaxLocal, alphaMinLocal, C_alpha, dt, time, compressible, config)
    (; alpha, alphaf) = model.fluid
    (; schemes, boundaries) = config
    mesh = model.domain

    @. alpha_prev.values = alpha.values

    grad!(∇alpha, alphaf, alpha, boundaries.alpha, time, config)
    limit_gradient!(schemes.alpha.limiter, ∇alpha, alpha, config)

    alpha_compression_flux!(mp_model, phirf, ∇alphaf, ∇alpha, mdotf, C_alpha, config)

    interpolate_upwind!(alphaf_upwind, alpha, mdotf, config)
    correct_boundaries!(alphaf_upwind, alpha, boundaries.alpha, time, config)
    interpolate_vanleer!(alphaf_HO, alpha, ∇alpha, mdotf, config)
    correct_boundaries!(alphaf_HO, alpha, boundaries.alpha, time, config)

    @. phiLf.values = mdotf.values * alphaf_upwind.values

    high_order_alpha_flux!(mp_model, phiHf, mdotf, alphaf_HO, alphaf_upwind, phirf, Urdotf)

    @. phiAf.values = phiHf.values - phiLf.values

    zero_boundary_faces!(phiAf, config)

    mules_limit!(mp_model,
                 phiAf, alpha_prev, phiLf,
                 Pplus, Pminus, Qplus, Qminus, Rplus, Rminus,
                 alphaMaxLocal, alphaMinLocal,
                 dt, mesh, config)

    @. alpha_fluxf.values = phiLf.values + phiAf.values
    div!(div_alpha, alpha_fluxf, config)
    div!(div_mdotf, mdotf, config)

    # Volume fraction update.
    #
    # The exact liquid volume equation, from dividing liquid mass conservation
    # by rho_l, is CONSERVATIVE:
    #
    #     d(alpha)/dt + div(alpha*u) = -Gamma/rho_l - (alpha/rho_l) Drho_l/Dt
    #
    # Subtracting `alpha*div(u)` turns it into the advective form
    # `d(alpha)/dt + u.grad(alpha) = ...`, which is what the `- alpha*div_mdotf`
    # term below does. That rearrangement is exact ONLY when `div(u) = 0`, and it
    # is the standard MULES formulation because in an incompressible run the term
    # is inert and removes the effect of any discrete continuity error.
    #
    # For a COMPRESSIBLE mixture `div(u)` is not zero - it carries the
    # compressibility, thermal expansion and, above all, the volume created by
    # phase change. With wall boiling on the LH2 pipe it reaches ~84 1/s in the
    # near-wall cells, against a physical phase-change sink `Gamma/rho_l` of
    # ~32 1/s: the rearrangement term is several times LARGER than the source it
    # sits next to, and with the wrong sign for alpha.
    #
    # Measured consequence before this fix: the discrete vapour mass balance
    #
    #     d((1-alpha) rho_v)/dt + div((1-alpha) rho_v u) - mdot
    #
    # had a residual of ~100% of `mdot`. Because the energy equation's latent
    # heat sink `S_T = -mdot*h_fg` is only correct when that balance holds, the
    # error appeared as a spurious energy source of 2-7e8 W/m^3 against a wall
    # input of 8.8e8 W/m^3 - i.e. 23-80% of the applied heat flux.
    #
    # ATTEMPTED AND REVERTED: simply dropping the `- alpha*div_mdotf` term on the
    # compressible path (i.e. using the conservative form directly) makes things
    # markedly WORSE, not better:
    #
    #                       vapour mass residual   alpha_min   max|U|
    #     advective (this)   ~1.0 x mdot            0.997       6.2
    #     conservative       15-60 x mdot           0.933      16.7
    #
    # The reason is that MULES limits the antidiffusive flux to keep alpha
    # bounded ASSUMING this update form. Change the form and the limiter no
    # longer guarantees boundedness, so alpha overshoots and the balance gets
    # worse rather than better.
    #
    # The correct fix is therefore NOT a one-line change here. The missing terms
    #
    #     -(alpha/rho_l) Drho_l/Dt  -  alpha*div(u)
    #
    # have to enter BEFORE `mules_limit!` computes its bounds, so that
    # boundedness is guaranteed by construction - which is precisely the
    # restructure that `apply_phase_change_alpha!` already flags as necessary
    # for vigorous boiling. The two are the same piece of work and cannot be
    # done independently.
    @. alpha.values = alpha_prev.values -
        dt * (div_alpha.values - alpha_prev.values * div_mdotf.values)

    interpolate_vanleer!(alphaf, alpha, ∇alpha, mdotf, config)
    correct_boundaries!(alphaf, alpha, boundaries.alpha, time, config)
    return nothing
end

function alpha_compression_flux!(::VOF, phirf, ∇alphaf, ∇alpha, mdotf, C_alpha, config)
    interpolate!(∇alphaf, ∇alpha.result, config)
    compression_flux!(phirf, ∇alphaf, mdotf, C_alpha, config)
end
alpha_compression_flux!(::Mixture, phirf, ∇alphaf, ∇alpha, mdotf, C_alpha, config) = nothing

# This needs to be generalised as part of a more comprehensive high-order scheme implementation
high_order_alpha_flux!(::VOF, phiHf, mdotf, alphaf_HO, alphaf_upwind, phirf, Urdotf) =
    @. phiHf.values = mdotf.values * alphaf_HO.values +
                        phirf.values * alphaf_HO.values * (1.0 - alphaf_HO.values)

high_order_alpha_flux!(::Mixture, phiHf, mdotf, alphaf_HO, alphaf_upwind, phirf, Urdotf) =
    @. phiHf.values = mdotf.values * alphaf_HO.values -
                        Urdotf.values * alphaf_upwind.values * (1.0 - alphaf_upwind.values)


# This needs to be turned into a fused kernel for performance
#
# Phase densities are read per cell / per face rather than as scalar snapshots,
# so a variable equation of state works. `ConstantScalar` is index-independent,
# so for constant-density phases this produces exactly the same arithmetic as the
# previous `rho[1]` form.
function update_mixture_properties!(model, alpha_fluxf, mdotf, rhoPhi, nueff, mueff,
                                    phase_faces, config)
    (; rho, rhof, nu, nuf, alpha, alphaf, phases) = model.fluid
    main = model.fluid.volume_fraction
    secondary = 3 - main

    phase_1 = phases[main]
    phase_2 = phases[secondary]

    # Face values of every per-phase property. Required because a variable
    # property stores a CELL field, which must not be indexed by face ID.
    update_phase_face_properties!(phase_faces, phase_1, phase_2, config)

    blend_indexed!(rho,  alpha,  phase_1.rho,       phase_2.rho,       config)
    blend_indexed!(rhof, alphaf, phase_faces.rho1f, phase_faces.rho2f, config)

    # The dynamic viscosities are indexed rather than passed as scalars, so a
    # tabulated viscosity varies per cell and per face.
    blend_mixture_nu!(nu,  alpha,  rho,  phase_1.mu,       phase_2.mu,       config)
    blend_mixture_nu!(nuf, alphaf, rhof, phase_faces.mu1f, phase_faces.mu2f, config)

    update_nueff!(nueff, nuf, model.turbulence, config)
    @. mueff.values  = rhof.values * nueff.values

    blend_rhoPhi!(model.fluid.model, rhoPhi, alpha_fluxf, mdotf, rhof,
                  phase_faces.rho1f, phase_faces.rho2f)
    return nothing
end

"""
    blend_indexed!(out, frac, f1, f2, config)

`out[i] = frac[i]*f1[i] + (1 - frac[i])*f2[i]`, where `f1`/`f2` may be either
`ConstantScalar` (index-independent) or indexable fields. Serves both the
cell-centred and face-centred blends.
"""
function blend_indexed!(out, frac, f1, f2, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(out)
    kernel! = _blend_indexed!(_setup(backend, workgroup, ndrange)...)
    kernel!(out, frac, f1, f2)
    return nothing
end

@kernel inbounds=true function _blend_indexed!(out, frac, f1, f2)
    i = @index(Global)
    TF = eltype(out.values)
    a = frac[i]
    out[i] = a*f1[i] + (one(TF) - a)*f2[i]
end

blend_rhoPhi!(::Mixture, rhoPhi, alpha_fluxf, mdotf, rhof, rho1f, rho2f) =
    @. rhoPhi.values = mdotf.values * rhof.values

blend_rhoPhi!(::VOF, rhoPhi, alpha_fluxf, mdotf, rhof, rho1f, rho2f) =
    @. rhoPhi.values =
        alpha_fluxf.values * (rho1f.values - rho2f.values) + mdotf.values * rho2f.values



function update_curvature!(model, ∇alpha, ∇alphaf, nhatf_prep, kappa, kappaf,
                           grad_alpha_mag, time, config)
    (; alpha, alphaf) = model.fluid
    (; schemes, boundaries) = config
    grad!(∇alpha, alphaf, alpha, boundaries.alpha, time, config)
    limit_gradient!(schemes.alpha.limiter, ∇alpha, alpha, config)
    interpolate!(∇alphaf, ∇alpha.result, config)
    nhat_prep!(nhatf_prep, alpha, ∇alphaf, config)
    div!(kappa, nhatf_prep, config)
    cell_grad_magnitude!(grad_alpha_mag, ∇alpha, config)
    interpolate_weighted!(kappaf, kappa, grad_alpha_mag, config)
    return nothing
end


"""
    blend_properties!(property_field, alpha_field, property_0, property_1)

Linearly blends a per-phase scalar property using the volume-fraction field:

    property_field = property_0 * alpha + property_1 * (1 - alpha)
"""
function blend_properties!(property_field, alpha_field, property_0, property_1)
    @. property_field.values = (property_0 * alpha_field.values) + (property_1 * (1.0 - alpha_field.values))
    nothing
end

"""
    blend_mixture_nu!(nu_field, alpha_field, rho_field, mu_0, mu_1, config)

Mixture kinematic viscosity built from the per-phase dynamic viscosities:

    nu = (mu_0 * alpha + mu_1 * (1 - alpha)) / rho_blend

`mu_0`/`mu_1` are indexed, so they may be `ConstantScalar` (index-independent)
or fields — the cell/face distinction is the caller's responsibility, exactly as
for the density blend.
"""
function blend_mixture_nu!(nu_field, alpha_field, rho_field, mu_0, mu_1, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(nu_field)
    kernel! = _blend_mixture_nu!(_setup(backend, workgroup, ndrange)...)
    kernel!(nu_field, alpha_field, rho_field, mu_0, mu_1)
    return nothing
end

@kernel inbounds=true function _blend_mixture_nu!(nu_field, alpha_field, rho_field, mu_0, mu_1)
    i = @index(Global)
    TF = eltype(nu_field.values)
    a = alpha_field[i]
    nu_field[i] = (a*mu_0[i] + (one(TF) - a)*mu_1[i])/rho_field[i]
end


"""
    compute_gh!(gh, g, config)

Computes `g . x` at cell centres. Used for hydrostatic pressure reconstruction.
"""
function compute_gh!(gh, g, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    cells = gh.mesh.cells

    ndrange = length(gh)
    kernel! = _compute_gh!(_setup(backend, workgroup, ndrange)...)
    kernel!(gh, g, cells)
end
@kernel inbounds=true function _compute_gh!(gh, g, cells)
    i = @index(Global)
    (; centre) = cells[i]
    gh[i] = (g ⋅ centre)
end


"""
    compute_ghf!(ghf, g, config)

Computes `g . x` at face centres.
"""
function compute_ghf!(ghf, g, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    faces = ghf.mesh.faces

    ndrange = length(ghf)
    kernel! = _compute_ghf!(_setup(backend, workgroup, ndrange)...)
    kernel!(ghf, g, faces)
end
@kernel inbounds=true function _compute_ghf!(ghf, g, faces)
    i = @index(Global)
    (; centre) = faces[i]
    ghf[i] = (g ⋅ centre)
end



"""
    phi_gf!(phi_gf, rho, rhof, ghf, g_vector, rho_ref, rDf, model, config)

Builds the gravity (buoyancy) contribution to the face mass flux. On each
face computes

    phi_gf[f] = -ghf[f] · area · snGrad(rho) · rDf[f]

It is summed into `mdotf` before the pressure solve and reconstructed to
a cell vector (`phi_g`) for the velocity correction. A face with 
no density jump (single-phase region) contributes zero.
"""
function phi_gf!(phi_gf, rho, rhof, ghf, g, rho_ref, rDf, model, config)
    (; faces) = model.domain
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(faces)
    if rho_ref === nothing
        kernel! = _phi_gf_local!(_setup(backend, workgroup, ndrange)...)
        kernel!(phi_gf, rho, ghf, rDf, faces)
    else
        kernel! = _phi_gf!(_setup(backend, workgroup, ndrange)...)
        kernel!(phi_gf, rhof, g, rho_ref, rDf, faces)
    end
end

# Local-density buoyancy: -g.h snGrad(rho). Exactly well balanced across a SHARP
# interface, because snGrad(rho) is the same discrete difference as
# snGrad(p_rgh) and the two cancel term by term. That is why it remains the
# default and why the VOF hydrostatic cases hold to 1e-8 with it.
@kernel function _phi_gf_local!(phi_gf, rho, ghf, rDf, faces)
    fID = @index(Global)
    @inbounds begin
        (; area, ownerCells, delta) = faces[fID]
        face_grad = area*(rho[ownerCells[2]] - rho[ownerCells[1]])/delta
        phi_gf[fID] = -ghf[fID]*face_grad*rDf[fID]
    end
end
@kernel function _phi_gf!(phi_gf, rhof, g, rho_ref, rDf, faces)
    fID = @index(Global)
    @inbounds begin
        (; area, normal) = faces[fID]
        gn = g[1]*normal[1] + g[2]*normal[2] + g[3]*normal[3]
        # Buoyancy as a BODY FORCE proportional to the density excess over the
        # reference, not as a gradient of density. See `multiphase_rho_ref`.
        phi_gf[fID] = rDf[fID]*(rhof[fID] - rho_ref)*gn*area
    end
end

"""
    pressure_grad!(p_rgh, ∇p_rghf_deconstructed, phi_gf, rDf, config)

Builds the face-normal pressure-gradient field used to correct the cell
velocity (Rhie-Chow consistent). On each face computes

    ∇p_rghf_deconstructed[f] = (phi_gf[f] - snGrad(p_rgh)·area·rDf[f]) / (rDf[f])

The result is a face scalar later reconstructed to a cell vector (`reconstruct!`) 
and fed to `correct_velocity_rgh!`.
"""
function pressure_grad!(p_rgh, ∇p_rghf_deconstructed, phi_gf, rDf, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    faces = ∇p_rghf_deconstructed.mesh.faces

    ndrange = length(∇p_rghf_deconstructed)
    kernel! = _pressure_grad!(_setup(backend, workgroup, ndrange)...)
    kernel!(p_rgh, ∇p_rghf_deconstructed, phi_gf, rDf, faces)
end
@kernel function _pressure_grad!(p_rgh, ∇p_rghf_deconstructed, phi_gf, rDf, faces)
    i = @index(Global)
    face = faces[i]
    (; area, normal, ownerCells, delta) = face

    cID1 = ownerCells[1]
    cID2 = ownerCells[2]
    p1 = p_rgh[cID1]
    p2 = p_rgh[cID2]
    face_grad = area * (p2 - p1) / delta

    ∇p_rghf_deconstructed[i] = (phi_gf[i] - (face_grad * rDf[i])) / (rDf[i] + eps())
end

"""
    correct_velocity_rgh!(U, Hv, ∇p, rD, config)

Slightly modified version of `correct_velocity!` for convenience.
"""
function correct_velocity_rgh!(U, Hv, ∇p, rD, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(U)
    kernel! = _correct_velocity_rgh!(_setup(backend, workgroup, ndrange)...)
    kernel!(U, Hv, ∇p, rD)
end
@kernel function _correct_velocity_rgh!(U, Hv, ∇p, rD)
    i = @index(Global)

    @uniform begin
        Ux, Uy, Uz = U.x, U.y, U.z
        Hvx, Hvy, Hvz = Hv.x, Hv.y, Hv.z
        dpdx, dpdy, dpdz = ∇p.x, ∇p.y, ∇p.z
        rDvalues = rD.values
    end

    @inbounds begin
        rDvalues_i = rDvalues[i]
        Ux[i] = Hvx[i] + dpdx[i] * rDvalues_i
        Uy[i] = Hvy[i] + dpdy[i] * rDvalues_i
        Uz[i] = Hvz[i] + dpdz[i] * rDvalues_i
    end
end

"""
    ReconstructWorkspace(mesh, backend)

Per-cell scratch holding the boundary-face contributions to the least-squares
system solved by [`reconstruct!`](@ref): the six unique components of the
symmetric moment matrix and the three components of the right-hand side.

Allocated once at solver setup because `reconstruct!` runs several times per
time step.
"""
struct ReconstructWorkspace{V}
    m11::V; m12::V; m13::V; m22::V; m23::V; m33::V
    b1::V;  b2::V;  b3::V
end
Adapt.@adapt_structure ReconstructWorkspace

function ReconstructWorkspace(mesh, backend)
    TF = _get_float(mesh)
    n_cells = length(mesh.cells)
    mk() = KernelAbstractions.zeros(backend, TF, n_cells)
    ReconstructWorkspace(mk(), mk(), mk(), mk(), mk(), mk(), mk(), mk(), mk())
end

"""
    reconstruct!(phi::VectorField, psif::FaceScalarField, config, ws::ReconstructWorkspace)

Least-squares reconstruction of a cell-centred vector field `phi` from a
face-normal scalar field `psif`. Required for discretisation consistency.

Finds the cell vector `u` minimising `sum_f area_f * (u . n_f - psif_f/area_f)^2`,
i.e. solves `M u = b` with `M = sum_f area_f n_f n_f'` and `b = sum_f n_f psif_f`.

`mesh.cell_faces` contains **internal faces only**, so the boundary faces of a
cell are accumulated separately (via `mesh.boundary_cellsID`, with atomics
because a cell may own several boundary faces) into `ws`. Omitting them
under-determines the system: on a one-cell-thick wedge both wedge faces are
boundary faces, leaving `M` exactly rank 2 in every cell and the reconstruction
identically zero. The same omission merely degrades accuracy in ordinary
near-boundary cells.

Since `psif` is zero on boundary faces at every call site here (those kernels
form a surface-normal gradient, and a boundary face has `ownerCells == [c, c]`),
including them imposes `u . n = 0` on each boundary face - which is exactly the
axisymmetry constraint on the wedge planes.
"""
function reconstruct!(phi::VectorField, psif::FaceScalarField, config,
                      ws::ReconstructWorkspace)
    mesh = phi.mesh
    (; cells, cell_nsign, cell_faces, faces, boundary_cellsID) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    F = _get_float(mesh)
    is2D = typeof(mesh) <: Mesh2

    # Pass 1: boundary-face contributions, scattered into per-cell scratch.
    fill!(ws.m11, 0); fill!(ws.m12, 0); fill!(ws.m22, 0)
    fill!(ws.b1, 0);  fill!(ws.b2, 0)
    if !is2D
        fill!(ws.m13, 0); fill!(ws.m23, 0); fill!(ws.m33, 0); fill!(ws.b3, 0)
    end

    nbfaces = length(boundary_cellsID)
    if nbfaces > 0
        kernel! = _reconstruct_boundary_accum!(_setup(backend, workgroup, nbfaces)...)
        kernel!(faces, boundary_cellsID,  psif,
                ws.m11, ws.m12, ws.m13, ws.m22, ws.m23, ws.m33,
                ws.b1, ws.b2, ws.b3, Val(is2D))
        KernelAbstractions.synchronize(backend)
    end

    # Pass 2: internal faces, then solve per cell.
    ndrange = length(cells)
    if is2D
        kernel! = _reconstruct_operation_2D!(_setup(backend, workgroup, ndrange)...)
        kernel!(cells, F, cell_faces, cell_nsign, faces, phi, psif,
                ws.m11, ws.m12, ws.m22, ws.b1, ws.b2)
    else
        kernel! = _reconstruct_operation_3D!(_setup(backend, workgroup, ndrange)...)
        kernel!(cells, F, cell_faces, cell_nsign, faces, phi, psif,
                ws.m11, ws.m12, ws.m13, ws.m22, ws.m23, ws.m33,
                ws.b1, ws.b2, ws.b3)
    end
end

# Boundary faces occupy indices 1:nbfaces of `mesh.faces`, and a cell may own
# more than one of them (corner cells), hence the atomics - the same pattern
# used by `div_slip_outer_boundary_kernel!`.
@kernel inbounds=true function _reconstruct_boundary_accum!(
    faces, boundary_cellsID, psif, m11, m12, m13, m22, m23, m33, b1, b2, b3,
    ::Val{is2D}
) where {is2D}
    i = @index(Global)
    cID = boundary_cellsID[i]
    (; area, normal) = faces[i]

    nx = normal[1]; ny = normal[2]
    ssf = psif[i]

    Atomix.@atomic m11[cID] += area*nx*nx
    Atomix.@atomic m12[cID] += area*nx*ny
    Atomix.@atomic m22[cID] += area*ny*ny
    Atomix.@atomic b1[cID]  += nx*ssf
    Atomix.@atomic b2[cID]  += ny*ssf

    if !is2D
        nz = normal[3]
        Atomix.@atomic m13[cID] += area*nx*nz
        Atomix.@atomic m23[cID] += area*ny*nz
        Atomix.@atomic m33[cID] += area*nz*nz
        Atomix.@atomic b3[cID]  += nz*ssf
    end
end

@kernel function _reconstruct_operation_2D!(
    cells::AbstractArray{Cell{TF,SV,UR}}, F, cell_faces, cell_nsign, faces, phi, psif,
    bm11, bm12, bm22, bb1, bb2
) where {TF,SV,UR}
    i = @index(Global)
    @inbounds begin
        (; faces_range) = cells[i]

        # Seed from the boundary-face contributions accumulated in pass 1;
        # `cell_faces` below covers internal faces only.
        m11 = bm11[i]; m12 = bm12[i]; m22 = bm22[i]
        b1  = bb1[i];  b2  = bb2[i]

        for fi ∈ faces_range
            fID = cell_faces[fi]
            (; area, normal) = faces[fID]
            nx = normal[1]; ny = normal[2]

            m11 += area * nx * nx
            m12 += area * nx * ny
            m22 += area * ny * ny

            ssf = psif[fID]
            b1 += nx * ssf
            b2 += ny * ssf
        end

        det = m11*m22 - m12*m12

        # The moment matrix has units of area, so `det` scales as area^2 and an
        # absolute tolerance would reject well-conditioned cells purely because
        # they are small (silently zeroing the reconstructed vector). Normalise
        # by the matrix scale so the test measures conditioning, not cell size.
        scale = (m11 + m22)/2
        is_invertible = abs(det) > eps(TF)*scale*scale
        invdet = is_invertible ? one(TF)/det : zero(TF)

        ux = ( m22*b1 - m12*b2) * invdet
        uy = (-m12*b1 + m11*b2) * invdet

        phi[i] = @SVector [ux, uy, zero(TF)]
    end
end

@kernel function _reconstruct_operation_3D!(
    cells::AbstractArray{Cell{TF,SV,UR}}, F, cell_faces, cell_nsign, faces, phi, psif,
    bm11, bm12, bm13, bm22, bm23, bm33, bb1, bb2, bb3
) where {TF,SV,UR}
    i = @index(Global)
    @inbounds begin
        (; faces_range) = cells[i]

        # Seed from the boundary-face contributions accumulated in pass 1;
        # `cell_faces` below covers internal faces only.
        m11 = bm11[i]; m12 = bm12[i]; m13 = bm13[i]
                       m22 = bm22[i]; m23 = bm23[i]
                                      m33 = bm33[i]
        b1  = bb1[i];  b2  = bb2[i];  b3  = bb3[i]

        for fi ∈ faces_range
            fID = cell_faces[fi]
            (; area, normal) = faces[fID]
            nx = normal[1]; ny = normal[2]; nz = normal[3]

            m11 += area * nx * nx
            m12 += area * nx * ny
            m13 += area * nx * nz
            m22 += area * ny * ny
            m23 += area * ny * nz
            m33 += area * nz * nz

            ssf = psif[fID]
            b1 += nx * ssf
            b2 += ny * ssf
            b3 += nz * ssf
        end

        A11 = m22*m33 - m23*m23
        A12 = m13*m23 - m12*m33
        A13 = m12*m23 - m13*m22

        det = m11*A11 + m12*A12 + m13*A13

        # See the 2D kernel: `det` scales as area^3 here, so an absolute
        # tolerance silently zeroes the reconstruction on fine meshes (a
        # perfectly conditioned cell of side h has det = 8h^6, which drops below
        # eps(Float64) at h ~ 1e-3 m). Normalise by the matrix scale instead.
        scale = (m11 + m22 + m33)/3
        is_invertible = abs(det) > eps(TF)*scale*scale*scale
        invdet = is_invertible ? one(TF)/det : zero(TF)

        ux = (A11*b1 + A12*b2 + A13*b3) * invdet
        uy = (A12*b1 + (m11*m33 - m13*m13)*b2 + (m13*m12 - m11*m23)*b3) * invdet
        uz = (A13*b1 + (m13*m12 - m11*m23)*b2 + (m11*m22 - m12*m12)*b3) * invdet

        phi[i] = @SVector [ux, uy, uz]
    end
end


# Removes boundary-related problems, required for mass conservation in some cases.
function zero_boundary_faces!(phif::FaceScalarField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = phif.mesh
    nbfaces = length(mesh.boundary_cellsID)
    if nbfaces > 0
        ndrange = nbfaces
        kernel! = _mmp_zero_boundary_faces!(_setup(backend, workgroup, ndrange)...)
        kernel!(phif)
    end
end
@kernel inbounds=true function _mmp_zero_boundary_faces!(phif)
    i = @index(Global)
    phif[i] = zero(eltype(phif.values))
end


# Zero the drift velocity on boundary faces so no slip flux through walls
function zero_wall_drift_velocity!(Urf, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    nbfaces = length(Urf.mesh.boundary_cellsID)

    if nbfaces > 0
        ndrange = nbfaces
        kernel! = _zero_wall_drift_velocity!(_setup(backend, workgroup, ndrange)...)
        kernel!(Urf)
    end
end
@kernel inbounds=true function _zero_wall_drift_velocity!(Urf)
    i = @index(Global)
    TF = eltype(Urf.x)
    Urf.x[i] = zero(TF)
    Urf.y[i] = zero(TF)
    Urf.z[i] = zero(TF)
end


"""
    mules_limit!(mp_model::AbstractMultiphaseModel,
                 phiAf, alpha_prev, phiLf,
                 Pplus, Pminus, Qplus, Qminus, Rplus, Rminus,
                 alphaMaxLocal, alphaMinLocal,
                 dt, mesh, config)

Unified MULES (Zalesak FCT) flux-limiter for explicit alpha transport, shared by
both `VOF` and `Mixture` sub models. Limits the anti-diffusive
face flux `phiAf = phiHf - phiLf` so the explicit update
`α^{n+1} = α^n - dt/V · div(phiLf + phiAf)` stays bounded.

Logic:

  1. `mules_set_bounds!`: sets per-cell `alphaMax`, `alphaMin`
      Dispatched:
        - `VOF`     local neighbour extrema of `alpha_prev` between [0,1];
                    prevents isolated alpha=1 cells; preserves sharp interface.
        - `Mixture` hard `[0,1]` limit; must tolerate uniform alpha
                                without freezing the anti-diffusive drift flux.
  2. Build low-order α*, P+-, Q+-
  3. Ratios: R+- = clamp(Q+-/P+-, 0, 1).
  4. Per-face apply lambda_f (scales phiAf).

Boundary faces are left unlimited (lambda = 1).
"""
function mules_limit!(mp_model::AbstractMultiphaseModel,
                      phiAf, alpha_prev, phiLf,
                      Pplus, Pminus, Qplus, Qminus, Rplus, Rminus,
                      alphaMaxLocal, alphaMinLocal,
                      dt, mesh, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    (; cells, cell_nsign, cell_faces, faces) = mesh

    n_cells  = length(cells)
    n_faces  = length(faces)
    nbfaces  = length(mesh.boundary_cellsID)

    fill!(Pplus.values,  0)
    fill!(Pminus.values, 0)

    # Required to prevent isolated alpha=1 cells
    mules_set_bounds!(mp_model, alphaMaxLocal, alphaMinLocal, alpha_prev, mesh, config)

    # Numerical consistency
    ndrange = n_cells
    kernel! = _mmp_mules_cell_accum!(_setup(backend, workgroup, ndrange)...)
    kernel!(cells, cell_faces, cell_nsign, faces,
            alpha_prev, phiLf, phiAf,
            Pplus, Pminus, Qplus, Qminus,
            alphaMaxLocal, alphaMinLocal, dt)

    ndrange = n_cells
    kernel! = _mmp_mules_ratios!(_setup(backend, workgroup, ndrange)...)
    kernel!(Pplus, Pminus, Qplus, Qminus, Rplus, Rminus)

    # Core MULES operation (compute phi contribution bounded by lambda)
    ndrange = n_faces
    kernel! = _mmp_mules_apply!(_setup(backend, workgroup, ndrange)...)
    kernel!(phiAf, faces, Rplus, Rminus, nbfaces)
end

function mules_set_bounds!(::Mixture, alphaMaxLocal, alphaMinLocal, alpha_prev, mesh, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(mesh.cells)
    kernel! = _mmp_mules_hard_bounds!(_setup(backend, workgroup, ndrange)...)
    kernel!(alphaMaxLocal, alphaMinLocal)
end

function mules_set_bounds!(::VOF, alphaMaxLocal, alphaMinLocal, alpha_prev, mesh, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    (; cells, cell_faces, faces) = mesh

    ndrange = length(cells)
    kernel! = _mmp_mules_stencil_bounds!(_setup(backend, workgroup, ndrange)...)
    kernel!(cells, cell_faces, faces, alpha_prev, alphaMaxLocal, alphaMinLocal)
end

@kernel inbounds=true function _mmp_mules_hard_bounds!(alphaMaxLocal, alphaMinLocal)
    i = @index(Global)
    TF = eltype(alphaMaxLocal.values)
    alphaMaxLocal[i] = one(TF)
    alphaMinLocal[i] = zero(TF)
end

@kernel inbounds=true function _mmp_mules_stencil_bounds!(
    cells::AbstractArray{Cell{TF,SV,UR}}, cell_faces, faces,
    alpha_prev, alphaMaxLocal, alphaMinLocal
) where {TF,SV,UR}
    i = @index(Global)
    (; faces_range) = cells[i]

    aMax = alpha_prev[i]
    aMin = alpha_prev[i]

    for fi in faces_range
        fID = cell_faces[fi]
        oc = faces[fID].ownerCells

        j = ifelse(oc[1] == i, oc[2], oc[1])

        aj = alpha_prev[j]
        aMax = max(aMax, aj)
        aMin = min(aMin, aj)
    end

    alphaMaxLocal[i] = min(one(TF),  aMax)
    alphaMinLocal[i] = max(zero(TF), aMin)
end

@kernel inbounds=true function _mmp_mules_cell_accum!(
    cells::AbstractArray{Cell{TF,SV,UR}}, cell_faces, cell_nsign, faces,
    alpha_prev, phiLf, phiAf, Pplus, Pminus, Qplus, Qminus,
    alphaMaxLocal, alphaMinLocal, dt
) where {TF,SV,UR}
    i = @index(Global)
    (; volume, faces_range) = cells[i]

    sum_L = zero(TF)
    sum_A_pos = zero(TF)
    sum_A_neg = zero(TF)

    for fi in faces_range
        fID   = cell_faces[fi]
        nsign = cell_nsign[fi]

        fL = phiLf[fID] * nsign
        fA = phiAf[fID] * nsign
        sum_L += fL

        if fA < zero(TF)
            sum_A_pos += -fA
        else
            sum_A_neg += fA
        end
    end

    alpha_star = alpha_prev[i] - dt / volume * sum_L

    Pplus[i]  = sum_A_pos
    Pminus[i] = sum_A_neg

    Qplus[i]  = max(zero(TF), (alphaMaxLocal[i] - alpha_star)) * volume / dt
    Qminus[i] = max(zero(TF), (alpha_star - alphaMinLocal[i])) * volume / dt
end

@kernel inbounds=true function _mmp_mules_ratios!(Pplus, Pminus, Qplus, Qminus, Rplus, Rminus)
    i = @index(Global)
    TF = eltype(Pplus.values)

    Pp = Pplus[i]; Pm = Pminus[i]
    Qp = Qplus[i]; Qm = Qminus[i]

    Rplus[i]  = Pp > eps(TF) ? clamp(Qp / Pp, zero(TF), one(TF)) : one(TF)
    Rminus[i] = Pm > eps(TF) ? clamp(Qm / Pm, zero(TF), one(TF)) : one(TF)
end

@kernel inbounds=true function _mmp_mules_apply!(phiAf, faces, Rplus, Rminus, nbfaces)
    i = @index(Global)
    TF = eltype(phiAf.values)
    if i > nbfaces
        face = faces[i]
        (; ownerCells) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
    
        fA = phiAf[i]

        lambda = if fA > zero(TF)
            min(Rplus[cID2], Rminus[cID1])
        elseif fA < zero(TF)
            min(Rplus[cID1], Rminus[cID2])
        else
            one(TF)
        end

        phiAf[i] = lambda * fA
    end
end


"""
    compression_flux!(phirf, ∇alphaf, mdotf, C_alpha, config)

Anti-diffusive compression face flux:

    phir_f · Sf = min(Cα · |phi_f|/|Sf|, Phi_max) · (nnhatf · Sf)
"""
function compression_flux!(phirf, ∇alphaf, mdotf, C_alpha, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh  = phirf.mesh
    faces = mesh.faces
    TF = _get_float(mesh)

    # Phi max limits compression flux for numerical stability, compute it separately:
    phi_over_S_buf = similar(mdotf.values)
    ndrange = length(faces)
    kernel! = _fill_phi_over_S!(_setup(backend, workgroup, ndrange)...)
    kernel!(phi_over_S_buf, mdotf, faces)
    phimax = maximum(phi_over_S_buf)

    ndrange = length(faces)
    kernel! = _compression_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(phirf, ∇alphaf, mdotf, faces, TF(C_alpha), TF(phimax))
end

@kernel inbounds=true function _fill_phi_over_S!(buf, mdotf, faces)
    i = @index(Global)
    TF = eltype(buf)
    area = faces[i].area
    buf[i] = abs(mdotf[i]) / (area + eps(TF))
end

@kernel inbounds=true function _compression_flux!(phirf, ∇alphaf, mdotf, faces, C_alpha, phimax)
    i = @index(Global)
    face = faces[i]
    (; area, normal, delta) = face
    TF = eltype(phirf.values)

    Sf = area * normal
    grad_alpha     = ∇alphaf[i]
    grad_alpha_mag = norm(grad_alpha)

    noise_floor = TF(1.0e-8) / delta

    if grad_alpha_mag > noise_floor
        nhat = grad_alpha / grad_alpha_mag
    else
        nhat = zero(grad_alpha)
    end

    phi_over_S = abs(mdotf[i]) / (area + eps(TF))
    compr_speed = min(C_alpha * phi_over_S, phimax)

    phirf[i] = compr_speed * (nhat ⋅ Sf)
end

"""
    cell_grad_magnitude!(mag_field, grad, config)

Cell-centred `|∇alpha|` from a gradient field, used as the |∇alpha|-weighted face
interpolation weight for kappa_f so that near wall cells with small |∇alpha| don't
ruin the surface-tension force.
"""
function cell_grad_magnitude!(mag_field, grad, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = mag_field.mesh
    ndrange = length(mesh.cells)
    kernel! = _cell_grad_magnitude!(_setup(backend, workgroup, ndrange)...)
    kernel!(mag_field, grad.result)
end

@kernel inbounds=true function _cell_grad_magnitude!(mag_field, grad_result)
    i = @index(Global)
    mag_field.values[i] = norm(grad_result[i])
end

"""
    interpolate_weighted!(phif, phi, weight_field, config)

Weighted face interpolation: `phif[f] = (phi_c1·w_c1 + phi_c2·w_c2) /
(w_c1 + w_c2)`. With `w = |∇alpha|` it preserves kappa_f at interfaces and zeros
it in bulk where kappa is noisy.
"""
function interpolate_weighted!(phif::FaceScalarField, phi::ScalarField,
                                 weight_field::ScalarField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = phif.mesh
    (; faces) = mesh

    ndrange = length(faces)
    kernel! = _interpolate_weighted!(_setup(backend, workgroup, ndrange)...)
    kernel!(phif, phi, weight_field, faces)
end

@kernel inbounds=true function _interpolate_weighted!(phif, phi, w, faces)
    i = @index(Global)
    face = faces[i]
    (; ownerCells) = face
    c1 = ownerCells[1]
    c2 = ownerCells[2]
    TF = eltype(phif.values)

    w1 = w.values[c1]
    w2 = w.values[c2]
    denom = w1 + w2 + eps(TF)
    phif[i] = (phi.values[c1] * w1 + phi.values[c2] * w2) / denom
end

"""
    nhat_prep!(nhatf_prep, alpha, ∇alphaf, config)

Builds a face-normal unit vector field from the face-interpolated alpha grad.
"""
function nhat_prep!(nhatf_prep, alpha, ∇alphaf, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = alpha.mesh
    faces = mesh.faces
    nbfaces = length(mesh.boundary_cellsID)
    nfaces  = length(faces)

    if nbfaces > 0
        kernel! = _nhat_zero_bfaces!(_setup(backend, workgroup, nbfaces)...)
        kernel!(nhatf_prep)
    end

    ninternal = nfaces - nbfaces
    if ninternal > 0
        kernel! = _nhat_normalise_ifaces!(_setup(backend, workgroup, ninternal)...)
        kernel!(nhatf_prep, faces, ∇alphaf, nbfaces)
    end
end

@kernel inbounds=true function _nhat_zero_bfaces!(nhatf_prep)
    i = @index(Global)
    nhatf_prep[i] = SVector(0.0, 0.0, 0.0)
end

@kernel inbounds=true function _nhat_normalise_ifaces!(nhatf_prep, faces, ∇alphaf_, nbfaces)
    i = @index(Global)
    fID = i + nbfaces
    face = faces[fID]
    (; delta) = face

    grad_alpha     = ∇alphaf_[fID]
    grad_alpha_mag = norm(grad_alpha)

    noise_floor = 1e-8 / delta
    if grad_alpha_mag < noise_floor
        nhatf_prep[fID] = SVector(0.0, 0.0, 0.0)
    else
        nhatf_prep[fID] = grad_alpha / grad_alpha_mag
    end
end

"""
    surface_tension_flux!(rDf, sigma, kappaf, alpha, phi_gf, config)

CSF surface-tension contribution to the face flux:
`phi_gf -= σ · κf · (∇αf · Sf) · rDf`.
"""
function surface_tension_flux!(rDf, sigma, kappaf, alpha, phi_gf, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    faces = phi_gf.mesh.faces

    ndrange = length(phi_gf)
    kernel! = _surface_tension_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(rDf, sigma, kappaf, alpha, phi_gf, faces)
end

@kernel inbounds=true function _surface_tension_flux!(rDf, sigma, kappaf, alpha, phi_gf, faces)
    i = @index(Global)
    face = faces[i]
    (; area, normal, ownerCells, delta) = face
    Sf = area * normal

    cID1 = ownerCells[1]
    cID2 = ownerCells[2]
    alpha1 = alpha[cID1]
    alpha2 = alpha[cID2]

    ∇alphaf_vec = normal * ((alpha2 - alpha1) / delta)

    phi_gf[i] -= sigma * kappaf[i] * (∇alphaf_vec ⋅ Sf) * rDf[i]
end

"""
    well_balanced_pressure_grad!(grad_field, face_buf, p_rgh, rho, ghf,
                                  mesh, config; sigma=0, kappaf=nothing,
                                  alpha=nothing)

Overrides `grad_field.values` with a face-snGrad reconstruction of the predictor body force terms:

    face_buf[f] = area_f · ( snGrad(p_rgh) + ghf · snGrad(rho) - sigma·kappaf · snGrad(alpha) )

*Important for stability.
"""
function well_balanced_pressure_grad!(
    grad_field, face_buf, p_rgh, rho, rhof, ghf, g, rho_ref, mesh, config, reconstruct_ws;
    sigma=zero(eltype(p_rgh.values)),
    kappaf, alpha,
)
    (; hardware) = config
    (; backend, workgroup) = hardware
    faces = mesh.faces

    ndrange = length(faces)
    if rho_ref === nothing
        kernel! = _well_balanced_pressure_face_local!(_setup(backend, workgroup, ndrange)...)
        kernel!(face_buf, p_rgh, rho, alpha, ghf, kappaf, sigma, faces)
    else
        kernel! = _well_balanced_pressure_face!(_setup(backend, workgroup, ndrange)...)
        kernel!(face_buf, p_rgh, rhof, alpha, g, rho_ref, kappaf, sigma, faces)
    end

    reconstruct!(grad_field, face_buf, config, reconstruct_ws)
end

@kernel inbounds=true function _well_balanced_pressure_face_local!(
    face_buf, p_rgh, rho, alpha, ghf, kappaf, sigma, faces
)
    i = @index(Global)
    (; area, ownerCells, delta) = faces[i]
    c1 = ownerCells[1]; c2 = ownerCells[2]
    snGrad_p   = (p_rgh[c2] - p_rgh[c1]) / delta
    snGrad_rho = (rho[c2]   - rho[c1])   / delta
    snGrad_a   = (alpha[c2] - alpha[c1]) / delta
    face_buf[i] = area * (snGrad_p + ghf[i] * snGrad_rho - sigma * kappaf[i] * snGrad_a)
end

@kernel inbounds=true function _well_balanced_pressure_face!(
    face_buf, p_rgh, rhof, alpha, g, rho_ref, kappaf, sigma, faces
)
    i = @index(Global)
    face = faces[i]
    (; area, normal, ownerCells, delta) = face
    c1 = ownerCells[1]
    c2 = ownerCells[2]

    snGrad_p = (p_rgh[c2] - p_rgh[c1]) / delta
    snGrad_a = (alpha[c2] - alpha[c1]) / delta
    gn = g[1]*normal[1] + g[2]*normal[2] + g[3]*normal[3]

    # Momentum source is -(grad p_rgh) + (rho - rho_ref) g + surface tension.
    # The buoyancy term no longer involves snGrad(rho), which is what removed
    # the amplification by gh/delta.
    face_buf[i] = area * (snGrad_p - (rhof[i] - rho_ref)*gn - sigma*kappaf[i]*snGrad_a)
end


function correct_mass_flux_mp!(mdotf, p_eqn, config; time=nothing)
    # sngrad = FaceScalarField(mesh)
    (; faces, cells, boundary_cellsID) = mdotf.mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    p = p_eqn.model.terms[1].phi
    A = _A(p_eqn)
    nzval = _nzval(A)
    colval = _colval(A)
    rowptr = _rowptr(A)

    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    ndrange = n_ifaces # length(n_ifaces) was a BUG! should be n_ifaces only!!!!
    kernel! = _correct_mass_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(mdotf, p, nzval, colval, rowptr, faces, cells, n_bfaces)
    KernelAbstractions.synchronize(backend)

    BCs = config.boundaries.p_rgh # this line had to be changed from ".p"
    for BC ∈ BCs
        correct_mass_periodic(
            BC, mdotf, p, nzval, colval, rowptr, cells, faces, backend, workgroup)
        KernelAbstractions.synchronize(backend)
    end

    correct_boundary_mass_flux!(mdotf, p_eqn, BCs, time, config)
end




function compute_DUmDt!(DUmDt, U, U_prev, gradU, dt, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(DUmDt)
    kernel! = _compute_DUmDt!(_setup(backend, workgroup, ndrange)...)
    kernel!(DUmDt, U, U_prev, gradU.result, dt)
end

@kernel inbounds=true function _compute_DUmDt!(DUmDt, U, U_prev, gradU_result, dt)
    i = @index(Global)
    TF = eltype(U.x)

    dUdt = (U[i] - U_prev[i]) / dt

    ux = U.x[i]
    uy = U.y[i]
    uz = U.z[i]

    dudx = gradU_result.xx[i]
    dudy = gradU_result.xy[i]
    dudz = gradU_result.xz[i]

    dvdx = gradU_result.yx[i]
    dvdy = gradU_result.yy[i]
    dvdz = gradU_result.yz[i]

    dwdx = gradU_result.zx[i]
    dwdy = gradU_result.zy[i]
    dwdz = gradU_result.zz[i]

    conv_x = ux*dudx + uy*dudy + uz*dudz
    conv_y = ux*dvdx + uy*dvdy + uz*dvdz
    conv_z = ux*dwdx + uy*dwdy + uz*dwdz

    DUmDt[i] = dUdt + @SVector [conv_x, conv_y, conv_z]
end

function compute_Ur!(Ur, alpha, rho, g, DUmDt, rho1, rho2, mu1, d, tau_d, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(Ur)
    kernel! = _compute_Ur!(_setup(backend, workgroup, ndrange)...)
    kernel!(Ur, alpha, rho, g, DUmDt, rho1, rho2, mu1, d, tau_d)
end

@kernel inbounds=true function _compute_Ur!(Ur, alpha, rho, g, DUmDt, rho1, rho2, mu1, d, tau_d)
    i = @index(Global)
    TF = eltype(rho.values)

    rho_m = rho[i]
    rho_c = rho1[i]
    rho_d = rho2[i]
    mu_c  = mu1[i]
    tau   = tau_d[i]

    Ur_mag = norm(Ur[i])
    Re_p   = rho_c * Ur_mag * d / (mu_c + eps(TF))

    f_drag = ifelse(
        Re_p < TF(1000),
        one(TF) + TF(0.15) * Re_p^TF(0.687),
        TF(0.0183) * Re_p
    )
    f_drag = max(f_drag, one(TF))

    a_eff    = g - DUmDt[i]
    buoyancy = (rho_d - rho_m) / (rho_d + eps(TF))

    U_dm = (tau / (f_drag + eps(TF))) * buoyancy * a_eff

    alpha_c = max(alpha[i], TF(1e-3))
    Ur[i] = U_dm / alpha_c
end

turbulent_dispersion!(Ur, alpha, ∇alpha, turbulence::Laminar, Sc_t, config) = nothing

function turbulent_dispersion!(Ur, alpha, ∇alpha, turbulence, Sc_t, config)

    if !hasproperty(turbulence, :nut)
        return nothing
    end

    (; hardware) = config
    (; backend, workgroup) = hardware
    nut = turbulence.nut

    ndrange = length(Ur)
    kernel! = _turbulent_dispersion!(_setup(backend, workgroup, ndrange)...)
    kernel!(Ur, alpha, ∇alpha.result, nut, Sc_t)
end

@kernel inbounds=true function _turbulent_dispersion!(Ur, alpha, gradA, nut, Sc_t)
    i = @index(Global)
    TF = eltype(alpha.values)

    alpha_c = alpha[i]
    alpha_c_safe = max(alpha_c, TF(1e-3))
    alpha_d_safe = max(one(TF) - alpha_c, TF(1e-3))

    D_t   = nut[i] / TF(Sc_t)
    denom = alpha_c_safe * alpha_d_safe + eps(TF)
    coef  = D_t / denom

    gx = gradA.x[i]
    gy = gradA.y[i]
    gz = gradA.z[i]

    Ur[i] = Ur[i] + @SVector [coef*gx, coef*gy, coef*gz]
end

function face_dot_Sf!(phidotf, phif, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh  = phidotf.mesh
    faces = mesh.faces

    ndrange = length(faces)
    kernel! = _face_dot_Sf!(_setup(backend, workgroup, ndrange)...)
    kernel!(phidotf, phif, faces)
end

@kernel inbounds=true function _face_dot_Sf!(phidotf, phif, faces)
    i = @index(Global)
    (; area, normal) = faces[i]
    phidotf[i] = area * (phif.x[i]*normal[1] + phif.y[i]*normal[2] + phif.z[i]*normal[3])
end


@inline function _slip_coeff(alphaf, rhof, rho1f, rho2f, i, TF)
    af = alphaf[i]
    (af * (one(TF) - af) * (rho1f[i] + rho2f[i])) / (rhof[i] + eps(TF))
end

function div_slip_outer!(vector::VectorField, alphaf, rhof, rho1f, rho2f, Urf, config)
    mesh = vector.mesh
    (; cells, cell_nsign, cell_faces, faces) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(cells)
    kernel! = div_slip_outer_kernel!(_setup(backend, workgroup, ndrange)...)
    kernel!(cells, cell_faces, cell_nsign, faces, vector, alphaf, rhof, rho1f, rho2f, Urf)

    nbfaces = length(mesh.boundary_cellsID)
    ndrange = nbfaces
    kernel! = div_slip_outer_boundary_kernel!(_setup(backend, workgroup, ndrange)...)
    kernel!(faces, cells, vector, alphaf, rhof, rho1f, rho2f, Urf)
end

@kernel inbounds=true function div_slip_outer_kernel!(cells::AbstractArray{Cell{TF,SV,UR}}, cell_faces, cell_nsign, faces, vector, alphaf, rhof, rho1f, rho2f, Urf) where {TF,SV,UR}
    i = @index(Global)

    @inbounds begin
        (; volume, faces_range) = cells[i]

        reduction_x = zero(TF)
        reduction_y = zero(TF)
        reduction_z = zero(TF)

        for fi ∈ faces_range
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]
            (; area, normal) = faces[fID]

            ux = Urf.x[fID]
            uy = Urf.y[fID]
            uz = Urf.z[fID]

            coeff = _slip_coeff(alphaf, rhof, rho1f, rho2f, fID, TF)
            w = coeff * (ux*normal[1] + uy*normal[2] + uz*normal[3]) * area * nsign

            reduction_x += w * ux
            reduction_y += w * uy
            reduction_z += w * uz
        end

        vector.x[i] = reduction_x / volume
        vector.y[i] = reduction_y / volume
        vector.z[i] = reduction_z / volume
    end
end

@kernel function div_slip_outer_boundary_kernel!(faces, cells, vector, alphaf, rhof, rho1f, rho2f, Urf)
    i = @index(Global)

    @inbounds begin
        TF = eltype(Urf.x)
        cID = faces[i].ownerCells[1]
        volume = cells[cID].volume
        (; area, normal) = faces[i]

        ux = Urf.x[i]
        uy = Urf.y[i]
        uz = Urf.z[i]

        coeff = _slip_coeff(alphaf, rhof, rho1f, rho2f, i, TF)
        w = coeff * (ux*normal[1] + uy*normal[2] + uz*normal[3]) * area / volume

        Atomix.@atomic vector.x.values[cID] += w * ux
        Atomix.@atomic vector.y.values[cID] += w * uy
        Atomix.@atomic vector.z.values[cID] += w * uz
    end
end