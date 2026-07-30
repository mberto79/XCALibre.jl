export TwoPhaseTemperature

"""
    TwoPhaseTemperature <: AbstractEnergyModel

Two-phase energy transport for the VOF/Mixture multiphase solver, solved
directly for **temperature** rather than enthalpy:

    d(rho*cp*T)/dt + div(rho*cp*phi*T) - div(keff*grad(T)) = S_T

Solving `T` rather than a sensible enthalpy `h = cp(T - Tref)` is deliberate.
Across a liquid/vapour interface `cp` is discontinuous (for LH2 at 20 K it jumps
from 9660 to 12200 J/kg/K, ~26%), so `h` is discontinuous even where `T` is
smooth. Recovering `T = h/cp + Tref` would then smear that jump across the
interface and corrupt `(T - T_sat)` — precisely the quantity every interfacial
phase change model keys off. Fernandes et al. (2026) likewise solve their
energy equation for temperature.

The mixture coefficients are volume-fraction blends,

    rho*cp = alpha*rho_l*cp_l + (1-alpha)*rho_v*cp_v
    keff   = alphaf*k_l + (1-alphaf)*k_v          (+ turbulent part)

and the advecting flux is built from the same `alpha_fluxf` used by
`blend_rhoPhi!`, so that energy and mass advection stay consistent at the
interface (an inconsistency there shows up as interface temperature drift).

### Fields
- `T`          -- Temperature ScalarField.
- `rho_cp`     -- Mixture volumetric heat capacity ScalarField.
- `keff`       -- Mixture (+turbulent) conductivity FaceScalarField.
- `rho_cp_phi` -- Advecting flux for the temperature equation FaceScalarField.
- `S_T`        -- Volumetric energy source ScalarField (phase change latent heat).
- `coeffs`     -- Model coefficients (`Tref`).

### Example
    energy = Energy{TwoPhaseTemperature}(Tref=20.43)
"""
struct TwoPhaseTemperature{S,FS,C} <: AbstractEnergyModel
    T::S
    rho_cp::S
    rho_cp_prev::S
    keff::FS
    rho_cp_phi::FS
    S_T::S
    coeffs::C
end
Adapt.@adapt_structure TwoPhaseTemperature

Energy{TwoPhaseTemperature}(; Tref) = begin
    coeffs = (Tref=Tref,)
    ARG = typeof(coeffs)
    Energy{TwoPhaseTemperature,ARG}(coeffs)
end

# More specific than the generic he_energy functor, so this wins for dispatch.
(energy::Energy{EnergyModel, ARG})(mesh, fluid) where {EnergyModel<:TwoPhaseTemperature,ARG} = begin
    TwoPhaseTemperature(
        ScalarField(mesh),      # T
        ScalarField(mesh),      # rho_cp
        ScalarField(mesh),      # rho_cp_prev
        FaceScalarField(mesh),  # keff
        FaceScalarField(mesh),  # rho_cp_phi
        ScalarField(mesh),      # S_T
        energy.args
    )
end

"""
    initialise(energy::TwoPhaseTemperature, model, mdotf, config)

Build the temperature transport equation and its solver workspace.
"""
function initialise(energy::TwoPhaseTemperature, model, mdotf, config)
    (; T, keff, rho_cp, rho_cp_prev, rho_cp_phi, S_T) = energy
    (; solvers, schemes, boundaries) = config

    _assert_phase_thermal_properties(model.fluid.phases)

    # Seed rho_cp from the initial alpha so the first time step has a valid
    # previous-time coefficient (see the conservative form note in `energy!`).
    (; alpha, phases) = model.fluid
    blend_rho_cp!(rho_cp, alpha, phases[1], phases[2], config)
    @. rho_cp_prev.values = rho_cp.values

    energy_eqn = (
        Time{schemes.T.time}(rho_cp, T)
        + Divergence{schemes.T.divergence}(rho_cp_phi, T)
        - Laplacian{schemes.T.laplacian}(keff, T)
        ==
        Source(S_T)
    ) → ScalarEquation(T, boundaries.T)

    @reset energy_eqn.preconditioner = set_preconditioner(solvers.T.preconditioner, energy_eqn)
    @reset energy_eqn.solver = _workspace(solvers.T.solver, _b(energy_eqn))

    state = ModelState((:T, 1.0), false)
    return EnergyEquationModel(energy_eqn, state)
end

# Fail at setup, naming the offending phase, rather than deep inside a kernel.
function _assert_phase_thermal_properties(phases)
    for (i, phase) in enumerate(phases)
        for name in (:k, :cp)
            getfield(phase, name) === nothing && throw(ArgumentError(
                """Phase $i is missing `$name`, which `Energy{TwoPhaseTemperature}` requires. \
Add it to the corresponding `Phase(...)`, e.g. `Phase(rho=..., mu=..., k=..., cp=...)`."""))
        end

        # `cp` and `k` are also read at FACES, where only a ConstantScalar
        # indexes correctly (a cell ScalarField would run out of bounds). Density
        # is exempt: the solver maintains per-phase face density fields for it.
        # Refuse a variable cp/k model rather than fail inside a kernel.
        #
        # Fernandes et al. use temperature-dependent cp and k from NIST, so
        # supporting those is a known future extension: it needs per-phase face
        # fields for cp and k, mirroring what the solver already does for rho.
        for (name, model_name) in ((:cp, :cp_model), (:k, :k_model))
            m = getfield(phase, model_name)
            m isa Union{ConstCp,ConstK} || throw(ArgumentError(
                """Phase $i has a variable `$name` model ($(typeof(m).name.wrapper)), which \
`Energy{TwoPhaseTemperature}` does not support yet - only the density may vary. \
Pass a constant, e.g. `$name = <value>`."""))
        end
    end
    return nothing
end

"""
    energy!(energyModel, model, alpha_fluxf, mdotf, nueff, time, dt, config)

Advance the two-phase temperature equation by one time step.

`alpha_fluxf` is the limited volume-fraction face flux produced by
`advance_alpha!`; it is reused here so the energy advection matches the mass
advection exactly at the interface.
"""
function energy!(energyModel::EnergyEquationModel{E,S}, model, alpha_fluxf, mdotf,
                 rho1f, rho2f, nueff, dpdt, mdot_pc, L, time, dt, config) where {E,S}
    (; energy_eqn, state) = energyModel
    (; T, rho_cp, rho_cp_prev, keff, rho_cp_phi, S_T) = model.energy
    (; alpha, alphaf, phases) = model.fluid
    (; solvers, boundaries) = config

    mesh = model.domain

    # Snapshot the previous-time coefficient BEFORE it is recomputed below.
    # Required for a conservative time term - see the `discretise!` call.
    @. rho_cp_prev.values = rho_cp.values

    # Volumetric energy sources: pressure work (zero when `dpdt === nothing`, i.e.
    # an all-incompressible mixture) and latent heat (zero when `mdot_pc ===
    # nothing`, i.e. no phase change).
    update_pressure_work!(S_T, alpha, phases, T, dpdt, config)
    add_latent_heat!(S_T, mdot_pc, L, config)

    # `rho1f`/`rho2f` are the per-phase FACE densities maintained by the solver.
    # They are required rather than indexing `phase.rho` directly, because a
    # variable-EOS phase stores a CELL field and indexing it by face ID is out of
    # bounds. They correspond to phases[1] and phases[2] because the tracked
    # phase index (`volume_fraction`) is always 1.
    update_two_phase_energy_coeffs!(
        rho_cp, keff, rho_cp_phi, alpha, alphaf, alpha_fluxf, mdotf,
        phases[1], phases[2], rho1f, rho2f, model.turbulence, nueff, config)

    # The time term MUST use the previous-time rho_cp, giving the conservative
    # form  (rho_cp^n T^n - rho_cp^{n-1} T^{n-1})/dt  to pair with the
    # conservative divergence  div(rho_cp_phi T).
    #
    # `discretise!` otherwise defaults `rho_prev` to the term's own (current)
    # flux, which yields the NON-conservative rho_cp*dT/dt. Mixing that with a
    # conservative divergence leaves a spurious `T*div(rho_cp_phi)` source. It is
    # scaled by (rho*cp)_l - (rho*cp)_v = 6.7e5 rather than the momentum
    # equation's rho_l - rho_v = 69.5, so the error is ~1e4 times more damaging
    # here: an adiabatic uniform-T field on the K-Site wedge drifted 7.9 K in
    # five 1 ms steps before this was fixed. The momentum equation already
    # passes its `rho_prev` explicitly for the same reason.
    discretise!(energy_eqn, T, config, rho_prev=rho_cp_prev)
    apply_boundary_conditions!(energy_eqn, boundaries.T, nothing, time, config)
    implicit_relaxation_diagdom!(energy_eqn, T.values, solvers.T.relax, nothing, config)
    update_preconditioner!(energy_eqn.preconditioner, mesh, config)
    T_res = solve_system!(energy_eqn, solvers.T, T, nothing, config)

    if !isnothing(solvers.T.limit)
        clamp!(T.values, solvers.T.limit[1], solvers.T.limit[2])
    end

    state.residuals = (:T, T_res)
    state.converged = T_res <= solvers.T.convergence
    return nothing
end

"""
    update_two_phase_energy_coeffs!(...)

Rebuild the volume-fraction-blended coefficients of the temperature equation.

Per-phase properties are indexed as `phase.rho[i]`, which works uniformly for
both `ConstantScalar` (index-independent) and `ScalarField` storage, so this is
already correct once variable-density phases land in step 5.
"""
function update_two_phase_energy_coeffs!(
    rho_cp, keff, rho_cp_phi, alpha, alphaf, alpha_fluxf, mdotf,
    phase_l, phase_v, rho1f, rho2f, turbulence, nueff, config)

    (; hardware) = config
    (; backend, workgroup) = hardware

    blend_rho_cp!(rho_cp, alpha, phase_l, phase_v, config)

    # Face-centred: densities MUST come from the face fields (see `energy!`).
    ndrange = length(alphaf)
    kernel! = _blend_energy_faces!(_setup(backend, workgroup, ndrange)...)
    kernel!(keff, rho_cp_phi, alphaf, alpha_fluxf, mdotf,
            rho1f, phase_l.cp, phase_l.k,
            rho2f, phase_v.cp, phase_v.k)

    add_turbulent_conductivity!(keff, turbulence, nueff, rho_cp, config)
    return nothing
end

"""
    update_pressure_work!(S_T, alpha, phases, T, dpdt, config)

Pressure-work source of the temperature equation,

    S_T = [alpha*(beta*T)_l + (1-alpha)*(beta*T)_v] * Dp/Dt

`Dp/Dt` is approximated by the local `dp/dt` (low-Mach: the convective part is
negligible here).

This term is what couples ullage pressurisation back into the gas temperature.
For an ideal gas `beta*T = 1`, so the vapour picks up the full `dp/dt`; for the
liquid it is `beta*T ~ 0.33` at 20 K, a smaller but not negligible contribution.
Omitting it entirely would make the self-pressurisation prediction wrong, not
merely approximate.

`dpdt === nothing` (an all-incompressible mixture) zeroes the source, which is
exact: with no compressible phase there is no pressure work to do.
"""
function update_pressure_work!(S_T, alpha, phases, T, ::Nothing, config)
    fill!(S_T.values, zero(eltype(S_T.values)))
    return nothing
end

function update_pressure_work!(S_T, alpha, phases, T, dpdt, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # beta is resolved to a plain number here (0 when the phase has none) so the
    # kernel never sees a `nothing`.
    beta_l = _phase_beta_value(phases[1])
    beta_v = _phase_beta_value(phases[2])

    ndrange = length(S_T)
    kernel! = _update_pressure_work!(_setup(backend, workgroup, ndrange)...)
    kernel!(S_T, alpha, T, dpdt, phases[1].rho_model, phases[2].rho_model, beta_l, beta_v)
    return nothing
end

"""
    add_latent_heat!(S_T, mdot, L, config)

Add the latent heat of phase change to the energy source (Fernandes et al. Eq. 7):

    S_T -= mdot * L

`mdot` is the volumetric phase change rate [kg/m^3/s], positive for evaporation,
so evaporation is a heat SINK — the latent heat is drawn out of the local fluid.
Condensation (`mdot < 0`) releases it.

`mdot === nothing` is the no-phase-change case and adds nothing.
"""
add_latent_heat!(S_T, ::Nothing, L, config) = nothing

function add_latent_heat!(S_T, mdot, L, config)
    @. S_T.values -= mdot.values*L
    return nothing
end

@kernel inbounds=true function _update_pressure_work!(
    S_T, alpha, T, dpdt, eos_l, eos_v, beta_l, beta_v)
    i = @index(Global)
    TF = eltype(S_T.values)
    a = alpha[i]
    t = T[i]
    betaT = a*phase_betaT(eos_l, beta_l, t) + (one(TF) - a)*phase_betaT(eos_v, beta_v, t)
    S_T[i] = betaT*dpdt[i]
end

"""
    blend_rho_cp!(rho_cp, alpha, phase_l, phase_v, config)

Cell-centred mixture volumetric heat capacity,
`rho_cp = alpha*(rho*cp)_l + (1-alpha)*(rho*cp)_v`.

`phase.rho` may be a `ConstantScalar` or a cell `ScalarField`; both index
correctly here.
"""
function blend_rho_cp!(rho_cp, alpha, phase_l, phase_v, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(alpha)
    kernel! = _blend_rho_cp!(_setup(backend, workgroup, ndrange)...)
    kernel!(rho_cp, alpha, phase_l.rho, phase_l.cp, phase_v.rho, phase_v.cp)
    return nothing
end

@kernel inbounds=true function _blend_rho_cp!(rho_cp, alpha, rho_l, cp_l, rho_v, cp_v)
    i = @index(Global)
    TF = eltype(rho_cp.values)
    a = alpha[i]
    rho_cp[i] = a*rho_l[i]*cp_l[i] + (one(TF) - a)*rho_v[i]*cp_v[i]
end

@kernel inbounds=true function _blend_energy_faces!(
    keff, rho_cp_phi, alphaf, alpha_fluxf, mdotf, rho_l, cp_l, k_l, rho_v, cp_v, k_v)
    i = @index(Global)
    TF = eltype(keff.values)
    af = alphaf[i]

    keff[i] = af*k_l[i] + (one(TF) - af)*k_v[i]

    # Mirrors `blend_rhoPhi!` for VOF: the volumetric flux of the tracked phase
    # carries (rho*cp)_l and the remainder carries (rho*cp)_v, so energy and
    # mass advection use the same limited alpha flux.
    rcp_l = rho_l[i]*cp_l[i]
    rcp_v = rho_v[i]*cp_v[i]
    rho_cp_phi[i] = alpha_fluxf[i]*(rcp_l - rcp_v) + mdotf[i]*rcp_v
end

# Turbulent contribution to the effective conductivity.
#
# Dispatch cannot be on `::Laminar` here: `ModelPhysics.jl` includes `Energy`
# before `Turbulence`, so the type does not exist yet at definition time. A
# runtime property check serves the same purpose.
#
# A turbulent closure is deliberately NOT implemented. Adding rho*cp*nut/Pr_t
# needs rho*cp interpolated to faces, and more importantly a two-phase
# turbulent Prandtl number that has not been validated here. Fernandes et al.
# (Sec. 3.2) run these cases laminar, reporting that laminar reproduces the
# pressure rise and vapour stratification better than the usual RANS closures.
# Refusing loudly beats silently applying an unvalidated formula.
function add_turbulent_conductivity!(keff, turbulence, nueff, rho_cp, config)
    hasproperty(turbulence, :nut) || return nothing
    # `Laminar` does carry a `nut`, but as a `ConstantScalar` (see
    # RANS_laminar.jl), whereas an actual closure allocates a `ScalarField`.
    # That distinction is usable here; the `Laminar` type itself is not, for the
    # include-order reason above.
    turbulence.nut isa ConstantScalar && return nothing
    throw(ArgumentError(
        """`Energy{TwoPhaseTemperature}` currently supports laminar flow only, but the \
turbulence model provides `nut`. Use `RANS{Laminar}()`, or implement the turbulent \
conductivity in `add_turbulent_conductivity!` (src/ModelPhysics/Energy/multiphase_energy.jl).

The reference cases for this solver path (Fernandes et al. 2026, K-Site/MHTB) are run \
laminar by design."""))
end
