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
- `coeffs`     -- Model coefficients (`Tref`, `Pr_t`).

### Turbulence

When the turbulence model supplies an eddy viscosity, the effective conductivity
picks up

    keff += (rho*cp)_f * nu_t / Pr_t

with `(rho*cp)_f` built from the same face blend as the rest of the equation.
`Pr_t` defaults to 0.85. A two-phase turbulent Prandtl number is not a settled
quantity, so it is exposed rather than buried; the laminar case is recovered
exactly when `nu_t = 0`.

### Example
    energy = Energy{TwoPhaseTemperature}(Tref=20.43)              # laminar or turbulent
    energy = Energy{TwoPhaseTemperature}(Tref=26.0, Pr_t=0.9)
"""
struct TwoPhaseTemperature{S,FS,C} <: AbstractEnergyModel
    T::S
    rho_cp::S
    rho_cp_prev::S
    keff::FS
    rho_cp_phi::FS
    rho_cp_imbalance::S
    S_T::S
    coeffs::C
end
Adapt.@adapt_structure TwoPhaseTemperature

Energy{TwoPhaseTemperature}(; Tref, Pr_t=0.85) = begin
    coeffs = (Tref=Tref, Pr_t=Pr_t)
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
        ScalarField(mesh),      # rho_cp_imbalance
        ScalarField(mesh),      # S_T
        energy.args
    )
end

"""
    initialise(energy::TwoPhaseTemperature, model, mdotf, config)

Build the temperature transport equation and its solver workspace.
"""
function initialise(energy::TwoPhaseTemperature, model, mdotf, config)
    (; T, keff, rho_cp, rho_cp_prev, rho_cp_phi, rho_cp_imbalance, S_T) = energy
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
        - Si(rho_cp_imbalance, T)
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

        # `cp` and `k` are read at FACES as well as at cells. A `ConstantScalar`
        # indexes correctly either way; a variable model stores a CELL field,
        # which must never be indexed by face ID. The solver therefore maintains
        # per-phase FACE fields for cp and k exactly as it already does for rho,
        # and those are what the face kernels are given.
        #
        # What remains unsupported is a model type the solver cannot refresh,
        # i.e. one with no `update_phase_property!` method - that would silently
        # leave the field at zero.
        for (name, model_name) in ((:cp, :cp_model), (:k, :k_model))
            m = getfield(phase, model_name)
            m isa Union{ConstCp,ConstK,TabulatedCp,TabulatedK} || throw(ArgumentError(
                """Phase $i has an unsupported `$name` model ($(typeof(m).name.wrapper)). \
`Energy{TwoPhaseTemperature}` accepts a constant (`$name = <value>`) or a tabulated \
model from `RealFluid(...)`."""))
        end
    end
    return nothing
end

"""
    energy!(energyModel, model, alpha_fluxf, mdotf, phi_drift, phase_faces, nueff,
            dpdt, mdot_pc, L, time, dt, config)

Advance the two-phase temperature equation by one time step.

`alpha_fluxf` is the limited volume-fraction face flux produced by
`advance_alpha!`; it is reused here so the energy advection matches the mass
advection exactly at the interface.

`phi_drift` is the volumetric drift flux `alpha*(1-alpha)*(Ur . Sf)` (zero for
`VOF`), which carries the enthalpy the two phases transport relative to the
mixture — see `energy_face_flux`.

`phase_faces` is the bundle of per-phase FACE property fields maintained by the
solver (`rho1f`, `cp1f`, `k1f`, ... ). Passing them as one named tuple rather
than as a growing list of positional arguments keeps this signature stable as
more properties are allowed to vary.
"""
function energy!(energyModel::EnergyEquationModel{E,S}, model, alpha_fluxf, mdotf,
                 phi_drift, phase_faces, nueff, dpdt, mdot_pc, L, time, dt,
                 config) where {E,S}
    (; energy_eqn, state) = energyModel
    (; T, rho_cp, rho_cp_prev, keff, rho_cp_phi, rho_cp_imbalance, S_T) = model.energy
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

    # `phase_faces` holds the per-phase FACE properties maintained by the solver.
    # They are required rather than indexing `phase.rho` (or `.cp`, `.k`)
    # directly, because a variable-property phase stores a CELL field and
    # indexing it by face ID is out of bounds. Entry 1/2 correspond to phases[1]
    # and phases[2] because the tracked phase index (`volume_fraction`) is
    # always 1.
    update_two_phase_energy_coeffs!(
        rho_cp, keff, rho_cp_phi, alpha, alphaf, alpha_fluxf, mdotf, phi_drift,
        phases[1], phases[2], phase_faces, model.turbulence,
        nueff, model.fluid.nuf, model.energy.coeffs.Pr_t, model.fluid.model, config)

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
    # DISCRETE rho_cp BALANCE CORRECTION.
    #
    # The conservative pair above is equivalent to rho_cp*DT/Dt ONLY when
    #
    #     (rho_cp^n - rho_cp^{n-1})/dt + div(rho_cp_phi) = 0
    #
    # holds discretely. Phase change breaks it two ways: it creates volume, so
    # the pressure equation deliberately imposes div(u) != 0, and it changes the
    # composition, so rho_cp itself has a source that the flux does not carry.
    #
    # Whatever remains is multiplied by T in the divergence term - and T here is
    # ABSOLUTE, ~29 K, so the residue is scaled by 29 rather than by any
    # temperature DIFFERENCE. Measured on the LH2 pipe: a volume source of
    # S_v ~ 19 /s gives a spurious -T*S_v ~ -550 K/s, which over 3e-4 s of
    # simulated time is -0.165 K. The observed near-wall cooling was -0.198 K
    # under 1e4 W/m2 of heating.
    #
    # Rather than derive the correct rho_cp source and hope it is complete, the
    # imbalance is MEASURED and cancelled: `- Si(imbalance, T)` removes exactly
    # the term that should not be there, whatever produced it. It vanishes
    # identically when the balance does hold, so single-phase and no-phase-change
    # cases are unaffected.
    div!(rho_cp_imbalance, rho_cp_phi, config)
    @. rho_cp_imbalance.values += (rho_cp.values - rho_cp_prev.values)/dt

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

Cell-centred properties are indexed as `phase.rho[i]`, which works uniformly for
`ConstantScalar` (index-independent) and `ScalarField` storage. Face-centred
properties come from `phase_faces`, never from the cell fields.
"""
function update_two_phase_energy_coeffs!(
    rho_cp, keff, rho_cp_phi, alpha, alphaf, alpha_fluxf, mdotf, phi_drift,
    phase_l, phase_v, phase_faces, turbulence, nueff, nuf, Pr_t, mp_model, config)

    (; hardware) = config
    (; backend, workgroup) = hardware

    blend_rho_cp!(rho_cp, alpha, phase_l, phase_v, config)

    # Face-centred: every property MUST come from the face fields (see `energy!`).
    # The turbulent conductivity is folded in here rather than in a second pass
    # because this kernel already has the face rho*cp the eddy term needs.
    nut_scale = turbulent_conductivity_scale(turbulence, Pr_t)

    ndrange = length(alphaf)
    kernel! = _blend_energy_faces!(_setup(backend, workgroup, ndrange)...)
    kernel!(keff, rho_cp_phi, alphaf, alpha_fluxf, mdotf, phi_drift,
            phase_faces.rho1f, phase_faces.cp1f, phase_faces.k1f,
            phase_faces.rho2f, phase_faces.cp2f, phase_faces.k2f,
            nueff, nuf, nut_scale, mp_model)

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

    # beta is resolved to an indexable field here (a ConstantScalar(0) when the
    # phase has none) so the kernel never sees a `nothing` and reads the same way
    # for constant and tabulated expansivity alike.
    beta_l = _phase_beta_field(phases[1])
    beta_v = _phase_beta_field(phases[2])

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
    betaT = a*phase_betaT(eos_l, beta_l[i], t) + (one(TF) - a)*phase_betaT(eos_v, beta_v[i], t)
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

"""
    energy_face_flux(mp_model, alpha_fluxf, mdotf, phi_drift, rcp_l, rcp_v, rho_cp_f)

Advecting flux of the temperature equation, `rho*cp*phi`, built to match the
mass flux `rhoPhi` of the SAME multiphase model - see `blend_rhoPhi!`.

The two must agree. If the energy equation advects with a different flux from
the one carrying mass, the mismatch appears as a spurious source scaled by
`(rho*cp)_l - (rho*cp)_v`, which for LH2/GH2 is ~1.2e6 against the momentum
equation's `rho_l - rho_v` of ~48 - four orders of magnitude more damaging.

- `VOF`: the limited volume-fraction flux carries `(rho*cp)_l` and the remainder
  carries `(rho*cp)_v`, mirroring `blend_rhoPhi!(::VOF, ...)`.
- `Mixture`: `mdotf*(rho*cp)_f`, plus the SLIP ENTHALPY FLUX below. It must not
  instead pick up the drift term that `alpha_fluxf` carries, which is the same
  transport with the wrong coefficient.

The distinction is invisible while `alpha = 1` everywhere, because then
`alpha_fluxf == mdotf` and both forms coincide. It only bites once a second
phase appears - i.e. exactly when wall boiling starts producing vapour.

## Slip enthalpy flux (`Mixture`)

The phases convect their own enthalpy at their own velocity, so the exact energy
flux is `sum_i alpha_i*rho_i*cp_i*T*u_i`. Splitting each `u_i` about the
volume-averaged `u_j` that this solver transports with, and using

    u_1 - u_j = -(1 - alpha)*u_r,      u_2 - u_j = +alpha*u_r

gives

    sum_i alpha_i*rho_i*cp_i*T*u_i
        = (rho*cp)_m*T*u_j  +  T*alpha*(1-alpha)*[(rho*cp)_2 - (rho*cp)_1]*u_r

The first term is the existing `mdotf*(rho*cp)_f`; the second is `phi_drift`
scaled by the difference of the phase heat capacities per unit volume. Rising
vapour carries its own thermal capacity with it, and without this term that
transport is simply absent.

Adding it to `rho_cp_phi` rather than as a separate source is deliberate: the
divergence of whatever `rho_cp_phi` holds is measured and cancelled by
`rho_cp_imbalance`, so the drift contributes the convective transport of `T`
without also injecting a spurious `T*div(drift flux)`.

**Sensible heat only.** `H_i = cp_i*T` here, with no per-phase reference
enthalpy, so the latent heat the vapour carries with it is NOT transported by
this term - it enters where the phase change happens, through `add_latent_heat!`.
That is consistent with solving a temperature equation rather than an enthalpy
equation, but it does mean drift transports sensible heat only.
"""
@inline energy_face_flux(::VOF, alpha_fluxf_i, mdotf_i, phi_drift_i,
                         rcp_l, rcp_v, rho_cp_f) =
    alpha_fluxf_i*(rcp_l - rcp_v) + mdotf_i*rcp_v

@inline energy_face_flux(::Mixture, alpha_fluxf_i, mdotf_i, phi_drift_i,
                         rcp_l, rcp_v, rho_cp_f) =
    mdotf_i*rho_cp_f + phi_drift_i*(rcp_v - rcp_l)

@kernel inbounds=true function _blend_energy_faces!(
    keff, rho_cp_phi, alphaf, alpha_fluxf, mdotf, phi_drift,
    rho_l, cp_l, k_l, rho_v, cp_v, k_v, nueff, nuf, nut_scale, mp_model)
    i = @index(Global)
    TF = eltype(keff.values)
    af = alphaf[i]

    rcp_l = rho_l[i]*cp_l[i]
    rcp_v = rho_v[i]*cp_v[i]
    rho_cp_f = af*rcp_l + (one(TF) - af)*rcp_v

    # Molecular part, then the eddy part. `nueff - nuf` is the eddy viscosity:
    # the solver builds `nueff` as the mixture laminar viscosity plus `nut`, so
    # the difference recovers `nut` without needing the turbulence model's own
    # field interpolated to faces. `max(..., 0)` guards the laminar case, where
    # the two are the same field and round-off could give a small negative.
    nut = max(nueff[i] - nuf[i], zero(TF))
    keff[i] = af*k_l[i] + (one(TF) - af)*k_v[i] + rho_cp_f*nut*nut_scale

    # Must match the mass flux of the SAME multiphase model - see
    # `energy_face_flux`. Getting this wrong is invisible while alpha = 1 and
    # catastrophic as soon as a second phase appears.
    rho_cp_phi[i] = energy_face_flux(mp_model, alpha_fluxf[i], mdotf[i],
                                     phi_drift[i], rcp_l, rcp_v, rho_cp_f)
end

"""
    turbulent_conductivity_scale(turbulence, Pr_t) -> 1/Pr_t or 0

The factor multiplying `(rho*cp)_f * nu_t` in the effective conductivity, i.e.
`1/Pr_t` for a turbulent closure and exactly zero for a laminar one.

Returning zero rather than skipping the term keeps `_blend_energy_faces!` a
single branch-free kernel that is used unchanged either way.

Dispatch cannot be on `::Laminar`: `ModelPhysics.jl` includes `Energy` before
`Turbulence`, so that type does not exist yet at definition time. The property
check below serves the same purpose - `Laminar` does carry a `nut`, but as a
`ConstantScalar` (see RANS_laminar.jl), whereas a real closure allocates a
`ScalarField`.
"""
function turbulent_conductivity_scale(turbulence, Pr_t)
    hasproperty(turbulence, :nut) || return 0.0
    turbulence.nut isa ConstantScalar && return 0.0
    Pr_t > 0 || throw(ArgumentError(
        "`Pr_t` must be positive, got $Pr_t. Set it on the energy model, e.g. \
`Energy{TwoPhaseTemperature}(Tref=..., Pr_t=0.85)`."))
    return 1.0/Pr_t
end
