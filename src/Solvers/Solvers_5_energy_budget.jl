export EnergyBudget, energy_budget!, report_energy_budget

# =============================================================================
#  Global energy budget for the two-phase multiphase solver
# =============================================================================
#
#  Integrated over the WHOLE domain, every internal transfer cancels and only
#  boundary terms and total storage survive:
#
#      d/dt Int(rho_m h_m dV)  =  Int_wall(q_w dA)  +  (mdot h)_in - (mdot h)_out
#
#  with the mixture enthalpy carrying the latent term explicitly,
#
#      rho_m h_m = rho_cp * T  +  (1 - alpha) * rho_v * h_fg
#
#  The imbalance IS the spurious energy source, in watts, directly comparable to
#  the applied wall power.
#
#  WHY A GLOBAL BUDGET AND NOT A LOCAL BALANCE
#
#  A local balance (e.g. the vapour mass balance) mixes discretisation error,
#  time-level offsets and genuine physics, and for a compressible phase it has no
#  independently-known right answer - so a non-zero residual cannot be
#  attributed. That ambiguity produced several wrong conclusions on the LH2 pipe
#  case; see dev_notes_LH2_pipe_boiling.md.
#
#  A global budget has exactly one right answer - zero - and it does not depend
#  on how any individual term is discretised. Compressibility, drift, alpha
#  transport and latent storage are all INTERNAL transfers here and cancel on
#  integration.
#
#  VALIDATE THE INSTRUMENT BEFORE TRUSTING IT
#
#  Run it first on a case whose answer is known:
#
#    1. q_w = 0, no phase change, no wall boiling  -> both sides zero
#       independently. Catches sign and area errors that a single non-zero case
#       can mask.
#    2. q_w on, alpha fixed at 1, no phase change  -> must close to the linear
#       solver tolerance.
#
#  Only then apply it to a boiling case, and attribute the imbalance by enabling
#  one mechanism at a time.
# =============================================================================

"""
    EnergyBudget(mesh; h_fg, wall_patches, inlet_patches, outlet_patches)

Working state for the global energy budget. Holds the previous total enthalpy so
successive calls can form `dE/dt`.

### Keywords
- `h_fg` -- latent heat [J/kg], the same scalar the solver uses for the energy
            sink, so the storage and source terms cannot disagree.
- `wall_patches`   -- heated patches. The flux is read from their `FixedHeatFlux`
                      condition on `T`, not passed separately, so it cannot drift
                      out of step with the boundary condition actually applied.
- `inlet_patches`, `outlet_patches` -- flow boundaries.
"""
mutable struct EnergyBudget{F<:AbstractFloat,S}
    h_fg::F
    wall_patches::S
    inlet_patches::S
    outlet_patches::S
    E_prev::F
    initialised::Bool
end

function EnergyBudget(; h_fg, wall_patches=(), inlet_patches=(), outlet_patches=())
    tup(x) = x isa Symbol ? (x,) : Tuple(x)
    return EnergyBudget(float(h_fg), tup(wall_patches), tup(inlet_patches),
                        tup(outlet_patches), 0.0, false)
end

"""
    total_mixture_enthalpy(model, h_fg) -> J

Volume integral of `rho_cp*T + (1-alpha)*rho_v*h_fg`.

The second term is the latent energy held in the vapour present. It must be
included: without it, converting liquid to vapour at constant temperature would
appear to change the domain's energy, and the budget would show an imbalance
that is an artefact of the accounting rather than of the solver.
"""
function total_mixture_enthalpy(model, h_fg)
    (; alpha, phases) = model.fluid
    main = model.fluid.volume_fraction
    secondary = 3 - main
    pl, pv = phases[main], phases[secondary]
    T = model.energy.T
    cells = model.domain.cells

    E = 0.0
    for i in eachindex(alpha.values)
        a = alpha.values[i]
        rcp = a*pl.rho[i]*pl.cp[i] + (1 - a)*pv.rho[i]*pv.cp[i]
        E += (rcp*T.values[i] + (1 - a)*pv.rho[i]*h_fg)*cells[i].volume
    end
    return E
end

"""
    boundary_enthalpy_flux(model, patches, h_fg, config) -> W

Net enthalpy leaving the domain through `patches`, positive outward.

Uses the VOLUMETRIC face flux times the VOLUMETRIC enthalpy,

    H_f = (U_f . S_f) * [ rho_cp_f * T_f + (1-alpha_f) * rho_v_f * h_fg ]

which avoids dividing by the mixture density and so cannot misbehave where one
phase vanishes.

Face values follow the boundary condition: a `Dirichlet` supplies its own value
(exact at a prescribed inlet), anything else takes the owner cell value, which is
what `correct_boundaries!` does for the zero-gradient family.
"""
function boundary_enthalpy_flux(model, patches, h_fg, config)
    mesh = model.domain
    (; faces, boundary_cellsID) = mesh
    (; alpha, phases) = model.fluid
    main = model.fluid.volume_fraction
    pl, pv = phases[main], phases[3 - main]
    T = model.energy.T
    U = model.momentum.U
    boundaries_cpu = get_boundaries(mesh.boundaries)
    BCs = config.boundaries

    total = 0.0
    for name in patches
        idx = boundary_index(boundaries_cpu, name)
        idx === nothing && throw(ArgumentError("`:$name` is not a boundary of this mesh"))

        bcU = _find_boundary_condition(BCs.U, idx)
        bcT = hasproperty(BCs, :T) ? _find_boundary_condition(BCs.T, idx) : nothing
        bcA = hasproperty(BCs, :alpha) ? _find_boundary_condition(BCs.alpha, idx) : nothing

        for fID in boundaries_cpu[idx].IDs_range
            cID = boundary_cellsID[fID]
            (; area, normal) = faces[fID]

            uvec = bcU isa Dirichlet ? bcU.value :
                   SVector(U.x.values[cID], U.y.values[cID], U.z.values[cID])
            flux = area*(uvec[1]*normal[1] + uvec[2]*normal[2] + uvec[3]*normal[3])

            Tf = bcT isa Dirichlet ? bcT.value : T.values[cID]
            af = bcA isa Dirichlet ? bcA.value : alpha.values[cID]

            rcp = af*pl.rho[cID]*pl.cp[cID] + (1 - af)*pv.rho[cID]*pv.cp[cID]
            total += flux*(rcp*Tf + (1 - af)*pv.rho[cID]*h_fg)
        end
    end
    return total
end

"""
    wall_heat_input(model, patches, config) -> W

Total power entering through `patches`, read from their `FixedHeatFlux`
condition on `T` so it always matches the flux actually applied.
"""
function wall_heat_input(model, patches, config)
    mesh = model.domain
    (; faces) = mesh
    boundaries_cpu = get_boundaries(mesh.boundaries)

    total = 0.0
    for name in patches
        idx = boundary_index(boundaries_cpu, name)
        idx === nothing && throw(ArgumentError("`:$name` is not a boundary of this mesh"))
        bc = _find_boundary_condition(config.boundaries.T, idx)
        bc isa FixedHeatFlux || throw(ArgumentError(
            """Energy budget: patch `:$name` carries a \
$(bc === nothing ? "missing" : string(typeof(bc).name.wrapper)) condition on `T`, \
but the budget reads the applied power from a `FixedHeatFlux`."""))
        for fID in boundaries_cpu[idx].IDs_range
            total += bc.value*faces[fID].area
        end
    end
    return total
end

"""
    energy_budget!(eb::EnergyBudget, model, dt, config) -> NamedTuple

Evaluate the global energy budget for the step just completed.

Returns `(E, dEdt, Q_wall, H_in, H_out, imbalance, relative)` in watts, where

    imbalance = dEdt - (Q_wall + H_in - H_out)

and `relative` is that imbalance as a fraction of the applied wall power - the
number to read. **Zero is the only correct answer**; anything comparable to one
means the solver is creating or destroying energy at the same rate as the heater.

The first call has no previous state and returns `dEdt = NaN`; use it to prime
the budget and read from the second call onwards.
"""
function energy_budget!(eb::EnergyBudget, model, dt, config)
    E = total_mixture_enthalpy(model, eb.h_fg)

    Q_wall = isempty(eb.wall_patches) ? 0.0 : wall_heat_input(model, eb.wall_patches, config)
    # `boundary_enthalpy_flux` is positive OUTWARD, so an inlet (flow entering,
    # negative outward flux) contributes a positive gain once negated.
    H_in  = isempty(eb.inlet_patches)  ? 0.0 :
            -boundary_enthalpy_flux(model, eb.inlet_patches, eb.h_fg, config)
    H_out = isempty(eb.outlet_patches) ? 0.0 :
            boundary_enthalpy_flux(model, eb.outlet_patches, eb.h_fg, config)

    dEdt = eb.initialised ? (E - eb.E_prev)/dt : NaN
    eb.E_prev = E
    eb.initialised = true

    imbalance = dEdt - (Q_wall + H_in - H_out)
    scale = max(abs(Q_wall), abs(H_in), 1e-30)

    return (E=E, dEdt=dEdt, Q_wall=Q_wall, H_in=H_in, H_out=H_out,
            imbalance=imbalance, relative=imbalance/scale)
end

"""
    report_energy_budget(b; step=nothing)

One-line summary of an [`energy_budget!`](@ref) result, in watts.
"""
function report_energy_budget(b; step=nothing)
    isnan(b.dEdt) && return nothing
    lbl = step === nothing ? "" : "step $(lpad(step,5))  "
    @info "$(lbl)energy budget [W]" dEdt=b.dEdt Q_wall=b.Q_wall H_in=b.H_in H_out=b.H_out imbalance=b.imbalance relative=b.relative
    return nothing
end
