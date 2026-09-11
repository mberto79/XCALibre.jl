export WallBoilingState, initialise_wall_boiling, wall_boiling_source!
export write_wall_boiling_surface
export wall_boiling_report, report_wall_boiling, LAST_WALL_REPORT

# =============================================================================
#  Solver-side coupling of the RPI wall boiling model
# =============================================================================
#
#  The physics lives in `src/ModelPhysics/2_wall_boiling_models.jl`; this file
#  is only the plumbing that evaluates it on the heated wall patches and turns
#  the evaporative flux into a volumetric vapour source in the near-wall cells.
#
#  Deliberately NOT a boundary condition. The wall's thermal boundary condition
#  is still `FixedHeatFlux`, which delivers the whole q_w into the near-wall
#  cell; RPI splits that flux only to decide how much of it leaves as vapour.
#  The latent heat sink that the vapour source already produces (through
#  `add_latent_heat!`) takes q_e back out of the liquid, so the liquid is
#  ultimately heated by q_c + q_q without either the boundary condition or the
#  energy equation needing to know that boiling is happening.
#
#  Doing it the other way round - modifying the boundary flux - would double
#  count, because the latent heat would then be removed twice.
# =============================================================================

"""
    WallBoilingState

Everything the wall boiling pass needs, resolved once at solver setup:

- `model`     -- the [`RPI`](@ref) model itself,
- `patch_BCs` -- the `FixedHeatFlux` boundary conditions of the heated patches,
                 in the order the user listed them,
- `mdot_wall` -- volumetric vapour generation rate [kg/m^3/s], per cell,
- `u_tau`     -- wall friction velocity [m/s], per boundary face,
- `T_wall`    -- solved wall temperature [K], per boundary face (diagnostic),
- `q_evap`, `q_quench`, `q_conv` -- the flux partition per boundary face
                 [W/m^2], written out for inspection.

Apart from `mdot_wall` and `u_tau` these are diagnostics rather than solver
state: nothing reads them back. They exist because the partition is the whole
substance of the model, and a run that cannot show how the wall flux was split
cannot be assessed.
"""
struct WallBoilingState{M,B,S,F,G}
    model::M
    patch_BCs::B
    mdot_wall::S
    u_tau::F
    T_wall::F
    q_evap::F
    q_quench::F
    q_conv::F
    dT_sup::F
    y_plus::F
    A_b::F
    mdot_area::F
    T_liquid::F
    h_conv::F
    D_departure::F
    K_dry::F
    alpha_delta::F
    alpha_delta_raw::F
    face_graph::G
end

"""
    initialise_wall_boiling(wall_boiling, model, config)

Resolve the heated wall patches named by the model and allocate its working
fields. Returns `nothing` when no wall boiling model was supplied, which is the
switch that keeps the whole pass out of a run that does not want it.

Fails at setup - naming the patch - rather than inside a kernel, if a named
patch does not exist or does not carry a `FixedHeatFlux` temperature condition.
"""
initialise_wall_boiling(::Nothing, model, config) = nothing

function initialise_wall_boiling(wb::RPI, model, config)
    mesh = model.domain
    (; boundaries) = config

    hasproperty(boundaries, :T) || throw(ArgumentError(
        """A wall boiling model needs a temperature field, but no `T` boundary conditions \
were assigned. Wall boiling requires `Energy{TwoPhaseTemperature}` and a `T` entry in \
`assign(...)`."""))

    model.energy isa TwoPhaseTemperature || throw(ArgumentError(
        """`RPI` wall boiling needs a temperature field, but the energy model is \
$(model.energy === nothing ? "Energy{Isothermal}" : typeof(model.energy).name.wrapper). \
Use `Energy{TwoPhaseTemperature}(Tref=...)`."""))

    boundaries_cpu = get_boundaries(mesh.boundaries)

    patch_BCs = map(wall_boiling_patches(wb)) do name
        idx = boundary_index(boundaries_cpu, name)
        idx === nothing && throw(ArgumentError(
            """Wall boiling patch `:$name` is not a boundary of this mesh. Available \
boundaries: $(join(string.(getfield.(boundaries_cpu, :name)), ", "))."""))

        bc = _find_boundary_condition(boundaries.T, idx)
        bc === nothing && throw(ArgumentError(
            """No `T` boundary condition was assigned to wall boiling patch `:$name`."""))

        bc isa FixedHeatFlux || throw(ArgumentError(
            """Wall boiling patch `:$name` carries a $(typeof(bc).name.wrapper) condition on \
`T`, but `RPI` inverts a prescribed wall heat flux to find the wall temperature. Use \
`FixedHeatFlux(:$name, q)`.

(A fixed-temperature wall would need the forward partition instead, which is a \
different - and currently unimplemented - coupling.)"""))

        bc
    end

    return WallBoilingState(
        wb,
        patch_BCs,
        ScalarField(mesh),      # mdot_wall  [kg/m^3/s], per cell
        FaceScalarField(mesh),  # u_tau      [m/s]
        FaceScalarField(mesh),  # T_wall     [K]
        FaceScalarField(mesh),  # q_evap     [W/m^2]
        FaceScalarField(mesh),  # q_quench   [W/m^2]
        FaceScalarField(mesh),  # q_conv     [W/m^2]
        FaceScalarField(mesh),  # dT_sup     [K]
        FaceScalarField(mesh),  # y_plus     [-]
        FaceScalarField(mesh),  # A_b        [-]
        FaceScalarField(mesh),  # mdot_area  [kg/m^2/s]
        # `T_liquid` and `h_conv` are the two INPUTS the wall temperature solve
        # keys off. Storing them alongside its outputs makes the inversion
        # checkable from the surface file alone: T_w must exceed T_l on every
        # face (both branches of `solve_wall_temperature` guarantee it), and
        # q_conv = h_c*(T_w - T_l)*(1 - A_b) must follow from them. Reconstructing
        # either one outside the kernel gets the time level wrong.
        FaceScalarField(mesh),  # T_liquid   [K]  near-wall liquid temperature
        FaceScalarField(mesh),  # h_conv     [W/m^2/K]
        # Departure diameter per face, which sets the bubbly-layer thickness
        # used by the Eqn (2112) expansion. Written by the partition pass and
        # read by the alpha_delta pass on the NEXT step.
        FaceScalarField(mesh),  # D_departure [m]
        FaceScalarField(mesh),  # K_dry       [-]  wall dryout area fraction
        FaceScalarField(mesh),  # alpha_delta [-]  bubbly-layer void it used
        FaceScalarField(mesh),  # alpha_delta_raw [-]  before tangential smoothing
        # Bubbly-layer stencil, built only when a `BubblyLayerAverage` measure is
        # in use. `nothing` otherwise, so a case that does not need it pays
        # neither the setup search nor the memory.
        # Face-to-face adjacency ON the patch, for wall-TANGENTIAL smoothing of
        # the dryout criterion. `nothing` unless `dryout_smoothing > 0`.
        build_wall_face_graph(wb, patch_BCs, mesh),
    )
end

_find_boundary_condition(BCs, idx) = begin
    for bc in BCs
        bc.ID == idx && return bc
    end
    nothing
end


# =============================================================================
#  Per-step evaluation
# =============================================================================

"""
    wall_boiling_source!(wbs, model, p_abs, sat, h_fg, g_mag, sigma, config)

Evaluate the RPI partition on every heated wall face and accumulate the
resulting vapour generation into `wbs.mdot_wall` [kg/m^3/s].

`mdot_wall` is zeroed first, so it always reflects the current step alone.

Returns the field, or `nothing` when wall boiling is not active.
"""
wall_boiling_source!(::Nothing, model, p_abs, sat, h_fg, g_mag, sigma, dt, config) = nothing

"""
    WallFaceGraph

Face-to-face adjacency ON the heated patch: which wall faces share an edge with
which. Setup only - the mesh does not move.

### Why the wall closure needs this

Every other coupling in the wall model is wall-NORMAL. `T_wall` is solved by an
independent bisection per face, `alpha_delta` extrapolates along the wall normal,
and `K_dry` is a pointwise function of it. Nothing whatsoever ties a face to its
neighbours ALONG the surface.

That missing tangential coupling has produced the same symptom three times on the
LH2 pipe: 54% azimuthal scatter in the first-cell void at fixed z and r; seven
isolated faces at full dryout sitting beside neighbours at zero; and a jagged
`K_dry` front. A steep closure on a per-face field with no lateral communication
lets neighbouring faces settle on different branches, and nothing pulls them back.

Averaging over a wall-normal layer does NOT help - a layer average and the
Eqn (2112) expansion both smooth in the direction the jaggedness is not in.

### Edge adjacency, not distance

Two faces are neighbours when they share at least TWO nodes, i.e. an edge. That
is exact, needs no length scale, and cannot accidentally connect across a gap the
way a centre-distance criterion can on a curved or graded surface.
"""
struct WallFaceGraph{VI}
    nbr::VI         # flattened neighbour list, local face indices
    lo::VI
    hi::VI
    faceID::VI      # global face ID for each local index
end

build_wall_face_graph(::Nothing, patch_BCs, mesh) = nothing

function build_wall_face_graph(rpi::RPI, patch_BCs, mesh)
    rpi.dryout_smoothing > 0 || return nothing
    faces = Array(mesh.faces)
    face_nodes = Array(mesh.face_nodes)

    fIDs = Int[]
    for BC in patch_BCs
        append!(fIDs, collect(BC.IDs_range))
    end
    isempty(fIDs) && return nothing

    # node -> local faces touching it
    touching = Dict{Int,Vector{Int}}()
    for (li, fID) in enumerate(fIDs)
        for nID in face_nodes[faces[fID].nodes_range]
            push!(get!(touching, nID, Int[]), li)
        end
    end

    nbr = Int[]; lo = Int[]; hi = Int[]
    shared = Dict{Int,Int}()
    for li in eachindex(fIDs)
        empty!(shared)
        for nID in face_nodes[faces[fIDs[li]].nodes_range]
            for lj in touching[nID]
                lj == li && continue
                shared[lj] = get(shared, lj, 0) + 1
            end
        end
        push!(lo, length(nbr) + 1)
        for (lj, count) in shared
            count >= 2 && push!(nbr, lj)      # >= 2 shared nodes == shares an edge
        end
        push!(hi, length(nbr))
    end

    @info("Wall face graph built", faces = length(fIDs),
          mean_neighbours = round(length(nbr)/length(fIDs), digits=2),
          passes = rpi.dryout_smoothing)
    return WallFaceGraph(nbr, lo, hi, copy(fIDs))
end

"""
    smooth_wall_face_field!(dst, src, graph, passes, weight)

`passes` Laplacian smoothing sweeps over the wall face graph,

    a_i <- (1 - w)*a_i + w*mean(a_j over edge neighbours)

which damps the odd-even face-to-face mode while leaving the smooth variation
along the heater essentially untouched. `dst` and `src` are face fields; a face
with no neighbours is copied through unchanged.
"""
smooth_wall_face_field!(dst, src, ::Nothing, passes, weight, relax, kind, hyst) = nothing

function smooth_wall_face_field!(dst, src, g::WallFaceGraph, passes, weight, relax,
                                 kind::Symbol = :median, hyst = nothing)
    n = length(g.faceID)
    T = eltype(dst.values)
    w = T(weight)

    # Gathered onto the host and written back in one shot. The sweep is an
    # irregular gather over a face graph of ~1e3 entries - far too small to be
    # worth a device kernel, and elementwise indexing of a device array would be
    # a scalar-indexing error rather than merely slow.
    srcv = Array(src.values)
    dstv = Array(dst.values)
    buf = Vector{T}(undef, n)
    cur = Vector{T}(undef, n)
    @inbounds for li in 1:n
        cur[li] = srcv[g.faceID[li]]
    end
    stencil = Vector{T}(undef, 16)   # face + neighbours; 3.81 typical, 16 is ample
    @inbounds for _ in 1:passes
        for li in 1:n
            lo, hi = g.lo[li], g.hi[li]
            if hi < lo
                buf[li] = cur[li]          # isolated face: nothing to combine with
            elseif kind === :median
                # MEDIAN of {self} U {neighbours}. Removes a lone outlier face
                # while leaving a coherent front intact, because the result is an
                # actual neighbour value, not an average - see `dryout_filter`.
                m = 1
                stencil[1] = cur[li]
                for k in lo:hi
                    m += 1
                    m > length(stencil) && break
                    stencil[m] = cur[g.nbr[k]]
                end
                sort!(view(stencil, 1:m))
                buf[li] = isodd(m) ? stencil[(m + 1) >> 1] :
                          T(0.5)*(stencil[m >> 1] + stencil[(m >> 1) + 1])
            else
                acc = zero(T)
                for k in lo:hi
                    acc += cur[g.nbr[k]]
                end
                buf[li] = (one(T) - w)*cur[li] + w*acc/(hi - lo + 1)
            end
        end
        copyto!(cur, buf)
    end
    # Blend into the PREVIOUS value rather than overwriting it. `relax = 1` is a
    # plain overwrite; below 1 this is a first-order low-pass in time, which is
    # what limits the dryout loop gain - see `dryout_relaxation` on `RPI`.
    # Relaxation, then the PLAY OPERATOR against the previous state. `dstv[f]`
    # carries `xi` from the last step, so no extra field is needed - the raw
    # instantaneous value lives in `src`, the played state in `dst`. With
    # `hyst = nothing` `play_update` is the identity and this is a plain write.
    r = T(relax)
    @inbounds for li in 1:n
        f = g.faceID[li]
        target = (one(T) - r)*dstv[f] + r*cur[li]
        dstv[f] = play_update(hyst, dstv[f], target)
    end
    copyto!(dst.values, dstv)
    return nothing
end


function wall_boiling_source!(
    wbs::WallBoilingState, model, p_abs, sat, h_fg, g_mag, sigma, dt, config;
    alpha_liq = model.fluid.alpha, grad_alpha_v = nothing, void_sign = 1.0)

    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = model.domain
    (; faces, cells, boundary_cellsID) = mesh

    # LIQUID/VAPOUR, not tracked/other. Every RPI closure here is written in terms
    # of the liquid carrier and the vapour it generates, so these roles are
    # physical and must not follow whichever phase `alpha` happens to track - see
    # `multiphase_liquid_phase`. For the same reason the kernels below read
    # `alpha_liq`, the LIQUID fraction, which the caller supplies (it is
    # `model.fluid.alpha` itself only when `alpha` tracks the liquid).
    phases = model.fluid.phases
    liq = multiphase_liquid_phase(model.fluid)
    vap = 3 - liq
    phase_l = phases[liq]
    phase_v = phases[vap]

    fill!(wbs.mdot_wall.values, zero(eltype(wbs.mdot_wall.values)))

    for BC in wbs.patch_BCs
        facesID_range = BC.IDs_range
        start_ID = facesID_range[1]
        ndrange = length(facesID_range)

        # Friction velocity first, in its own pass, so the boiling kernel stays
        # independent of which turbulence closure is in use.
        wall_friction_velocity!(
            wbs.u_tau, model.turbulence, model, faces, boundary_cellsID,
            start_ID, ndrange, phase_l, config;
            method = wbs.model.friction_velocity)

        # Bubbly-layer void next, also in its own pass, so the tangential
        # smoothing below can see the whole patch before the partition reads it.
        vkernel! = _bubbly_layer_void!(_setup(backend, workgroup, ndrange)...)
        vkernel!(wbs.alpha_delta_raw, wbs.u_tau, wbs.D_departure, grad_alpha_v,
                 void_sign, wbs.model, faces, boundary_cellsID, start_ID,
                 alpha_liq, phase_v.rho, phase_v.mu)
        KernelAbstractions.synchronize(backend)

        # WALL-TANGENTIAL SMOOTHING. Every other coupling in this closure is
        # wall-NORMAL, so nothing tied a face to the ones beside it - see
        # `build_wall_face_graph`. A no-op when `dryout_smoothing = 0`, in which
        # case the raw field is copied straight through.
        # TEMPORAL RELAXATION applies on BOTH paths, so it is usable with the
        # tangential smoothing off. The two are independent: smoothing damps the
        # face-to-face (spatial) mode, relaxation damps the step-to-step
        # (temporal) one that drives the dryout feedback unstable.
        relax = wbs.model.dryout_relaxation
        hyst  = wbs.model.hysteresis
        if wbs.face_graph === nothing
            relax_wall_face_field!(wbs.alpha_delta, wbs.alpha_delta_raw,
                                   wbs.patch_BCs, relax, hyst)
        else
            smooth_wall_face_field!(wbs.alpha_delta, wbs.alpha_delta_raw,
                                    wbs.face_graph, wbs.model.dryout_smoothing,
                                    wbs.model.dryout_smoothing_weight, relax,
                                    wbs.model.dryout_filter, hyst)
        end

        kernel! = _wall_boiling_source!(_setup(backend, workgroup, ndrange)...)
        kernel!(
            wbs.mdot_wall, wbs.u_tau, wbs.T_wall, wbs.q_evap, wbs.q_quench, wbs.q_conv,
            wbs.dT_sup, wbs.y_plus, wbs.A_b, wbs.mdot_area, wbs.T_liquid, wbs.h_conv,
            wbs.D_departure,
            wbs.K_dry, wbs.alpha_delta, wbs.alpha_delta_raw, grad_alpha_v, void_sign,
            wbs.model, BC.value, dt, faces, cells, boundary_cellsID, start_ID,
            alpha_liq, model.energy.T, p_abs,
            phase_l.rho, phase_l.cp, phase_l.k, phase_l.mu,
            phase_v.rho, phase_v.cp, phase_v.k, phase_v.mu,
            turbulent_ke(model.turbulence),
            sat, h_fg, g_mag, sigma)
    end

    # Layer-averaged void, if the transition wants it. Done AFTER the partition
    # pass because it needs `D_d`, which the pass writes - so the average used at
    # step n is built from the departure diameter of step n-1. That lag is
    # harmless: `D_d` varies smoothly and the alternative is an inner iteration
    # for a quantity that only sets a blend width.

    return wbs.mdot_wall
end

"""
    relax_wall_face_field!(dst, src, patch_BCs, relax)

`dst <- (1-relax)*dst + relax*src` over the heated patches. The no-smoothing
path: `relax = 1` is a plain copy, below 1 a first-order low-pass in time.
"""
function relax_wall_face_field!(dst, src, patch_BCs, relax, hyst = nothing)
    T = eltype(dst.values)
    r = T(relax)
    if r == one(T) && hyst === nothing
        copyto!(dst.values, src.values)
        return nothing
    end
    d = Array(dst.values); v = Array(src.values)
    for BC in patch_BCs, f in BC.IDs_range
        @inbounds begin
            target = (one(T) - r)*d[f] + r*v[f]
            d[f] = play_update(hyst, d[f], target)
        end
    end
    copyto!(dst.values, d)
    return nothing
end

"""
    update_bubbly_layer_void!(wbs, alpha, grad_alpha_v, void_sign, phase_v,
                              facesID_range, config)

Fill `alpha_delta` - the vapour fraction over the bubbly layer that the dryout
criterion tests - for every face of one heated patch, then apply the optional
wall-tangential smoothing.

### Why this is a separate pass

It was originally computed inline in the partition kernel, which is fine as long
as nothing needs to look sideways. Smoothing does: a work item sees one face, and
a Laplacian sweep needs the whole patch. Splitting the pass keeps the value at the
CURRENT step while still allowing the sweep between the two.

Reading the previous step's field instead - the cheap alternative, and the one
tried first - is NOT viable. `alpha_delta` sets the mixture properties behind
`h_c` under the `:mmp` partition, so it sits inside the closed loop

    alpha_delta -> rho_m, k_m, mu_m, cp_m -> h_c -> T_wall -> q_evap -> alpha

and lagging a term inside that loop by a step took the LH2 pipe to NaN in 60
steps. The same split is already used for `u_tau`, for the same structural
reason.

`D_dep_f` remains from the previous pass; that lag is real but benign, and
predates this - it only sets the layer THICKNESS, not the void inside it.
"""
@kernel inbounds=true function _bubbly_layer_void!(
    alpha_delta_raw_f, u_tau_f, D_dep_f, grad_alpha_v, void_sign, rpi,
    faces, boundary_cellsID, start_ID, alpha, rho_v_f, mu_v_f)

    i = @index(Global)
    fID = i + start_ID - 1
    cID = boundary_cellsID[fID]
    face = faces[fID]
    (; delta, normal) = face
    TF = eltype(alpha_delta_raw_f.values)

    # VAPOUR FRACTION OVER THE BUBBLY LAYER, for the dryout criterion.
    #
    # STAR-CCM+ User Guide Eqn (2112) - a ONE-TERM expansion about the wall cell
    # centre rather than an average over the cells inside the layer:
    #
    #     alpha_delta = alpha(y_c) + alpha'(y_c)*(delta/2 - y_c)
    #
    # The expansion is what makes this robust. A stencil average is undefined
    # whenever the layer is THINNER than the first cell - no cell qualifies - and
    # that is not hypothetical: with K-I at the measured 4 deg contact angle,
    # D_d = 12.5 um against a first cell centre of 27.9 um, so the stencil version
    # silently returned zero and the criterion could never fire. The expansion
    # simply extrapolates INWARD when delta/2 < y_c.
    #
    # `void_sign` carries the tracking convention: `grad_alpha_v` is the gradient
    # of whichever fraction `alpha` measures, so it needs negating when that is
    # the liquid.
    a_v_cell = one(TF) - alpha[cID]
    dadn = if grad_alpha_v === nothing
        zero(TF)
    else
        g = grad_alpha_v[cID]
        # `normal` points OUT of the domain on a boundary face, so the fluid-side
        # (into-the-flow) direction is -n.
        TF(void_sign)*(-(g[1]*normal[1] + g[2]*normal[2] + g[3]*normal[3]))
    end
    nu_v_local = mu_v_f[cID]/max(rho_v_f[cID], eps(TF))

    alpha_delta_raw_f[fID] = bubbly_layer_void(
        rpi.bubbly_layer, a_v_cell, dadn, delta, D_dep_f[fID], nu_v_local,
        u_tau_f[fID])
end

"""
    turbulent_ke(turbulence) -> field or ConstantScalar

Turbulent kinetic energy for [`MassBalanceLayer`](@ref)'s fluctuating velocity
`v' = c_vp*sqrt(k)`. A model that does not carry `k` returns a zero constant,
which `mass_balance_void` reads as "no turbulent transport information" and falls
back to the cell value rather than dividing by zero.
"""
turbulent_ke(t) = hasproperty(t, :k) ? t.k : ConstantScalar(0.0)

"""
    _wall_mixture_htc(rpi, a_l, y_plus, u_tau, rho_l, rho_v, cp_l, cp_v, k_l, k_v,
                      mu_l, mu_v, Pr_t)

Convective coefficient the wall sees at liquid fraction `a_l`.

Under `:mmp` the wall is in contact with the MIXTURE, so `k` and `mu` are
volume-weighted and `cp` is MASS-weighted - `cp` multiplies `rho` in
`h_c = rho cp u_tau/T+`, and the product must be the mixture's volumetric heat
capacity. Under `:kurul_podowski` the wall sees liquid only.
"""
@inline function _wall_mixture_htc(rpi, a_l::TF, y_plus, u_tau, rho_l, rho_v,
                                   cp_l, cp_v, k_l, k_v, mu_l, mu_v, Pr_t) where TF
    a_v = one(TF) - a_l
    mmp = rpi.partition === :mmp
    rho_m = mmp ? a_l*rho_l + a_v*rho_v : rho_l
    k_m   = mmp ? a_l*k_l   + a_v*k_v   : k_l
    mu_m  = mmp ? a_l*mu_l  + a_v*mu_v  : mu_l
    cp_m  = if mmp
        rc = a_l*rho_l*cp_l + a_v*rho_v*cp_v
        rho_m > zero(TF) ? rc/rho_m : cp_l
    else
        cp_l
    end
    return single_phase_htc(y_plus, u_tau, rho_m, cp_m, mu_m, k_m, Pr_t)
end

"""
    _wall_solve_coupled(layer, rpi, a_delta_in, a_v_cell, ...) -> (a_delta, h_c, T_w, part)

Wall temperature and heat partition at a self-consistent bubbly-layer void.

For every layer EXCEPT [`MassBalanceLayer`](@ref) this is a single pass at the
`alpha_delta` the pre-pass computed - identical to the previous behaviour.

For `MassBalanceLayer` the void and the partition are mutually dependent
(`alpha_bl -> h_c, K_dry -> solve -> q_E -> alpha_bl`), so it is closed here by a
fixed-point iteration with a FIXED count, which keeps the kernel branch-free.
The map is contracting in the normal case because the feedback is negative:
raising `alpha_bl` raises `K_dry`, which cuts `q_E`, which lowers `alpha_bl`.
"""
@inline function _wall_solve_coupled(::AbstractBubblyLayer, rpi, a_delta_in::TF,
        a_v_cell, y_plus, u_tau, k_turb, T_l, T_sat, rho_l, rho_v, cp_l, k_l,
        mu_l, cp_v, k_v, mu_v, sigma, h_fg, g_mag, q_w, T_w_prev, dt) where TF
    a_l = one(TF) - a_delta_in
    h_c = _wall_mixture_htc(rpi, a_l, y_plus, u_tau, rho_l, rho_v, cp_l, cp_v,
                            k_l, k_v, mu_l, mu_v, TF(rpi.Pr_t))
    st = BoilingState{TF}(T_l, T_l, T_sat, T_l - T_sat, T_sat - T_l,
        rho_l, rho_v, cp_l, k_l, mu_l, sigma, h_fg, g_mag, cp_v, k_v, mu_v, a_l)
    T_w, part = solve_wall_temperature_transient(rpi, st, q_w, h_c, T_w_prev, dt)
    return (a_delta_in, h_c, T_w, part)
end

@inline function _wall_solve_coupled(m::MassBalanceLayer, rpi, a_delta_in::TF,
        a_v_cell, y_plus, u_tau, k_turb, T_l, T_sat, rho_l, rho_v, cp_l, k_l,
        mu_l, cp_v, k_v, mu_v, sigma, h_fg, g_mag, q_w, T_w_prev, dt) where TF
    # Seed from the previous step's converged value, which is a far better guess
    # than the cell value once the layer is established.
    a_bl = clamp(a_delta_in, a_v_cell, one(TF))
    r = TF(m.relax)
    h_c = zero(TF); T_w = T_w_prev
    part = wall_heat_partition(rpi, BoilingState{TF}(T_l, T_l, T_sat, zero(TF),
        zero(TF), rho_l, rho_v, cp_l, k_l, mu_l, sigma, h_fg, g_mag,
        cp_v, k_v, mu_v, one(TF)), one(TF))
    # `inner` update passes, then ONE FINAL SOLVE at the converged value.
    #
    # The final solve is not optional. Without it the returned `part` is the one
    # computed at the PREVIOUS iterate while `a_bl` has already been updated, so
    # the pair is inconsistent whenever the iteration has not fully converged -
    # and near the dryout cliff it oscillates. Measured symptom: the solve ran at
    # a high `a_bl` (K_dry = 1, so q_e = 0), that zero drove the update back to
    # the bare cell value, and the mismatched pair was written out as
    # `alpha_delta = 0.35` alongside `q_evap = 0`, which is impossible for a
    # single consistent state.
    for _ in 1:m.inner
        a_l = one(TF) - a_bl
        h_c = _wall_mixture_htc(rpi, a_l, y_plus, u_tau, rho_l, rho_v, cp_l, cp_v,
                                k_l, k_v, mu_l, mu_v, TF(rpi.Pr_t))
        st = BoilingState{TF}(T_l, T_l, T_sat, T_l - T_sat, T_sat - T_l,
            rho_l, rho_v, cp_l, k_l, mu_l, sigma, h_fg, g_mag, cp_v, k_v, mu_v, a_l)
        _, p_it = solve_wall_temperature_transient(rpi, st, q_w, h_c, T_w_prev, dt)
        target = mass_balance_void(m, a_v_cell, p_it.q_e, rho_v, h_fg, k_turb)
        a_bl = (one(TF) - r)*a_bl + r*target
    end
    a_l = one(TF) - a_bl
    h_c = _wall_mixture_htc(rpi, a_l, y_plus, u_tau, rho_l, rho_v, cp_l, cp_v,
                            k_l, k_v, mu_l, mu_v, TF(rpi.Pr_t))
    st = BoilingState{TF}(T_l, T_l, T_sat, T_l - T_sat, T_sat - T_l,
        rho_l, rho_v, cp_l, k_l, mu_l, sigma, h_fg, g_mag, cp_v, k_v, mu_v, a_l)
    T_w, part = solve_wall_temperature_transient(rpi, st, q_w, h_c, T_w_prev, dt)
    return (a_bl, h_c, T_w, part)
end

@kernel inbounds=true function _wall_boiling_source!(
    mdot_wall, u_tau_f, T_wall, q_evap, q_quench, q_conv,
    dT_sup_f, y_plus_f, A_b_f, mdot_area_f, T_liquid_f, h_conv_f,
    D_dep_f,
    K_dry_f, alpha_delta_f, alpha_delta_raw_f, grad_alpha_v, void_sign,
    rpi, q_w, dt, faces, cells, boundary_cellsID, start_ID,
    alpha, T, p_abs, rho_l_f, cp_l_f, k_l_f, mu_l_f,
    rho_v_f, cp_v_f, k_v_f, mu_v_f, k_turb_f, sat, h_fg, g_mag, sigma)

    i = @index(Global)
    fID = i + start_ID - 1
    cID = boundary_cellsID[fID]

    face = faces[fID]
    (; area, delta, normal) = face
    volume = cells[cID].volume

    TF = eltype(mdot_wall.values)

    T_l = T[cID]
    p = p_abs[cID]
    T_sat = saturation_temperature(sat, p)

    rho_l = rho_l_f[cID]
    rho_v = rho_v_f[cID]
    cp_l = cp_l_f[cID]
    k_l = k_l_f[cID]
    mu_l = mu_l_f[cID]

    u_tau = u_tau_f[fID]
    nu_l = mu_l/rho_l
    y_plus = u_tau*delta/max(nu_l, eps(TF))

    # CONVECTIVE COEFFICIENT. Which fluid the wall sees depends on the partition:
    #
    #   :kurul_podowski  LIQUID properties. The classical assumption is that only
    #                    liquid touches the wall, and `q_c` is then taken over the
    #                    un-influenced fraction `(1 - A_b)`.
    #
    #   :mmp             MIXTURE properties, per STAR-CCM+: "there [are] convection
    #                    contributions from vapor and liquid, always the mixture in
    #                    contact with the wall". Volume-weighted k and mu, and a
    #                    MASS-weighted cp, because cp multiplies rho in
    #                    `h_c = rho cp u_tau/T+` and the product must be the
    #                    mixture's volumetric heat capacity.
    #
    # `y_plus` stays on the LIQUID viscosity in both cases: it is the same wall
    # distance the momentum treatment used, and rescaling it is the film model's
    # job (see `ForcedConvectionFilm`).
    # VAPOUR FRACTION OVER THE BUBBLY LAYER, for the dryout criterion.
    #
    # STAR-CCM+ User Guide Eqn (2112) - a ONE-TERM expansion about the wall cell
    # centre rather than an average over the cells inside the layer:
    #
    #     alpha_delta = alpha(y_c) + alpha'(y_c)*(delta/2 - y_c)
    #
    # The expansion is what makes this robust. A stencil average is undefined
    # whenever the layer is THINNER than the first cell - no cell qualifies - and
    # that is not hypothetical: with K-I at the measured 4 deg contact angle,
    # D_d = 12.5 um against a first cell centre of 27.9 um, so the stencil version
    # silently returned zero and the criterion could never fire. The expansion
    # simply extrapolates INWARD when delta/2 < y_c.
    #
    # `void_sign` carries the tracking convention: `grad_alpha_v` is the gradient
    # of whichever fraction `alpha` measures, so it needs negating when that is
    # the liquid.
    # `alpha_delta` is prepared by `update_bubbly_layer_void!` in its OWN pass
    # ahead of this one, for the same reason `u_tau` is: the optional tangential
    # smoothing needs every face's value at once, and a work item here sees only
    # its own. It is still the CURRENT step's value - the separate pass exists to
    # allow the smoothing, not to introduce a lag.
    #
    # It must not be recomputed inline. An earlier attempt read the PREVIOUS
    # step's field instead of splitting the pass, and the resulting lag drove the
    # alpha_delta -> mixture properties -> h_c -> T_wall -> evaporation -> alpha
    # loop to NaN inside 60 steps. Same-step is not an optimisation here.
    # COUPLED SOLVE. `a_delta` sets the mixture properties behind `h_c` AND
    # `K_dry`, so the wall solve depends on it; with `MassBalanceLayer` the
    # reverse is also true, since `alpha_bl` depends on the `q_E` the solve
    # produces. `_wall_solve_coupled` closes that loop by fixed-point iteration
    # rather than by lagging a step - see `MassBalanceLayer`. Every other layer
    # takes the single-pass branch and is unaffected.
    a_delta_in = alpha_delta_f[fID]
    a_delta, h_c, T_w, part = _wall_solve_coupled(
        rpi.bubbly_layer, rpi, a_delta_in, one(TF) - alpha[cID],
        y_plus, u_tau, k_turb_f[cID],
        T_l, T_sat, rho_l, rho_v, cp_l, k_l, mu_l,
        cp_v_f[cID], k_v_f[cID], mu_v_f[cID],
        TF(sigma), TF(h_fg), TF(g_mag), TF(q_w), T_wall[fID], TF(dt))
    a_l = one(TF) - a_delta

    # Record what the coupled solve settled on, so the dryout ramp, the output
    # and the next step all see the same value.
    alpha_delta_f[fID] = a_delta
    T_liquid_f[fID] = T_l
    h_conv_f[fID] = h_c

    # Ramp the source out as the near-wall liquid disappears: RPI has no
    # validity once the wall is not liquid-wetted (see the `RPI` docstring).
    #
    # UNDER `:mmp` THIS IS ALREADY DONE. `_partition_weights` applies
    # `(1 - K_dry)` to `q_e` and `q_q` INSIDE the wall-temperature inversion, so
    # applying it again here would square it - and, more importantly, the whole
    # point of the `:mmp` form is that dryout feeds back on `T_w`, which only
    # happens if the weighting is inside the solve. So the post-hoc factor is
    # unity there.
    factor = rpi.partition === :mmp ?
        one(TF) : wall_boiling_liquid_factor(rpi, alpha[cID])

    # Evaporative flux driving vapour generation, ramped out by `factor` as the
    # near-wall liquid runs out: once the wall cell holds no liquid there is
    # nothing there to evaporate. Past that point the wall flux enters the vapour
    # as SENSIBLE heat through the unchanged `FixedHeatFlux` condition, and
    # evaporation becomes the job of the BULK interfacial phase change model
    # rather than of a wall closure - so a case relying on wall boiling as its
    # only phase change source will superheat the near-wall vapour without ever
    # consuming the latent heat.
    q_evaporative = part.q_e

    # W/m^2 over the face -> kg/m^3/s in the owner cell. The same `h_fg` used
    # here is the one the energy equation's latent heat sink uses, so the energy
    # removed from the liquid is exactly `q_evaporative * area`.
    mdot_cell = factor*q_evaporative*area/(TF(h_fg)*volume)

    # A cell can own more than one wall face (a corner cell, or a boundary layer
    # cell on a curved wall), so the accumulation must be atomic.
    Atomix.@atomic mdot_wall.values[cID] += mdot_cell

    T_wall[fID] = T_w
    q_evap[fID] = factor*part.q_e
    q_quench[fID] = part.q_q
    q_conv[fID] = part.q_c
    # `1 - K_dry` is what multiplies q_evap/q_quench in the :mmp partition.
    K_dry_f[fID] = one(TF) - wall_boiling_liquid_factor(rpi, a_l)
    # Departure diameter for the NEXT pass's bubbly-layer thickness (Eqn 2112).
    D_dep_f[fID] = part.D_d

    # Diagnostics. The flux partition IS the model, so a run that cannot show
    # how the wall flux was split cannot be assessed - and `dT_sup` in
    # particular is what the paper's boiling curve is plotted against.
    dT_sup_f[fID] = T_w - T_sat
    y_plus_f[fID] = y_plus
    A_b_f[fID] = part.A_b
    mdot_area_f[fID] = factor*q_evaporative/TF(h_fg)  # kg/m^2/s at the wall
end

"""
    wall_friction_velocity!(u_tau_f, turbulence, model, faces, boundary_cellsID,
                            start_ID, ndrange, phase_l, config)

Fill the wall friction velocity `u_tau` on one boundary patch.

Two cases, chosen on the host so the boiling kernel does not have to branch:

- **turbulent** (`k` available): the wall-function form `u_tau = Cmu^(1/4) sqrt(k)`,
  which is the standard equilibrium estimate and the one consistent with the
  log-law used by [`single_phase_htc`](@ref);
- **laminar**: from the wall shear stress directly,
  `u_tau = sqrt(nu * |U_P| / y_P)`.

The pipe cases this was written for are meshed for `y+` of 30-50 and run with a
`k`-based closure, so the first branch is the one in use; the second exists so
that a laminar sanity run does not fall over.
"""
function wall_friction_velocity!(
    u_tau_f, turbulence, model, faces, boundary_cellsID, start_ID, ndrange, phase_l, config;
    method = :k)

    (; hardware) = config
    (; backend, workgroup) = hardware

    if method === :loglaw
        kernel! = _u_tau_loglaw!(_setup(backend, workgroup, ndrange)...)
        kernel!(u_tau_f, model.momentum.U, phase_l.rho, phase_l.mu,
                faces, boundary_cellsID, start_ID)
    elseif hasproperty(turbulence, :k) && !(turbulence.k isa ConstantScalar)
        kernel! = _u_tau_from_k!(_setup(backend, workgroup, ndrange)...)
        kernel!(u_tau_f, turbulence.k, boundary_cellsID, start_ID)
    else
        kernel! = _u_tau_laminar!(_setup(backend, workgroup, ndrange)...)
        kernel!(u_tau_f, model.momentum.U, phase_l.rho, phase_l.mu,
                faces, boundary_cellsID, start_ID)
    end
    return nothing
end

"""
Friction velocity from the LOG LAW, by Newton iteration on

    |U_t| = (u_tau/kappa)*ln(E*u_tau*delta/nu)

the same equation the `nut` wall function solves in `RANS_functions.jl`.

### Why this exists alongside the k-based form

`u_tau = Cmu^0.25*sqrt(k)` assumes LOCAL EQUILIBRIUM - production balancing
dissipation in the near-wall cell. A boiling wall is exactly where that fails:
vapour generation, the latent sink and the density change all disturb the
near-wall balance, so the k-based estimate degrades precisely where the boiling
model leans on it hardest. `h_c = rho*cp*u_tau/T+` is linear in `u_tau`, so the
error passes straight into the convective share of the RPI partition.

The log-law form is driven by VELOCITY and does not care whether `k` has
equilibrated. It is also what the momentum wall treatment already uses, so
choosing it makes `h_c` and the wall shear derive from the same friction
velocity rather than two that can disagree.

### The duplication, stated plainly

This repeats the Newton solve rather than sharing it. The turbulence model
computes the same quantity inside its `nut` wall-function kernel and discards it -
it is used locally and never stored. Sharing would mean restructuring that kernel
to write a field the boiling model can read, which is the better fix and a larger
one. Until then these two solves must be kept in step by hand.
"""
@kernel inbounds=true function _u_tau_loglaw!(
    u_tau_f, U, rho_l_f, mu_l_f, faces, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1
    cID = boundary_cellsID[fID]
    TF = eltype(u_tau_f.values)

    face = faces[fID]
    (; delta, normal) = face
    nuc = mu_l_f[cID]/rho_l_f[cID]

    # Wall-tangential velocity at the cell centre (no-slip wall, so U_wall = 0).
    Uc = U[cID]
    Un = Uc[1]*normal[1] + Uc[2]*normal[2] + Uc[3]*normal[3]
    ut1 = Uc[1] - Un*normal[1]
    ut2 = Uc[2] - Un*normal[2]
    ut3 = Uc[3] - Un*normal[3]
    U_t = sqrt(ut1*ut1 + ut2*ut2 + ut3*ut3)

    kappa = TF(0.41)
    E = TF(9.8)

    # Viscous-sublayer initial guess, then 10 Newton steps. Fixed count keeps the
    # kernel branch-free for GPU execution; the residual is well converged by then
    # for any y+ this model is valid at.
    u_tau = sqrt(nuc*U_t/max(delta, eps(TF)) + TF(1e-20))
    for _ in 1:10
        yp = u_tau*delta/nuc
        lv = log(max(E*yp, TF(1.0 + 1e-4)))
        f  = U_t*kappa - u_tau*lv
        df = -(lv + one(TF))
        u_tau = max(u_tau - f/df, TF(1e-20))
    end
    u_tau_f[fID] = u_tau
end

@kernel inbounds=true function _u_tau_from_k!(u_tau_f, k, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1
    cID = boundary_cellsID[fID]
    TF = eltype(u_tau_f.values)
    # Cmu^(1/4) with Cmu = 0.09.
    u_tau_f[fID] = TF(0.09)^TF(0.25)*sqrt(max(k[cID], zero(TF)))
end

@kernel inbounds=true function _u_tau_laminar!(
    u_tau_f, U, rho_l, mu_l, faces, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1
    cID = boundary_cellsID[fID]
    TF = eltype(u_tau_f.values)

    (; delta) = faces[fID]
    Umag = sqrt(U.x[cID]^2 + U.y[cID]^2 + U.z[cID]^2)
    nu = mu_l[cID]/rho_l[cID]
    u_tau_f[fID] = sqrt(nu*Umag/max(delta, eps(TF)))
end


# =============================================================================
#  Surface output
# =============================================================================

"""
    write_wall_boiling_surface(wbs, mesh, iteration, time; prefix="wallBoiling")

Write the heated wall patches as a **surface** `.vtu`, one polygon per boundary
face, carrying the RPI flux partition as cell data.

Written as a separate surface file rather than folded into the volume output
because every quantity here lives on a boundary FACE and has no cell-centred
counterpart: `T_wall` is the solved wall temperature, not the near-wall cell
temperature, and the three fluxes are per unit WALL area.

Fields written:

| name | units | meaning |
|---|---|---|
| `T_wall` | K | wall temperature from the inverted partition |
| `dT_sup` | K | wall superheat, `T_wall - T_sat` |
| `q_conv`, `q_quench`, `q_evap` | W/m^2 | the three RPI components |
| `q_total` | W/m^2 | their sum - should equal the imposed flux |
| `evap_fraction` | - | `q_evap/q_total`, the share generating vapour |
| `mdot_area` | kg/m^2/s | evaporative mass flux at the wall |
| `A_b` | - | bubble influence area fraction |
| `y_plus`, `u_tau` | -, m/s | what the thermal wall function saw |

`q_total` is worth checking first: it must reproduce the `FixedHeatFlux` value,
and any departure means the wall temperature solve did not converge.

`K_dry` is the one to look at on a case near departure - it localises dryout on
the surface, and a run intended to stay in nucleate boiling should show it
identically zero.
"""

function write_wall_boiling_surface(
    wbs::WallBoilingState, mesh, iteration, time; prefix="wallBoiling")

    # `Array` is a no-op on CPU and copies to the host on GPU, which is what is
    # wanted either way for file output.
    nodes = Array(mesh.nodes)
    faces = Array(mesh.faces)
    face_nodes = Array(mesh.face_nodes)

    # Face IDs of every patch the model is applied to.
    fIDs = Int[]
    for BC in wbs.patch_BCs
        append!(fIDs, collect(BC.IDs_range))
    end
    isempty(fIDs) && return nothing

    # Compact the point list: only the nodes these faces actually use, remapped
    # to a local numbering. Writing the whole mesh's nodes would work but makes
    # a surface file as large as the volume one.
    local_id = Dict{Int,Int}()
    coords = Vector{eltype(nodes)}()
    for fID in fIDs, nID in face_nodes[faces[fID].nodes_range]
        if !haskey(local_id, nID)
            push!(coords, nodes[nID])
            local_id[nID] = length(coords)
        end
    end

    get_face(f) = Array(getfield(wbs, f).values)
    T_wall = get_face(:T_wall); dT_sup = get_face(:dT_sup)
    q_conv = get_face(:q_conv); q_quench = get_face(:q_quench)
    q_evap = get_face(:q_evap); A_b = get_face(:A_b)
    y_plus = get_face(:y_plus); u_tau = get_face(:u_tau)
    mdot_area = get_face(:mdot_area)
    K_dry = get_face(:K_dry); alpha_delta = get_face(:alpha_delta)
    alpha_delta_raw = get_face(:alpha_delta_raw)

    filename = "$(prefix)_$(iteration).vtu"
    open(filename, "w") do io
        println(io, """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">
 <UnstructuredGrid>
  <FieldData>
   <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1" format="ascii">
    $(time)
   </DataArray>
  </FieldData>
  <Piece NumberOfPoints="$(length(coords))" NumberOfCells="$(length(fIDs))">
   <Points>
    <DataArray type="Float64" NumberOfComponents="3" format="ascii">""")
        for c in coords
            println(io, "     $(c.coords[1]) $(c.coords[2]) $(c.coords[3])")
        end
        println(io, """    </DataArray>
   </Points>
   <Cells>
    <DataArray type="Int64" Name="connectivity" format="ascii">""")
        for fID in fIDs
            ids = [local_id[n] - 1 for n in face_nodes[faces[fID].nodes_range]]
            println(io, "     $(join(ids, " "))")
        end
        println(io, """    </DataArray>
    <DataArray type="Int64" Name="offsets" format="ascii">""")
        offset = 0
        for fID in fIDs
            offset += length(faces[fID].nodes_range)
            println(io, "     $offset")
        end
        println(io, """    </DataArray>
    <DataArray type="UInt8" Name="types" format="ascii">""")
        # VTK_POLYGON = 7: boundary faces may have any number of nodes.
        for _ in fIDs
            println(io, "     7")
        end
        println(io, """    </DataArray>
   </Cells>
   <CellData>""")

        q_total = [q_conv[f] + q_quench[f] + q_evap[f] for f in fIDs]
        evap_frac = [q_total[i] > 0 ? q_evap[f]/q_total[i] : 0.0
                     for (i, f) in enumerate(fIDs)]

        for (name, data) in (
            ("T_wall",        [T_wall[f] for f in fIDs]),
            ("dT_sup",        [dT_sup[f] for f in fIDs]),
            ("q_conv",        [q_conv[f] for f in fIDs]),
            ("q_quench",      [q_quench[f] for f in fIDs]),
            ("q_evap",        [q_evap[f] for f in fIDs]),
            ("K_dry",         [K_dry[f] for f in fIDs]),
            ("alpha_delta",   [alpha_delta[f] for f in fIDs]),
            # Pre-smoothing value alongside the smoothed one, so the effect of
            # `dryout_smoothing` is visible rather than inferred.
            ("alpha_delta_raw", [alpha_delta_raw[f] for f in fIDs]),
            ("q_total",       q_total),
            ("evap_fraction", evap_frac),
            ("mdot_area",     [mdot_area[f] for f in fIDs]),
            ("A_b",           [A_b[f] for f in fIDs]),
            ("y_plus",        [y_plus[f] for f in fIDs]),
            ("u_tau",         [u_tau[f] for f in fIDs]),
        )
            println(io, "    <DataArray type=\"Float64\" Name=\"$name\" format=\"ascii\">")
            for v in data
                println(io, "     $v")
            end
            println(io, "    </DataArray>")
        end

        println(io, """   </CellData>
  </Piece>
 </UnstructuredGrid>
</VTKFile>""")
    end
    return filename
end

write_wall_boiling_surface(::Nothing, mesh, iteration, time; kwargs...) = nothing


"""
    add_wall_boiling_rate!(mdot_pc, mdot_wall, config)

Add the wall vapour generation rate into the total phase change rate.

Summing into the same field the bulk model writes is what makes the coupling
work with no further changes: `mdot_pc` is already applied to the volume
fraction equation, to the pressure equation's volume creation, and to the energy
equation's latent heat - so wall boiling inherits all three at once, and
switching either mechanism off leaves the other intact.
"""
add_wall_boiling_rate!(mdot_pc, ::Nothing, config) = nothing
add_wall_boiling_rate!(::Nothing, mdot_wall, config) = nothing

function add_wall_boiling_rate!(mdot_pc, mdot_wall, config)
    @. mdot_pc.values += mdot_wall.values
    return nothing
end

"""
    wall_boiling_report(wbs, mesh) -> NamedTuple

AREA-WEIGHTED averages of the RPI wall state over every heated patch, which is
what a boiling curve is plotted from.

Returns `(area, T_wall, T_liquid, dT_sup, q_conv, q_quench, q_evap, q_total,
q_applied, closure, A_b, mdot_area, evap_frac, y_plus, h_conv)`.

Area weighting matters: the wall faces of an O-grid are not equal in area, so a
plain mean over faces silently biases the result toward wherever the mesh is
finest.

### The two numbers to read first

- **`closure = q_total/q_applied`** means DIFFERENT things on the two wall
  solves, and reading it the wrong way wastes a lot of time.

  With `wall_capacity = 0` the steady inversion `solve_wall_temperature` forces
  the partition to sum to the applied flux, so closure must be 1 and anything
  else means the solve did not converge.

  With `wall_capacity > 0` the TRANSIENT balance is solved instead, and it is
  `(C/dt)*(T_w - T_w_prev) + q_total = q_applied`. So

      closure = 1 - C*(dT_w/dt)/q_applied

  and closure BELOW 1 is the expected signature of a wall that is still heating
  up - the deficit is the energy going into wall inertia rather than the fluid.
  It is a physical rate, not a numerical residual, and is largely INDEPENDENT of
  the time step (halving dt roughly halves `dT_w` per step too). Read the deficit
  as a map of how far each face is from thermal equilibrium. It should decay on
  the wall time constant, roughly `C/(q_applied/dT_sup)`; if it persists far
  beyond that, the flow conditions are still evolving, not the wall.
- **`evap_frac = q_evap/q_total`** is the evaporative share. Above 1 is
  impossible - it would mean evaporation removing more than the wall supplies -
  and indicates `q_conv`/`q_quench` have gone negative, which happens when the
  solve returns a wall temperature below the local liquid temperature.

`T_wall` versus `q_applied` is the comparison against Tatsumoto's boiling curve;
`dT_sup` is the wall superheat that every RPI sub-model keys off.

Returns `nothing` when wall boiling is not active.
"""
wall_boiling_report(::Nothing, mesh) = nothing

function wall_boiling_report(wbs::WallBoilingState, mesh)
    faces = Array(mesh.faces)
    T_w = Array(wbs.T_wall.values);    T_l = Array(wbs.T_liquid.values)
    dTs = Array(wbs.dT_sup.values);    qc  = Array(wbs.q_conv.values)
    qq  = Array(wbs.q_quench.values);  qe  = Array(wbs.q_evap.values)
    Ab  = Array(wbs.A_b.values);       md  = Array(wbs.mdot_area.values)
    yp  = Array(wbs.y_plus.values);    hc  = Array(wbs.h_conv.values)

    A = 0.0
    # T_w, T_l, dT_sup, q_c, q_q, q_e, A_b, mdot, y+, h_c
    acc = zeros(10)
    q_applied = 0.0
    for BC in wbs.patch_BCs
        for f in BC.IDs_range
            a = faces[f].area
            A += a
            q_applied += BC.value*a
            acc .+= a .* (T_w[f], T_l[f], dTs[f], qc[f], qq[f], qe[f],
                          Ab[f], md[f], yp[f], hc[f])
        end
    end
    A <= 0 && return nothing
    acc ./= A
    q_applied /= A
    q_total = acc[4] + acc[5] + acc[6]

    return (area=A, T_wall=acc[1], T_liquid=acc[2], dT_sup=acc[3],
            q_conv=acc[4], q_quench=acc[5], q_evap=acc[6],
            q_total=q_total, q_applied=q_applied,
            closure=q_total/max(abs(q_applied), eps()),
            evap_frac=q_evap_frac(acc[6], q_total),
            A_b=acc[7], mdot_area=acc[8], y_plus=acc[9], h_conv=acc[10])
end

q_evap_frac(qe, qt) = abs(qt) <= eps() ? zero(qe) : qe/qt

"""
    report_wall_boiling(wbs, mesh; iteration=nothing)

One-line `@info` summary of [`wall_boiling_report`](@ref). No-op when wall
boiling is inactive.
"""
report_wall_boiling(::Nothing, mesh; iteration=nothing) = nothing

"""
    LAST_WALL_REPORT[]

The most recent [`wall_boiling_report`](@ref), or `nothing`.

`WallBoilingState` is local to the solver, so a driver script that calls `run!`
repeatedly - a heat-flux staircase, say - has no other way to recover the wall
state at the end of each run. Stashing it here means the values can be collected
programmatically instead of transcribed out of the log by hand.

Overwritten at every write interval, so read it immediately after `run!`.
"""
const LAST_WALL_REPORT = Ref{Any}(nothing)

function report_wall_boiling(wbs::WallBoilingState, mesh; iteration=nothing)
    r = wall_boiling_report(wbs, mesh)
    LAST_WALL_REPORT[] = r
    r === nothing && return nothing
    lbl = iteration === nothing ? "" : "step $(lpad(iteration,6))  "
    @info(
        "$(lbl)wall boiling (area-averaged)",
        T_wall = r.T_wall, dT_sup = r.dT_sup,
        q_conv = r.q_conv, q_quench = r.q_quench, q_evap = r.q_evap,
        q_applied = r.q_applied, closure = r.closure, evap_frac = r.evap_frac,
        mdot_area = r.mdot_area, y_plus = r.y_plus,
    )
    return r
end
