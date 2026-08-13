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
struct WallBoilingState{M,B,S,F,L}
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
    q_film::F
    w_film::F
    alpha_wall::F
    D_departure::F
    layer::L
end

"""
    BubblyLayerStencil

Precomputed association between each heated wall face and the cells lying within
the bubbly layer in front of it, so the layer-averaged void fraction of
[`BubblyLayerAverage`](@ref) can be assembled with a contiguous per-face sum.

Built ONCE at setup - the mesh does not move - and stored flat rather than as a
vector of vectors so the runtime pass stays GPU friendly.

- `cells`  cell IDs, grouped by face
- `dist`   wall-normal distance of each cell centre [m], parallel to `cells`
- `vol`    cell volume [m^3], parallel to `cells`
- `range`  `(first, last)` index into the three arrays above, per face
- `faceID` the wall face each range belongs to

`dist` is carried per cell rather than filtered at build time because the layer
thickness `L = min(D_d, cap)` depends on the LOCAL departure diameter, which
changes every step. The stencil is therefore built out to `cap` and the runtime
sum selects `dist < L` from it.
"""
struct BubblyLayerStencil{VI,VF}
    cells::VI
    dist::VF
    vol::VF
    range_lo::VI
    range_hi::VI
    faceID::VI
end
Adapt.@adapt_structure BubblyLayerStencil

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
        # Film boiling branch. `w_film` is the diagnostic that matters: it says
        # where on the boiling curve each face sits, and a run that is nowhere
        # near DNB should show it identically zero.
        FaceScalarField(mesh),  # q_film     [W/m^2]
        FaceScalarField(mesh),  # w_film     [-]
        FaceScalarField(mesh),  # alpha_wall [-]  vapour fraction the transition saw
        # Departure diameter per face, which sets the bubbly layer thickness.
        # Written by the partition pass and read by the layer average.
        FaceScalarField(mesh),  # D_departure [m]
        # Bubbly-layer stencil, built only when a `BubblyLayerAverage` measure is
        # in use. `nothing` otherwise, so a case that does not need it pays
        # neither the setup search nor the memory.
        build_bubbly_layer_stencil(wb, patch_BCs, mesh),
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
    build_bubbly_layer_stencil(rpi, patch_BCs, mesh) -> BubblyLayerStencil or nothing

Associate each heated wall face with the cells in front of it, out to the
`BubblyLayerAverage` cap. Returns `nothing` unless a measure that needs it is in
use, so a case that does not want it pays neither the setup search nor the
memory.

### Method

The wall-normal distance from a face to a cell is the projection of the centre
offset onto the face's inward normal. A cell joins a face's layer when

    0 < d_normal < cap        and        |offset - d_normal*n| < r_face

with `r_face` the face's own radius, from its area. The second condition keeps
the layer a COLUMN in front of the face rather than a hemisphere around it -
without it a cell would be claimed by every face within `cap` and the average
would smear ALONG the wall as well as away from it, which is the opposite of
what a near-wall measure should do.

Cells that satisfy both for several faces go to the nearest, so each contributes
its volume exactly once and the result stays a true volume average.

### Cost

Setup only. The mesh does not move, so it is never rebuilt, and the runtime pass
is a contiguous per-face sum.
"""
build_bubbly_layer_stencil(::Nothing, patch_BCs, mesh) = nothing

function build_bubbly_layer_stencil(rpi::RPI, patch_BCs, mesh)
    fb = rpi.film_boiling
    (fb === nothing || !needs_layer_average(fb)) && return nothing
    cap = fb.transition.measure.cap

    faces = Array(mesh.faces)
    cells = Array(mesh.cells)

    fIDs = Int[]
    for BC in patch_BCs
        append!(fIDs, collect(BC.IDs_range))
    end
    isempty(fIDs) && return nothing

    TF = typeof(cells[1].volume)

    # Stored boundary normals point OUT of the domain, so the fluid side is -n.
    fc = [faces[f].centre for f in fIDs]
    fn = [-faces[f].normal for f in fIDs]
    fr = [sqrt(faces[f].area/pi) for f in fIDs]

    best_face = zeros(Int, length(cells))
    best_d    = fill(TF(Inf), length(cells))
    for i in eachindex(fIDs), cID in eachindex(cells)
        off = cells[cID].centre - fc[i]
        dn  = off ⋅ fn[i]
        (dn > 0 && dn < cap) || continue
        dtan = sqrt(max(off ⋅ off - dn*dn, zero(TF)))
        dtan < fr[i] || continue
        if dn < best_d[cID]
            best_d[cID] = dn
            best_face[cID] = i
        end
    end

    cellsv = Int[]; distv = TF[]; volv = TF[]
    lo = Int[]; hi = Int[]
    for i in eachindex(fIDs)
        push!(lo, length(cellsv) + 1)
        for cID in eachindex(cells)
            best_face[cID] == i || continue
            push!(cellsv, cID)
            push!(distv, best_d[cID])
            push!(volv, cells[cID].volume)
        end
        push!(hi, length(cellsv))
    end

    n_empty = count(i -> hi[i] < lo[i], eachindex(fIDs))
    n_empty == 0 || @warn """`BubblyLayerAverage` found no cells in front of \
$n_empty of $(length(fIDs)) wall faces. Those faces fall back to the wall-cell \
value, so the criterion is mesh dependent there. Usually means `cap` is below \
the first cell height."""

    @info("Bubbly layer stencil built",
          faces = length(fIDs), cap_mm = cap*1e3,
          cells = length(cellsv),
          mean_cells_per_face = round(length(cellsv)/length(fIDs), digits = 1))

    return BubblyLayerStencil(cellsv, distv, volv, lo, hi, copy(fIDs))
end

function wall_boiling_source!(
    wbs::WallBoilingState, model, p_abs, sat, h_fg, g_mag, sigma, dt, config)

    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = model.domain
    (; faces, cells, boundary_cellsID) = mesh

    phases = model.fluid.phases
    main = model.fluid.volume_fraction
    secondary = 3 - main
    phase_l = phases[main]
    phase_v = phases[secondary]

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

        kernel! = _wall_boiling_source!(_setup(backend, workgroup, ndrange)...)
        kernel!(
            wbs.mdot_wall, wbs.u_tau, wbs.T_wall, wbs.q_evap, wbs.q_quench, wbs.q_conv,
            wbs.dT_sup, wbs.y_plus, wbs.A_b, wbs.mdot_area, wbs.T_liquid, wbs.h_conv,
            wbs.q_film, wbs.w_film, wbs.alpha_wall, wbs.D_departure,
            wbs.model, BC.value, dt, faces, cells, boundary_cellsID, start_ID,
            model.fluid.alpha, model.energy.T, p_abs,
            phase_l.rho, phase_l.cp, phase_l.k, phase_l.mu,
            phase_v.rho, phase_v.cp, phase_v.k, phase_v.mu,
            sat, h_fg, g_mag, sigma)
    end

    # Layer-averaged void, if the transition wants it. Done AFTER the partition
    # pass because it needs `D_d`, which the pass writes - so the average used at
    # step n is built from the departure diameter of step n-1. That lag is
    # harmless: `D_d` varies smoothly and the alternative is an inner iteration
    # for a quantity that only sets a blend width.
    update_layer_void!(wbs, model.fluid.alpha, config)

    return wbs.mdot_wall
end

"""
    update_layer_void!(wbs, alpha, config)

Fill `alpha_wall` with the volume-averaged vapour fraction over the bubbly layer
in front of each heated wall face. No-op when the transition uses the wall-cell
value, in which case the kernel reads `alpha` directly.
"""
update_layer_void!(wbs, alpha, config) =
    _update_layer_void!(wbs, wbs.layer, alpha, config)

_update_layer_void!(wbs, ::Nothing, alpha, config) = nothing

function _update_layer_void!(wbs, layer::BubblyLayerStencil, alpha, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    ndrange = length(layer.faceID)
    kernel! = _layer_void_kernel!(_setup(backend, workgroup, ndrange)...)
    kernel!(wbs.alpha_wall, layer.cells, layer.dist, layer.vol,
            layer.range_lo, layer.range_hi, layer.faceID,
            alpha, wbs.D_departure)
end

@kernel inbounds=true function _layer_void_kernel!(
    alpha_wall, cellsv, distv, volv, lo, hi, faceID, alpha, D_dep)
    i = @index(Global)
    fID = faceID[i]
    TF = eltype(alpha_wall.values)

    # Layer thickness for THIS face: the local departure diameter, already
    # bounded by `cap` when the stencil was built.
    L = D_dep[fID]

    num = zero(TF); den = zero(TF)
    for k in lo[i]:hi[i]
        distv[k] < L || continue
        v = volv[k]
        num += v*(one(TF) - alpha[cellsv[k]])   # vapour fraction
        den += v
    end
    # No cells inside the layer (cap below the first cell height): fall back to
    # the wall cell, which is what `NearWallCell` would have given.
    alpha_wall[fID] = den > zero(TF) ? num/den : alpha_wall[fID]
end

@kernel inbounds=true function _wall_boiling_source!(
    mdot_wall, u_tau_f, T_wall, q_evap, q_quench, q_conv,
    dT_sup_f, y_plus_f, A_b_f, mdot_area_f, T_liquid_f, h_conv_f,
    q_film_f, w_film_f, alpha_wall_f, D_dep_f,
    rpi, q_w, dt, faces, cells, boundary_cellsID, start_ID,
    alpha, T, p_abs, rho_l_f, cp_l_f, k_l_f, mu_l_f,
    rho_v_f, cp_v_f, k_v_f, mu_v_f, sat, h_fg, g_mag, sigma)

    i = @index(Global)
    fID = i + start_ID - 1
    cID = boundary_cellsID[fID]

    face = faces[fID]
    (; area, delta) = face
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

    h_c = single_phase_htc(y_plus, u_tau, rho_l, cp_l, mu_l, k_l, TF(rpi.Pr_t))

    T_liquid_f[fID] = T_l
    h_conv_f[fID] = h_c

    state = BoilingState{TF}(
        T_l, T_l, T_sat, T_l - T_sat, T_sat - T_l,
        rho_l, rho_v, cp_l, k_l, mu_l, TF(sigma), TF(h_fg), TF(g_mag),
        cp_v_f[cID], k_v_f[cID], mu_v_f[cID])

    # Resolve the post-CHF blend for this face: the CHF superheat, the minimum
    # film boiling superheat and the film heat transfer coefficient. Done ONCE
    # here rather than inside the wall temperature iteration, since none of the
    # three depends on `T_w`. `nothing` when no film boiling model is attached,
    # in which case every film term below compiles out.
    fc = film_closure(rpi, rpi.film_boiling, state, h_c, y_plus, u_tau)

    # Vapour fraction the transition sees. `NearWallCell` reads the wall cell
    # directly; `BubblyLayerAverage` uses the layer average assembled after the
    # previous pass, which is already sitting in `alpha_wall_f`. Selecting here
    # rather than in the blend keeps the choice out of the bisection loop.
    alpha_v = wall_void_fraction(rpi.film_boiling, alpha[cID], alpha_wall_f[fID])

    # Transient wall energy balance when the model carries a wall capacity;
    # identical to the steady inversion when it does not. `T_wall` holds the
    # PREVIOUS step's value on entry and is overwritten below.
    T_w, part = solve_wall_temperature_transient(
        rpi, state, TF(q_w), h_c, T_wall[fID], TF(dt), fc, alpha_v)

    # Ramp the source out as the near-wall liquid disappears: RPI has no
    # validity once the wall is not liquid-wetted (see the `RPI` docstring).
    factor = wall_boiling_liquid_factor(rpi, alpha[cID])

    # Evaporative flux driving vapour generation. In the transition and film
    # regimes the heat crossing the vapour film evaporates liquid at the film
    # interface, so `q_f` belongs here alongside the nucleation term `q_e`.
    #
    # Both are ramped out together by `factor` as the near-wall liquid runs out.
    # That is deliberate: once the wall cell holds no liquid there is nothing
    # there to evaporate, and the interface has moved away from the wall. The
    # wall flux then enters the vapour as SENSIBLE heat through the unchanged
    # `FixedHeatFlux` condition, superheating the film, and evaporation at the
    # film-liquid interface becomes the job of the bulk interfacial phase change
    # model rather than of a wall closure. Fully established film boiling
    # therefore requires that model to be active - a case run with wall boiling
    # as the only phase change source will heat the film without ever consuming
    # the latent heat.
    q_evaporative = part.q_e + part.q_f

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
    q_film_f[fID] = factor*part.q_f
    w_film_f[fID] = part.w
    # Layer thickness for the NEXT pass's average, bounded by `cap`. Zero when
    # the measure does not use a layer, in which case nothing reads it.
    D_dep_f[fID] = void_layer_thickness_for(rpi.film_boiling, part.D_d)
    # `NearWallCell` records what it used, so the field means the same thing in
    # the output whichever measure is active.
    alpha_wall_f[fID] =
        recorded_void_fraction(rpi.film_boiling, alpha[cID], alpha_wall_f[fID])

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
| `q_film` | W/m^2 | the film boiling component, zero without [`FilmBoiling`](@ref) |
| `w_film` | - | film boiling blend weight: 0 nucleate, 1 fully blanketed |
| `q_total` | W/m^2 | their sum - should equal the imposed flux |
| `evap_fraction` | - | `q_evap/q_total`, the share generating vapour |
| `mdot_area` | kg/m^2/s | evaporative mass flux at the wall |
| `A_b` | - | bubble influence area fraction |
| `y_plus`, `u_tau` | -, m/s | what the thermal wall function saw |

`q_total` is worth checking first: it must reproduce the `FixedHeatFlux` value,
and any departure means the wall temperature solve did not converge.

`w_film` is the one to look at on a case near departure - it localises DNB on the
surface, and a run intended to stay in nucleate boiling should show it
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
    q_film = get_face(:q_film); w_film = get_face(:w_film)

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

        q_total = [q_conv[f] + q_quench[f] + q_evap[f] + q_film[f] for f in fIDs]
        evap_frac = [q_total[i] > 0 ? q_evap[f]/q_total[i] : 0.0
                     for (i, f) in enumerate(fIDs)]

        for (name, data) in (
            ("T_wall",        [T_wall[f] for f in fIDs]),
            ("dT_sup",        [dT_sup[f] for f in fIDs]),
            ("q_conv",        [q_conv[f] for f in fIDs]),
            ("q_quench",      [q_quench[f] for f in fIDs]),
            ("q_evap",        [q_evap[f] for f in fIDs]),
            ("q_film",        [q_film[f] for f in fIDs]),
            ("w_film",        [w_film[f] for f in fIDs]),
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

- **`closure = q_total/q_applied`** must be 1. The partition is constructed to
  sum to the applied flux, so anything else means the wall temperature solve did
  not converge.
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
    qf  = Array(wbs.q_film.values);    wf  = Array(wbs.w_film.values)

    A = 0.0
    # T_w, T_l, dT_sup, q_c, q_q, q_e, A_b, mdot, y+, h_c, q_f, w_film
    acc = zeros(12)
    q_applied = 0.0
    for BC in wbs.patch_BCs
        for f in BC.IDs_range
            a = faces[f].area
            A += a
            q_applied += BC.value*a
            acc .+= a .* (T_w[f], T_l[f], dTs[f], qc[f], qq[f], qe[f],
                          Ab[f], md[f], yp[f], hc[f], qf[f], wf[f])
        end
    end
    A <= 0 && return nothing
    acc ./= A
    q_applied /= A
    q_total = acc[4] + acc[5] + acc[6] + acc[11]

    return (area=A, T_wall=acc[1], T_liquid=acc[2], dT_sup=acc[3],
            q_conv=acc[4], q_quench=acc[5], q_evap=acc[6], q_film=acc[11],
            w_film=acc[12],
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
        q_film = r.q_film, w_film = r.w_film,
        q_applied = r.q_applied, closure = r.closure, evap_frac = r.evap_frac,
        mdot_area = r.mdot_area, y_plus = r.y_plus,
    )
    return r
end
