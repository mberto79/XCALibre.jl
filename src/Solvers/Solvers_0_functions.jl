export restart_fields!, restart_flux!
export flux!, update_nueff!, inverse_diagonal!, remove_pressure_source!, H!, correct_velocity!

## UPDATE EFFECTIVE VISCOSITY

function update_nueff!(nueff, nu, turb_model, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(nueff)
    if typeof(turb_model) <: Laminar
        kernel! = _sized(update_nueff_laminar!, backend, workgroup, ndrange)
        kernel!(nu, nueff)
    else
        (; nutf) = turb_model
        kernel! = _sized(update_nueff_turbulent!, backend, workgroup, ndrange)
        kernel!(nu, nutf, nueff)
    end

end

@kernel function update_nueff_laminar!(nu, nueff)
    i = @index(Global)

    @inbounds begin
        nueff[i] = nu[i]
    end
end

@kernel function update_nueff_turbulent!(nu, νtf, nueff)
    i = @index(Global)

    @inbounds begin
        nueff[i] = nu[i] + νtf[i]
    end
end


## FLUX CALCULATION

function flux!(phif::FS, psif::FV, config) where {FS<:FaceScalarField,FV<:FaceVectorField}
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(phif)
    kernel! = _sized(flux_kernel!, backend, workgroup, ndrange)
    kernel!(phif, psif)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function flux_kernel!(phif, psif)
    i = @index(Global)

    @uniform begin
        (; values) = phif
        (; faces) = psif.mesh
    end

    @inbounds begin
        (; area, normal) = faces[i]
        Sf = area * normal
        values[i] = psif[i] ⋅ Sf
    end
end

function flux!(phif::FS, psif::FV, rhof::FS, config) where {FS<:FaceScalarField,FV<:FaceVectorField}
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(phif)
    kernel! = _sized(_flux!, backend, workgroup, ndrange)
    kernel!(phif, psif, rhof)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _flux!(phif, psif, rhof)
    i = @index(Global)

    @uniform begin
        (; values) = phif
        (; faces) = psif.mesh
    end

    @inbounds begin
        (; area, normal) = faces[i]
        Sf = area * normal
        values[i] = (psif[i] ⋅ Sf) * rhof[i]
    end
end


volumes(mesh) = [mesh.cells[i].volume for i ∈ eachindex(mesh.cells)]

# INVERSE DIAGONAL CALCULATION

# SIMPLEC (simple!, csimple!): rDt = V/(a_P + Σ a_N) does not shrink with the velocity relaxation
# like rD = V/a_P; rows whose sum is not clearly positive keep rD; ghosts stale until the exchange
function consistent_inverse_diagonal!(rDt, rD, eqn, config; halo=true)
    (; backend, workgroup) = config.hardware
    nzval, colval, rowptr = get_sparse_fields(eqn.equation.A)
    ndrange = length(rDt)
    kernel! = _sized(_consistent_inverse_diagonal!, backend, workgroup, ndrange)
    kernel!(rDt, rD, nzval, rowptr, eqn.equation.diag_nz)
    halo && sync!(rDt, rDt.mesh, config) # self-syncing seam (no-op serial)
end

@kernel function _consistent_inverse_diagonal!(rDt, rD, nzval, rowptr, diag_nz)
    i = @index(Global)
    @uniform cells = rDt.mesh.cells
    @inbounds begin
        rowsum = zero(eltype(nzval))
        for k ∈ rowptr[i]:(rowptr[i+1] - 1)
            rowsum += nzval[k]
        end
        aP = nzval[diag_nz[i]]
        rDt[i] = rowsum > aP/1000 ? cells[i].volume/rowsum : rD[i]
    end
end

# Face part of the consistent correction, ρ_f (rDt - rD)_f (p_N - p_P) gDiff_f, with the face
# coefficient of the pressure Laplacian; boundary faces use the owner cell and the boundary
# pressure in pf. rhof = nothing for incompressible (volumetric) fluxes.
function consistent_flux!(mdotf, rhof, rDt, rD, p, pf, config)
    (; backend, workgroup) = config.hardware
    ndrange = length(mdotf)
    kernel! = _sized(_consistent_flux!, backend, workgroup, ndrange)
    kernel!(mdotf, rhof, rDt, rD, p, pf)
end

@kernel function _consistent_flux!(mdotf, rhof, rDt, rD, p, pf)
    fID = @index(Global)
    @uniform begin
        (; faces, boundary_cellsID, face_gDiff) = p.mesh
        n_bfaces = length(boundary_cellsID)
    end
    @inbounds begin
        (; ownerCells, weight) = faces[fID]
        rho = _face_rho(rhof, fID, weight)
        if fID <= n_bfaces
            c = boundary_cellsID[fID]
            mdotf[fID] += rho*(rDt[c] - rD[c])*(pf[fID] - p[c])*face_gDiff[fID]
        else
            c1, c2 = ownerCells
            drD = weight*(rDt[c1] - rD[c1]) + (one(weight) - weight)*(rDt[c2] - rD[c2])
            mdotf[fID] += rho*drD*(p[c2] - p[c1])*face_gDiff[fID]
        end
    end
end

@inline _face_rho(rhof, fID, w) = rhof[fID]
@inline _face_rho(::Nothing, fID, w) = one(w)

# Cell part of the consistent correction, H/a_P + (rDt - rD) ∇p, so that U = Hv - rDt ∇p.
function consistent_H!(Hv, rDt, rD, ∇p, config)
    gradp = ∇p.result
    xcal_foreach(rD, config) do i
        Hv[i] = Hv[i] + (rDt[i] - rD[i])*gradp[i]
    end
end

## TIME-CONSISTENT FACE FLUX

# The face flux interpolates H/a_P, whose old-time source rD·ρⁿUⁿ/Δt is a cell value; on interior
# faces it is moved towards rD_f·ṁⁿ/Δt from the face's own old flux, so that a steady state depends
# far less on Δt. Carried in full, the face memory makes the face/cell mismatch e = ṁⁿ - (ρU)ⁿ_f·S
# evolve as e ← (rD_f/Δt)·e, which grows where a_P < V/Δt (Linear convection on graded meshes).
# The correction is therefore weighted per face by θ_f·c_f: θ_f = min(1, Δt/rD_f) keeps that gain
# at most one, and c_f = 1 - min(|e|/|ṁⁿ|, 1), OpenFOAM's ddtCouplingCoeff, makes it decay.
struct TimeFlux{F,V,W,M,T}
    mold::F     # ṁⁿ (mass flux of the time term's quantity)
    uold::V     # ρⁿUⁿ
    w::V        # scratch: rD·ρⁿUⁿ/Δt
    wf::W
    rDf::F
    phi::F
    mask::M     # faces corrected (interior)
    dt::T       # [Δtⁿ]
end

# `nothing` for equations without a time derivative (steady solvers)
function TimeFlux(U_eqn, mesh, BCs)
    unwrap_eqn(U_eqn).model.terms[1].type isa Time{SteadyState} && return nothing
    nb = length(mesh.boundary_cellsID)
    mask = zeros(Bool, length(mesh.faces))
    mask[nb+1:end] .= true
    backend = _get_backend(mesh)
    TimeFlux(FaceScalarField(mesh), VectorField(mesh), VectorField(mesh), FaceVectorField(mesh),
        FaceScalarField(mesh), FaceScalarField(mesh), adapt(backend, mask), [one(_get_float(mesh))])
end

# old levels at the start of a step: face mass flux rhof*mdotf (`rhof` is `nothing` for a mass flux)
# and cell rho*U (`rho` is `nothing` for unit density)
begin_time_flux!(::Nothing, args...) = nothing
function begin_time_flux!(tf::TimeFlux, mdotf, U, rho, rhof, dt)
    isnothing(rhof) ? copyto!(tf.mold.values, mdotf.values) :
        (@. tf.mold.values = rhof.values*mdotf.values)
    r = isnothing(rho) ? one(eltype(U.x.values)) : rho.values
    @. tf.uold.x.values = r*U.x.values
    @. tf.uold.y.values = r*U.y.values
    @. tf.uold.z.values = r*U.z.values
    tf.dt[1] = dt
    nothing
end

# in the face flux of the interpolated H/a_P, the old-time source by its face value rD_f·ṁⁿ/Δt
time_flux!(::Nothing, args...) = nothing
function time_flux!(tf::TimeFlux, flux, rD, BCs, time, config)
    dt = eltype(flux.values)(tf.dt[1])
    (; w, wf, uold, rDf, phi, mold, mask) = tf
    @. w.x.values = rD.values*uold.x.values/dt
    @. w.y.values = rD.values*uold.y.values/dt
    @. w.z.values = rD.values*uold.z.values/dt
    interpolate!(wf, w, config)
    correct_boundaries!(wf, w, BCs, time, config)
    flux!(phi, wf, config)
    interpolate!(rDf, rD, config)
    correct_interpolation_periodic(rDf, rD, BCs, config)
    @. flux.values = ifelse(mask,
        flux.values + _time_flux_weight(rDf.values, mold.values, phi.values, dt)*
            (rDf.values*mold.values/dt - phi.values), flux.values)
    nothing
end

# θ_f·c_f for the face-minus-cell old source rD_f·ṁⁿ/Δt - φ_f, whose Δt/rD_f multiple is the
# mismatch e ≈ ṁⁿ - (ρU)ⁿ_f·S
@inline function _time_flux_weight(rDf, m, phi, dt)
    e = m - phi*dt/rDf
    θ = min(one(dt), dt/rDf)
    c = one(dt) - min(abs(e)/(abs(m) + eps(typeof(dt))^2), one(dt))
    θ*c
end

# ν_t and ν_eff of the current fields, for time schemes that extrapolate the explicit terms
explicit_terms(turbulenceModel, model, S, nueff, nu, time, config) = () -> begin
    eddy_viscosity!(turbulenceModel, model, S, time, config)
    update_nueff!(nueff, nu, model.turbulence, config)
end

# halo=false leaves ghosts stale for a caller that exchanges rD together with Hv
function inverse_diagonal!(rD::S, eqn, config; halo=true) where {S<:ScalarField}
    (; hardware) = config
    (; backend, workgroup) = hardware
    A = eqn.equation.A # Or should I use A0
    nzval, colval, rowptr = get_sparse_fields(A)

    ndrange = length(rD)
    kernel! = _sized(_inverse_diagonal!, backend, workgroup, ndrange)
    kernel!(rD, nzval, colval, rowptr, eqn.equation.diag_nz)
    # # KernelAbstractions.synchronize(backend)
    halo && sync!(rD, rD.mesh, config) # self-syncing seam (no-op serial)
end

@kernel function _inverse_diagonal!(rD, nzval, colval, rowptr, diag_nz)
    i = @index(Global)

    @uniform begin
        (; mesh, values) = rD
        cells = mesh.cells
    end

    @inbounds begin
        idx = diag_nz[i]
        D = nzval[idx]
        (; volume) = cells[i]
        values[i] = volume / D
    end
end

## VELOCITY CORRECTION

function correct_velocity!(U, Hv, ∇p, rD, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(U)
    kernel! = _sized(_correct_velocity!, backend, workgroup, ndrange)
    kernel!(U, Hv, ∇p, rD)
    # no sync!: ghost U is already consistent (Hv/∇p/rD ghosts synced, kernel is pointwise)
end

@kernel function _correct_velocity!(U, Hv, ∇p, rD)
    i = @index(Global)

    @uniform begin
        Ux, Uy, Uz = U.x, U.y, U.z
        Hvx, Hvy, Hvz = Hv.x, Hv.y, Hv.z
        dpdx, dpdy, dpdz = ∇p.result.x, ∇p.result.y, ∇p.result.z
        rDvalues = rD.values
    end

    @inbounds begin
        rDvalues_i = rDvalues[i]
        Ux[i] = Hvx[i] - dpdx[i] * rDvalues_i
        Uy[i] = Hvy[i] - dpdy[i] * rDvalues_i
        Uz[i] = Hvz[i] - dpdz[i] * rDvalues_i
    end
end

## PRESSURE CORRECTION AND SOURCE REMOVAL

remove_pressure_source!(U_eqn::ME, ∇p, config) where {ME} = begin # Extend to 3D
    # backend = _get_backend(get_phi(ux_eqn).mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    cells = get_phi(U_eqn).mesh.cells
    source_sign = get_source_sign(U_eqn, 1)
    (; bx, by, bz) = U_eqn.equation

    ndrange = length(bx)
    kernel! = _sized(_remove_pressure_source!, backend, workgroup, ndrange)
    kernel!(cells, source_sign, ∇p, bx, by, bz)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _remove_pressure_source!(cells, source_sign, ∇p, bx, by, bz) #Extend to 3D
    i = @index(Global)


    @inbounds begin
        (; volume) = cells[i]
        calc = source_sign*∇p[i]*volume
        bx[i] -= calc[1]
        by[i] -= calc[2]
        bz[i] -= calc[3]
    end
end

# Pressure correction
function H!(Hv, U::VF, U_eqn, config; halo=true) where {VF<:VectorField} # Extend to 3D!
    (; cells, cell_neighbours) = Hv.mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    A = _A(U_eqn)
    nzval, colval, rowptr = get_sparse_fields(A)
    (; bx, by, bz) = U_eqn.equation

    ndrange = length(cells)
    kernel! = _sized(_H!, backend, workgroup, ndrange)
    kernel!(cells, cell_neighbours,
        nzval, rowptr, colval, U_eqn.equation.diag_nz, bx, by, bz, U, Hv)
    # # KernelAbstractions.synchronize(backend)
    halo && sync!(Hv, Hv.mesh, config) # self-syncing seam (no-op serial)
end

# Pressure correction kernel
@kernel function _H!(cells::AbstractArray{Cell{TF,SV,UR}}, cell_neighbours,
    nzval, rowptr, colval, diag_nz, bx, by, bz, U, Hv) where {TF,SV,UR}
    i = @index(Global)

    @uniform begin
        Ux, Uy, Uz = U.x, U.y, U.z
        Hx, Hy, Hz = Hv.x, Hv.y, Hv.z
    end

    sumx = zero(TF)
    sumy = zero(TF)
    sumz = zero(TF)

    @inbounds begin

        start_index = rowptr[i]
        end_index = rowptr[i+1] - 1
        for nzi ∈ start_index:end_index
            nID = colval[nzi]
            val = nzval[nzi]
            sumx += val * Ux[nID]
            sumy += val * Uy[nID]
            sumz += val * Uz[nID]
        end

        DIndex = diag_nz[i]

        # remove diagonal contribution
        D = nzval[DIndex]
        sumx -= D*Ux[i]
        sumy -= D*Uy[i]
        sumz -= D*Uz[i]

        rD = one(D)/D
        Hx[i] = (bx[i] - sumx)*rD
        Hy[i] = (by[i] - sumy)*rD
        Hz[i] = (bz[i] - sumz)*rD
    end
end

## COURANT NUMBER

# global_max seam: serial = identity, Distribute = MPI.Allreduce(max). _base_mesh unwraps
# a DistributedMesh so the Mesh2/Mesh3 courant kernel still dispatches on the concrete geometry.
global_max(v, mesh) = v
_base_mesh(mesh) = mesh

max_courant_number!(cellsCourant, model, config) = begin
    (; U) = model.momentum
    (; mesh) = U
    # (; cells) = mesh
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    ndrange = length(cellsCourant)
    kernel! = _sized(_max_courant_number!, backend, workgroup, ndrange)
    kernel!(cellsCourant, U, runtime, _base_mesh(mesh))
    # # KernelAbstractions.synchronize(backend)
    return global_max(maximum(cellsCourant), mesh)
end

@kernel function _max_courant_number!(cellsCourant, U, runtime, mesh::Mesh3)
    i = @index(Global)
    @uniform cells = mesh.cells
    dt = runtime.dt[1]
    umag = norm(U[i])
    volume = cells[i].volume
    dx = volume^(one(volume)/typeof(volume)(3))
    cellsCourant[i] = umag * dt / dx
end

@kernel function _max_courant_number!(cellsCourant, U, runtime, mesh::Mesh2)
    i = @index(Global)
    @uniform cells = mesh.cells
    dt = runtime.dt[1]
    umag = norm(U[i])
    volume = cells[i].volume
    dx = sqrt(volume)
    cellsCourant[i] = umag * dt / dx
end

## ALPHA COURANT NUMBER

max_alpha_courant_number!(cellsAlphaCourant, alpha, mdotf, model, config, dt) = begin
    (; U) = model.momentum
    (; mesh) = U
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    ndrange = length(cellsAlphaCourant)
    kernel! = _sized(_max_alpha_courant_number!, backend, workgroup, ndrange)
    kernel!(cellsAlphaCourant, alpha, mdotf, runtime, dt, mesh)
    # # KernelAbstractions.synchronize(backend)
    return maximum(cellsAlphaCourant)
end


@kernel function _max_alpha_courant_number!(cellsAlphaCourant, alpha, mdotf, runtime, dt, mesh)
    i = @index(Global)

    @uniform cells = mesh.cells
    @uniform cell_faces = mesh.cell_faces

    # dt = runtime.dt
    volume = cells[i].volume
    alphaVal = alpha[i]

    nearInterfaceVal = nearInterface(alphaVal)
    sumAbsMdotf = zero(alphaVal)

    fr = cells[i].faces_range
    @inbounds for k in fr
        pointer = cell_faces[k]
        # faceID = mesh.faces[pointer]
        sumAbsMdotf += abs(mdotf[pointer])
    end

    cellsAlphaCourant[i] = dt * nearInterfaceVal * sumAbsMdotf / volume
end

@inline nearInterface(alpha) = ifelse((alpha > 0.01) & (alpha < 0.99), one(alpha), zero(alpha)) #Combines the two functions below into one
# @inline pos0(x) = ifelse(x >= zero(x), one(x), zero(x))
# @inline nearInterface(α) = pos0(α - 0.01) * pos0(0.99 - α)


update_dt!(runtime::Runtime{<:Any,<:Any,<:Any,Nothing}, ::Any) = nothing
update_dt!(runtime::Runtime{<:Any,<:Any,<:Any,Nothing}, ::Any, ::Any) = nothing

function update_dt!(runtime::Runtime{<:Any,<:Any,<:Any,<:AdaptiveTimeStepping}, courant)
    (; maxCo, maxGrow, minShrink) = runtime.adaptive

    courant_factor = maxCo / (courant + eps())
    new_dt_factor = clamp(courant_factor, minShrink, maxGrow)
    runtime.dt .= runtime.dt .* new_dt_factor
end

function update_dt!(runtime::Runtime{<:Any,<:Any,<:Any,<:AdaptiveTimeStepping}, courant, alphaCourant)
    (; maxCo, maxAlphaCo, maxGrow, minShrink) = runtime.adaptive

    courant_factor = maxCo / (courant + eps())
    alphaCourant_factor = maxAlphaCo / (alphaCourant + eps())
    
    new_dt_factor = min(courant_factor, alphaCourant_factor)
    new_dt_factor = clamp(new_dt_factor, minShrink, maxGrow)

    runtime.dt .= runtime.dt .* new_dt_factor
end

# NEW SECTION: restart hooks

# cell state and loop position before the initial calculations, face flux after them; the
# distributed module implements both for results written with output=OpenFOAM()
restart_fields!(mesh, model, ::Nothing, config) = (0, nothing)
restart_fields!(mesh, model, restart, config) =
    error("restart is supported on a distributed mesh whose results were written with output=OpenFOAM()")
restart_flux!(mesh, mdotf, ::Nothing, config) = nothing
restart_flux!(mesh, mdotf, restart, config) = restart_fields!(mesh, nothing, restart, config)

# NEW SECTION: run continuation
# A continued run! starts from the conservative face flux of the previous run when U and p are
# untouched since; otherwise (first run, initialise!, failed run) the flux stays rebuilt from U.
continue_flux!(mdotf, momentum, mesh, ::Nothing) = nothing
continue_flux!(mdotf, momentum, mesh, restart) = nothing # results read from file take precedence
function continue_flux!(mdotf, momentum::ModelPhysics.regular, mesh, ::Nothing)
    (; U, p, Umdot, pmdot) = momentum
    fresh = U.x.values == Umdot.x.values && U.y.values == Umdot.y.values &&
        U.z.values == Umdot.z.values && p.values == pmdot.values &&
        all(isfinite, momentum.mdotf.values)
    global_any(!fresh, mesh) && return nothing # all ranks rebuild if any rank cannot reuse
    copyto!(mdotf.values, momentum.mdotf.values)
    nothing
end

store_flux!(momentum, mdotf) = nothing
function store_flux!(momentum::ModelPhysics.regular, mdotf)
    (; U, p, Umdot, pmdot) = momentum
    copyto!(momentum.mdotf.values, mdotf.values)
    copyto!(Umdot.x.values, U.x.values); copyto!(Umdot.y.values, U.y.values)
    copyto!(Umdot.z.values, U.z.values); copyto!(pmdot.values, p.values)
    nothing
end

# NEW SECTION: distributed support
# Opt-in: a combination without a distributed solve seam would silently solve per-rank blocks.
distributed_ready(::Any) = false
distributed_ready(::Nothing) = true                 # a model the case does not define
distributed_ready(::Incompressible) = true
distributed_ready(::Uniform) = true
distributed_ready(::Isothermal) = true
distributed_ready(::Conduction) = true
distributed_ready(::Laminar) = true
distributed_ready(::KOmega) = true
distributed_ready(::KOmegaSST) = true
distributed_ready(::WALE) = true

# per-solver readiness: the density-based solver only stores T in its energy model (no
# equation to distribute), and SupersonicFlow is only ever solved by it
distributed_ready(::Val, m) = distributed_ready(m)
distributed_ready(::Val{:Godunov}, ::SupersonicFlow) = true
distributed_ready(::Val{:Godunov}, ::SensibleEnthalpy) = true
distributed_ready(::Val{:Godunov}, ::WALE) = false

const DISTRIBUTED_SOLVERS = (:SIMPLE, :PISO, :Laplace, :potential_flow, :Godunov)

# runtime post-processing has no distributed implementation; Distribute warns once from rank 0
_warn_skipped_postprocess(mesh, postprocess) = nothing
_has_postprocess(p) = !(p === nothing || (p isa Union{Tuple,AbstractVector} && isempty(p)))

## Time schemes

_time_schemes(s::Schemes) = (s.time,)
_time_schemes(schemes) = Tuple(s.time for s ∈ values(schemes) if s isa Schemes)
_is_crank_nicolson(t) = t === CrankNicolson || t isa CrankNicolson

# the explicit terms are taken at the time level of one scheme, shared by all fields
function check_time_schemes(schemes)
    t = _time_schemes(schemes)
    any(_is_crank_nicolson, t) && Euler ∈ t && throw(ArgumentError(
        "CrankNicolson and Euler cannot be combined: use the same time scheme for every field"))
    nothing
end

# solvers whose time integration outside the Time terms is explicit and first order (or absent)
function reject_crank_nicolson(schemes, solver, reason)
    any(_is_crank_nicolson, _time_schemes(schemes)) && throw(ArgumentError(
        "CrankNicolson is not available in $solver: $reason"))
    nothing
end

# called by every solver entry point, named for itself; a no-op on a serial mesh
function check_distributed_support(solver::Symbol, model)
    is_distributed_mesh(model.domain) || return nothing
    gaps = String[]
    solver ∈ DISTRIBUTED_SOLVERS || push!(gaps, "the $(solver) solver")
    for (kind, m) ∈ (("fluid", model.fluid), ("solid", model.solid),
                     ("turbulence", model.turbulence), ("energy", model.energy))
        distributed_ready(Val(solver), m) || push!(gaps, "the $(nameof(typeof(m))) $(kind) model")
    end
    isempty(gaps) && return nothing
    error("""
    Not supported on a distributed mesh: $(join(gaps, ", ")).

    Distributed runs currently support the SIMPLE, PISO and Laplace solvers and `potential_flow!`,
    with an Incompressible fluid or Uniform solid, Isothermal or Conduction energy, and Laminar,
    KOmega, KOmegaSST or WALE turbulence, and the density-based solver (`godunov!`, a
    SupersonicFlow fluid) with Laminar, KOmega or KOmegaSST turbulence. Everything else still needs its distributed linear-solve seam:
    without one each rank solves its own block and the result is wrong without any error.

    Run this case on a single process, or see the distributed section of the documentation.
    """)
end

# Adds pconv_f p_f to mdotf with the face pressure of the scheme that discretises
# Divergence(pconv, p) in the pressure matrix, so mdotf is the flux the pressure equation balances.
# Boundary faces take the boundary pressure pf, as the matrix does.
function add_pressure_convection_flux!(mdotf, pconv, p, pf, scheme::AbstractScheme, config)
    (; backend, workgroup) = config.hardware
    (; faces, boundary_cellsID) = p.mesh
    n_bfaces = length(boundary_cellsID)
    @views @. mdotf.values[1:n_bfaces] += pconv.values[1:n_bfaces]*pf.values[1:n_bfaces]
    kernel! = _sized(_add_pressure_convection_flux!, backend, workgroup, length(faces) - n_bfaces)
    kernel!(mdotf, pconv, p, faces, scheme, n_bfaces)
end

@kernel function _add_pressure_convection_flux!(mdotf, pconv, p, faces, scheme, n_bfaces)
    i = @index(Global)
    fID = i + n_bfaces
    @inbounds begin
        cID1, cID2 = faces.ownerCells[fID]
        F = pconv[fID]
        mdotf[fID] += F*_pconv_face_value(scheme, F, p[cID1], p[cID2], faces.weight[fID])
    end
end

@inline _pconv_face_value(::Linear, F, p1, p2, w) = w*p1 + (one(w) - w)*p2
@inline _pconv_face_value(::Union{Upwind,BoundedUpwind}, F, p1, p2, w) = F >= zero(F) ? p1 : p2
@inline _pconv_face_value(::LUST, F, p1, p2, w) =
    typeof(w)(0.75)*_pconv_face_value(Linear(), F, p1, p2, w) +
    typeof(w)(0.25)*_pconv_face_value(Upwind(), F, p1, p2, w)
