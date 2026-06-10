# Phase 5d: wire the greenfield matrix-free pipeline into the live AMG solve path. The device
# matrix-free V-cycle (device/4) + frozen-sparsity device refresh (device/5) are reused unchanged;
# this file is the seam: storage on the hierarchy, build/refresh in update!, and a permute wrapper
# that lets the existing amg_solve!/amg_cg_solve! outer loop use mf_ml_cycle as its preconditioner.
#
# Scope (Option B): fused_top=0 — P/R are matrix-free at every level, the coarse operators A_l stay
# materialized so the G1 frozen-sparsity refresh applies per timestep. This banks the bounded VRAM
# win (P/R erased) and is refreshable; Option C (fused_top>0, also erasing coarse A_l) trades cycle
# time for more VRAM and has no device refresh yet, so it is NOT used on the transient path.

# NEW SECTION: permute gather/scatter (natural <-> aggregate-contiguous order), precision-crossing

# r_perm[k] = r[perm[k]]: gather the natural-order residual into permuted (aggregate-contiguous) order.
@kernel function _amg_gather_kernel!(out, @Const(src), @Const(perm))
    k = @index(Global)
    @inbounds out[k] = src[perm[k]]
end

# z[perm[k]] = x_perm[k]: scatter the permuted correction back to natural order.
@kernel function _amg_scatter_kernel!(out, @Const(src), @Const(perm))
    k = @index(Global)
    @inbounds out[perm[k]] = src[k]
end

# NEW SECTION: greenfield state holder (lives in hierarchy.greenfield[])

mutable struct MFGreenfield{ST, PL, VI, VT}
    st::ST        # MFMLState — device matrix-free hierarchy (device/4)
    plan::PL      # MFRefreshPlan — frozen-sparsity device refresh (device/5)
    perm::VI      # device cell_perm (permuted position -> original cell), Int32
    r_perm::VT    # reusable device scratch: residual in permuted order (cycle storage type)
    nrows::Int
    nnz::Int
end

_greenfield_active(hierarchy::AMGHierarchy) = hierarchy.greenfield[] !== nothing

# Coarsest device dense-inverse threshold from the solver's coarse_solve (mirrors reference OnDevice).
_greenfield_coarse_max_rows(cs::OnDevice) = cs.max_rows
_greenfield_coarse_max_rows(::Any) = 512

_greenfield_omega(s::AMGJacobi) = s.omega
_greenfield_omega(::Any) = 4 / 3

# Host matrix re-typed at the cycle storage precision TS (coarse_storage=Float32 path). Keeps the
# rowptr/colval ordering so the refresh value map (built from this matrix) stays valid for the
# same-pattern Float64 system matrix streamed in each timestep.
function _amg_matrix_storage(A, ::Type{TS}) where {TS}
    Am = _amg_matrix(A)
    eltype(_nzval(Am)) === TS && return Am
    return AMGMatrixCSR(_rowptr(Am), _colval(Am), TS.(_nzval(Am)), _m(Am), _n(Am))
end

# NEW SECTION: G5 guard (activation-time)

# The greenfield matrix-free path is opt-in and has two edges the user should know, surfaced once at
# activation. (1) VRAM/RAM: with the Item 4 lean refresh (device/5) Option B now SAVES VRAM (~-163MB on
# F1 1.68M vs reference) but moves ~97MB to host RAM and pays a slightly slower per-timestep refresh.
# (2) fuse_levels>1 (matrix-free coarse levels, Option C) has no device refresh yet, so the transient
# path runs Option B (fused_top=0) regardless; warn and clamp. No fused single-launch kernel exists (G3
# rejected in the preconditioner regime) so there is no register/occupancy pressure to estimate — the
# SOTA "stop at 2-3 fused levels" rule is moot here.
# Returns the effective fused_top for the transient path (clamped to 0 = Option B).
function _greenfield_guard(solver::AMG)
    if solver.fuse_levels > 1
        @warn "AMG greenfield: fuse_levels>1 (matrix-free coarse levels / Option C) not yet supported on the transient path; running Option B (fused_top=0)." fuse_levels=solver.fuse_levels maxlog=1
    end
    @warn "AMG greenfield matrix-free path is EXPERIMENTAL (opt-in, not default). With the Item 4 lean refresh it now SAVES VRAM (~-163MB on F1 1.68M vs reference) at the cost of ~97MB extra host RAM and a slightly slower per-timestep refresh." maxlog=1
    return 0
end

# NEW SECTION: build / refresh

# Build the greenfield state ONCE (or on a sparsity change) from the static mesh operator.
function _greenfield_build!(workspace::AMGWorkspace, A, solver::AMG, hardware)
    backend = hardware.backend
    setup_matrix = _amg_setup_matrix(A, _amg_setup_backend(backend))  # host CSR at finest precision T
    T = eltype(_nzval(_amg_matrix(setup_matrix)))
    TS = _effective_storage(T, _amg_storage(solver.coarse_storage))  # coarse-level (>=2) storage type
    merge_levels = solver.coarsening.merge_levels
    fused_top = _greenfield_guard(solver)  # transient path supports Option B only -> clamps to 0
    handle = _build_mf_ml(setup_matrix, merge_levels, backend; pre=solver.pre_sweeps, post=solver.post_sweeps,
                          omega_nominal=_greenfield_omega(solver.smoother),
                          max_coarse=solver.max_coarse_rows, fused_top=fused_top,
                          coarse_max_rows=_greenfield_coarse_max_rows(solver.coarse_solve),
                          scale_correction=(solver.scale_correction && solver.mode isa AMGSolver),
                          coarse_storage=TS)
    plan = build_mf_refresh_plan(handle)
    st = handle.st
    dev(v) = backend isa CPU ? copy(v) : Adapt.adapt(backend, v)
    gf = MFGreenfield(st, plan, dev(st.cell_perm), dev(zeros(T, st.n)),
                      _m(_amg_matrix(setup_matrix)), length(_nzval(_amg_matrix(setup_matrix))))
    workspace.hierarchy.greenfield[] = gf
    return workspace
end

# Per-timestep numeric refresh: frozen aggregation/permutation/sparsity, restream A0 values + cascade
# the coarse Galerkin operators + smoother data, all on device (device/5).
function _greenfield_refresh!(workspace::AMGWorkspace, A, solver::AMG, hardware)
    gf = workspace.hierarchy.greenfield[]
    # Pass the live system matrix straight in: refresh gathers finest values on device via _nzval(A) in the
    # frozen fvm order, coarse cascade uses frozen device operators. Routing through _amg_setup_matrix(A,CPU)
    # forces a full D->H copy of every nonzero (~310ms/outer-iter on F1) since the GPU matrix is a bare
    # CuSparseMatrixCSR — that defeats the P7 device gather and is the matrix-free CFD per-iter deficit.
    refresh_mf_ml!(gf.st, gf.plan, A; omega_nominal=_greenfield_omega(solver.smoother))
    return workspace
end

function _greenfield_update!(workspace::AMGWorkspace, A, solver::AMG, hardware)
    gf = workspace.hierarchy.greenfield[]
    if gf === nothing || gf.nrows != _m(A) || gf.nnz != length(_nzval(A))
        return _greenfield_build!(workspace, A, solver, hardware)
    end
    _greenfield_refresh!(workspace, A, solver, hardware)
    workspace.refresh_count += 1
    return workspace
end

# NEW SECTION: preconditioner application (the seam into amg_solve!/amg_cg_solve!)

# z = M⁻¹ r via one matrix-free V-cycle: gather r into permuted order, cycle, scatter the correction
# back. Crosses precision (T residual <-> TS cycle) inside the gather/scatter assignment.
function _greenfield_apply_preconditioner!(z, gf::MFGreenfield, hierarchy::AMGHierarchy, r)
    backend = hierarchy.backend; wg = hierarchy.workgroup; n = gf.nrows
    _launch_amg_kernel!(backend, wg, _amg_gather_kernel!, n, gf.r_perm, r, gf.perm)
    x_perm = mf_ml_cycle(gf.st, gf.r_perm)  # synchronizes the backend before returning
    _launch_amg_kernel!(backend, wg, _amg_scatter_kernel!, n, z, x_perm, gf.perm)
    return z
end

# NEW SECTION: standalone validation driver (gate-independent; call directly in the REPL)

# End-to-end seam check: run the chosen mode through update!+_amg_solve_mode! on the greenfield path
# and on the reference path, on the SAME device system, returning both convergence summaries. `A` and
# `b` must already be on `backend` (caller converts; the module keeps no CUDA dep). Requires the gate
# forced on (e.g. _greenfield_implemented()=true) so the greenfield solver routes matrix-free.
function greenfield_solve_spike(A, b, backend; mode=Cg(), merge_levels=1, fuse_levels=1,
                                coarse_storage=Float64, itmax=200, rtol=1e-8, atol=1e-8,
                                workgroup=64, scale_correction=true, n_update=1)
    config = (hardware=(backend=backend, workgroup=workgroup),)

    # n_update>1 exercises the transient path: update! #1 builds, #2.. refresh (frozen sparsity, restream
    # A0 + cascade coarse operators). Isolates whether the per-timestep refresh degrades cycle quality
    # (iters) vs the static isolated build, for both gf and ref on the SAME device system.
    run_one = (solver) -> begin
        ws = _workspace(solver, b)
        for _ in 1:n_update; update!(ws, A, solver, config); end
        x = similar(b); fill!(x, 0)
        _amg_solve_mode!(ws, ws.hierarchy, solver, solver.mode, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
        (iters=ws.iterations, converged=ws.converged, relres=ws.last_relative_residual,
         factor=ws.hierarchy.last_cycle_factor, greenfield=_greenfield_active(ws.hierarchy))
    end

    gf_solver = AMG(mode=mode, coarsening=Geometric(merge_levels=merge_levels), smoother=AMGJacobi(),
                    fuse_levels=fuse_levels, coarse_storage=coarse_storage, max_coarse_rows=4096,
                    scale_correction=scale_correction)
    ref_solver = AMG(mode=mode, coarsening=Geometric(merge_levels=merge_levels), smoother=AMGJacobi(),
                     fuse_levels=0, max_coarse_rows=4096)
    return (greenfield=run_one(gf_solver), reference=run_one(ref_solver))
end
