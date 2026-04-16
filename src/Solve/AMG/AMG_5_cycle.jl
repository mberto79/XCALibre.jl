# ─── Multigrid cycles ─────────────────────────────────────────────────────────
# Two-tier design: fine level (Float64) separate from coarse (Float32) with cast kernels at boundary.
# Both cycles are type-stable: concrete field accesses and kernel arguments (no dispatch).

# ── Coarse-level V-cycle (all Float32) ────────────────────────────────────────

function vcycle_coarse!(coarse::Vector{<:MultigridLevel}, lvl::Int,
                         opts::AMG, backend, workgroup)
    L  = coarse[lvl]
    nc = length(coarse)

    if lvl == nc
        amg_coarse_solve!(L, opts.coarse_sweeps, backend, workgroup)
        return
    end

    apply_level_smoother!(L, opts.pre_sweeps, opts, backend, workgroup)

    amg_residual!(L.r, L.A, L.x, L.b, backend, workgroup)

    Lc = coarse[lvl + 1]
    amg_spmv!(Lc.b, L.R, L.r, backend, workgroup)
    amg_zero!(Lc.x, backend, workgroup)

    vcycle_coarse!(coarse, lvl + 1, opts, backend, workgroup)

    amg_spmv_add!(L.x, L.P, Lc.x, one(eltype(L.x)), backend, workgroup)

    apply_level_smoother!(L, opts.post_sweeps, opts, backend, workgroup)
end

# ── Fine-level V-cycle entry (Float64 fine, Float32 coarse) ───────────────────
# Two cast kernels at fine↔coarse boundary; fine P/R stored as Float32 (bandwidth-efficient).
# TcVec parameter keeps r_Tc/tmp_Tc type-stable at call site.

function vcycle_fine!(
    fine::MultigridLevel{<:Any, <:Any, <:Any, <:Any, <:LevelExtras{<:Any, TcVec}},
    coarse::Vector{<:MultigridLevel},
    opts::AMG, backend, workgroup
) where {TcVec}
    if isempty(coarse)
        # Single-level hierarchy: fine IS the coarsest level; direct solve only.
        amg_coarse_solve!(fine, opts.coarse_sweeps, backend, workgroup)
        return
    end

    # Extract boundary buffers (safe: coarse non-empty ⟹ TcVec ≠ Nothing).
    r_Tc   = fine.extras.r_Tc::TcVec
    tmp_Tc = fine.extras.tmp_Tc::TcVec

    _apply_fine_smoother!(fine, opts.pre_sweeps, opts, backend, workgroup)
    amg_residual!(fine.r, fine.A, fine.x, fine.b, backend, workgroup)

    Lc = coarse[1]
    # Fine→coarse: cast residual Float64→Float32, then restrict.
    amg_cast_copy!(r_Tc, fine.r, backend, workgroup)
    amg_spmv!(Lc.b, fine.R, r_Tc, backend, workgroup)
    amg_zero!(Lc.x, backend, workgroup)
    vcycle_coarse!(coarse, 1, opts, backend, workgroup)

    # Coarse→fine: prolongate in Float32, then cast to Float64.
    amg_spmv!(tmp_Tc, fine.P, Lc.x, backend, workgroup)
    amg_cast_copy!(fine.tmp, tmp_Tc, backend, workgroup)
    amg_axpy!(fine.x, fine.tmp, one(eltype(fine.x)), backend, workgroup)
    _apply_fine_smoother!(fine, opts.post_sweeps, opts, backend, workgroup)
end

# ── Coarse-level W-cycle (all Float32) ────────────────────────────────────────

function wcycle_coarse!(coarse::Vector{<:MultigridLevel}, lvl::Int,
                         opts::AMG, backend, workgroup)
    L  = coarse[lvl]
    nc = length(coarse)

    if lvl == nc
        amg_coarse_solve!(L, opts.coarse_sweeps, backend, workgroup)
        return
    end

    apply_level_smoother!(L, opts.pre_sweeps, opts, backend, workgroup)

    amg_residual!(L.r, L.A, L.x, L.b, backend, workgroup)
    Lc = coarse[lvl + 1]
    amg_spmv!(Lc.b, L.R, L.r, backend, workgroup)
    amg_zero!(Lc.x, backend, workgroup)
    wcycle_coarse!(coarse, lvl + 1, opts, backend, workgroup)
    amg_spmv_add!(L.x, L.P, Lc.x, one(eltype(L.x)), backend, workgroup)

    amg_residual!(L.r, L.A, L.x, L.b, backend, workgroup)
    amg_spmv!(Lc.b, L.R, L.r, backend, workgroup)
    amg_zero!(Lc.x, backend, workgroup)
    wcycle_coarse!(coarse, lvl + 1, opts, backend, workgroup)
    amg_spmv_add!(L.x, L.P, Lc.x, one(eltype(L.x)), backend, workgroup)

    apply_level_smoother!(L, opts.post_sweeps, opts, backend, workgroup)
end

# ── Fine-level W-cycle entry ───────────────────────────────────────────────────

function wcycle_fine!(
    fine::MultigridLevel{<:Any, <:Any, <:Any, <:Any, <:LevelExtras{<:Any, TcVec}},
    coarse::Vector{<:MultigridLevel},
    opts::AMG, backend, workgroup
) where {TcVec}
    if isempty(coarse)
        amg_coarse_solve!(fine, opts.coarse_sweeps, backend, workgroup)
        return
    end

    r_Tc   = fine.extras.r_Tc::TcVec
    tmp_Tc = fine.extras.tmp_Tc::TcVec

    _apply_fine_smoother!(fine, opts.pre_sweeps, opts, backend, workgroup)

    amg_residual!(fine.r, fine.A, fine.x, fine.b, backend, workgroup)

    Lc = coarse[1]

    # First descent: restrict, coarse solve, prolongate
    amg_cast_copy!(r_Tc, fine.r, backend, workgroup)
    amg_spmv!(Lc.b, fine.R, r_Tc, backend, workgroup)
    amg_zero!(Lc.x, backend, workgroup)
    wcycle_coarse!(coarse, 1, opts, backend, workgroup)
    amg_spmv!(tmp_Tc, fine.P, Lc.x, backend, workgroup)
    amg_cast_copy!(fine.tmp, tmp_Tc, backend, workgroup)
    amg_axpy!(fine.x, fine.tmp, one(eltype(fine.x)), backend, workgroup)

    # Second descent (W-cycle extra cycle).
    amg_residual!(fine.r, fine.A, fine.x, fine.b, backend, workgroup)
    amg_cast_copy!(r_Tc, fine.r, backend, workgroup)
    amg_spmv!(Lc.b, fine.R, r_Tc, backend, workgroup)
    amg_zero!(Lc.x, backend, workgroup)
    wcycle_coarse!(coarse, 1, opts, backend, workgroup)
    amg_spmv!(tmp_Tc, fine.P, Lc.x, backend, workgroup)
    amg_cast_copy!(fine.tmp, tmp_Tc, backend, workgroup)
    amg_axpy!(fine.x, fine.tmp, one(eltype(fine.x)), backend, workgroup)
    _apply_fine_smoother!(fine, opts.post_sweeps, opts, backend, workgroup)
end

# ── Dispatch on cycle type (accesses workspace for two-tier types) ──────────────

function run_cycle!(ws::AMGWorkspace{LFType}, opts::AMG, ::VCycle,
                     backend, workgroup) where {LFType}
    vcycle_fine!(ws.fine_level::LFType, ws.coarse_levels, opts, backend, workgroup)
end

function run_cycle!(ws::AMGWorkspace{LFType}, opts::AMG, ::WCycle,
                     backend, workgroup) where {LFType}
    wcycle_fine!(ws.fine_level::LFType, ws.coarse_levels, opts, backend, workgroup)
end
