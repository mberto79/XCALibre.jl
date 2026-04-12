# ─── Multigrid cycles ─────────────────────────────────────────────────────────
#
# Two-tier design: the fine level (Float64) is handled separately from the
# coarse levels (Float32). The boundary between level 1 and level 2 requires
# two cast kernels (Float64↔Float32); all other transitions are within Float32.
#
# Fine level: `vcycle_fine!` / `wcycle_fine!` — accept `fine::LFType` and
#             `coarse::Vector{LCType}`; specialised at the concrete types.
# Coarse levels: `vcycle_coarse!` / `wcycle_coarse!` — recurse on `Vector{LCType}`.
#
# Both are type-stable: every field access (level.A, level.x, …) is concrete,
# every kernel argument is concrete — no boxing, no dynamic dispatch.

# ── Coarse-level V-cycle (all Float32) ────────────────────────────────────────

function vcycle_coarse!(coarse::Vector{<:MultigridLevel}, lvl::Int,
                         opts::AMG, backend, workgroup)
    L  = coarse[lvl]
    nc = length(coarse)

    if lvl == nc
        amg_coarse_solve!(L, backend)
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
# The two amg_cast_copy! calls are the only mixed-precision operations:
#   fine.r (Float64) → fine.extras.r_Tc (Float32)  before restriction
#   fine.extras.tmp_Tc (Float32) → fine.tmp (Float64) after prolongation
#
# fine.R and fine.P are stored as Float32 sparse matrices so that the boundary
# SpMVs (R*r_Tc and P*x_c) run entirely in Float32, halving bandwidth at the
# fine→coarse and coarse→fine transfer steps.
#
# The `TcVec` type parameter from `LevelExtras{Tv, TcVec, CpuSpT}` is used to
# annotate r_Tc / tmp_Tc at the call site — keeps the cycle body fully type-stable.

function vcycle_fine!(
    fine::MultigridLevel{<:Any, <:Any, <:Any, <:Any, <:LevelExtras{<:Any, TcVec}},
    coarse::Vector{<:MultigridLevel},
    opts::AMG, backend, workgroup
) where {TcVec}
    if isempty(coarse)
        # Single-level hierarchy: fine IS the coarsest level; direct solve only.
        amg_coarse_solve!(fine, backend)
        return
    end

    # Extract boundary buffers (safe: coarse is non-empty, so TcVec ≠ Nothing).
    r_Tc   = fine.extras.r_Tc::TcVec
    tmp_Tc = fine.extras.tmp_Tc::TcVec

    apply_level_smoother!(fine, opts.pre_sweeps, opts, backend, workgroup)

    amg_residual!(fine.r, fine.A, fine.x, fine.b, backend, workgroup)

    Lc = coarse[1]
    # Fine→coarse boundary: cast residual Float64→Float32, then restrict.
    amg_cast_copy!(r_Tc, fine.r, backend, workgroup)
    amg_spmv!(Lc.b, fine.R, r_Tc, backend, workgroup)
    amg_zero!(Lc.x, backend, workgroup)

    vcycle_coarse!(coarse, 1, opts, backend, workgroup)

    # Coarse→fine boundary: prolongate in Float32, then cast back to Float64.
    amg_spmv!(tmp_Tc, fine.P, Lc.x, backend, workgroup)
    amg_cast_copy!(fine.tmp, tmp_Tc, backend, workgroup)
    amg_axpy!(fine.x, fine.tmp, one(eltype(fine.x)), backend, workgroup)

    apply_level_smoother!(fine, opts.post_sweeps, opts, backend, workgroup)
end

# ── Coarse-level W-cycle (all Float32) ────────────────────────────────────────

function wcycle_coarse!(coarse::Vector{<:MultigridLevel}, lvl::Int,
                         opts::AMG, backend, workgroup)
    L  = coarse[lvl]
    nc = length(coarse)

    if lvl == nc
        amg_coarse_solve!(L, backend)
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
        amg_coarse_solve!(fine, backend)
        return
    end

    r_Tc   = fine.extras.r_Tc::TcVec
    tmp_Tc = fine.extras.tmp_Tc::TcVec

    apply_level_smoother!(fine, opts.pre_sweeps, opts, backend, workgroup)

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

    # Second descent
    amg_residual!(fine.r, fine.A, fine.x, fine.b, backend, workgroup)
    amg_cast_copy!(r_Tc, fine.r, backend, workgroup)
    amg_spmv!(Lc.b, fine.R, r_Tc, backend, workgroup)
    amg_zero!(Lc.x, backend, workgroup)
    wcycle_coarse!(coarse, 1, opts, backend, workgroup)
    amg_spmv!(tmp_Tc, fine.P, Lc.x, backend, workgroup)
    amg_cast_copy!(fine.tmp, tmp_Tc, backend, workgroup)
    amg_axpy!(fine.x, fine.tmp, one(eltype(fine.x)), backend, workgroup)

    apply_level_smoother!(fine, opts.post_sweeps, opts, backend, workgroup)
end

# ── Dispatch on cycle type (takes workspace for two-tier access) ───────────────

function run_cycle!(ws::AMGWorkspace{LFType}, opts::AMG, ::VCycle,
                     backend, workgroup) where {LFType}
    vcycle_fine!(ws.fine_level::LFType, ws.coarse_levels, opts, backend, workgroup)
end

function run_cycle!(ws::AMGWorkspace{LFType}, opts::AMG, ::WCycle,
                     backend, workgroup) where {LFType}
    wcycle_fine!(ws.fine_level::LFType, ws.coarse_levels, opts, backend, workgroup)
end
