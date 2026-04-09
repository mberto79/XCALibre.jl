# ─── Multigrid cycles ─────────────────────────────────────────────────────────
#
# All cycle functions accept a concrete parametric Vector{<:MultigridLevel} so
# that every field access (level.A, level.x, …) is type-stable and kernel
# arguments are always concrete types.  No dynamic dispatch, no boxing.

# ── V-cycle ───────────────────────────────────────────────────────────────────

function vcycle!(levels::Vector{<:MultigridLevel}, lvl::Int,
                  opts::AMG, backend, workgroup)
    L  = levels[lvl]
    nc = length(levels)

    if lvl == nc
        amg_coarse_solve!(L, backend)
        return
    end

    # Pre-smoothing
    apply_level_smoother!(L, opts.pre_sweeps, opts, backend, workgroup)

    # Residual r = b - A x
    amg_residual!(L.r, L.A, L.x, L.b, backend, workgroup)

    # Restrict:  b_c = R * r
    Lc = levels[lvl + 1]
    amg_spmv!(Lc.b, L.R, L.r, backend, workgroup)
    amg_zero!(Lc.x, backend, workgroup)

    # Coarser solve
    vcycle!(levels, lvl + 1, opts, backend, workgroup)

    # Prolongate correction:  x += P * x_c
    amg_spmv_add!(L.x, L.P, Lc.x, one(eltype(L.x)), backend, workgroup)

    # Post-smoothing
    apply_level_smoother!(L, opts.post_sweeps, opts, backend, workgroup)
end

# ── W-cycle ───────────────────────────────────────────────────────────────────

function wcycle!(levels::Vector{<:MultigridLevel}, lvl::Int,
                  opts::AMG, backend, workgroup)
    L  = levels[lvl]
    nc = length(levels)

    if lvl == nc
        amg_coarse_solve!(L, backend)
        return
    end

    # Pre-smoothing
    apply_level_smoother!(L, opts.pre_sweeps, opts, backend, workgroup)

    # Residual + first coarse solve
    amg_residual!(L.r, L.A, L.x, L.b, backend, workgroup)
    Lc = levels[lvl + 1]
    amg_spmv!(Lc.b, L.R, L.r, backend, workgroup)
    amg_zero!(Lc.x, backend, workgroup)
    wcycle!(levels, lvl + 1, opts, backend, workgroup)
    amg_spmv_add!(L.x, L.P, Lc.x, one(eltype(L.x)), backend, workgroup)

    # Second residual + coarse solve
    amg_residual!(L.r, L.A, L.x, L.b, backend, workgroup)
    amg_spmv!(Lc.b, L.R, L.r, backend, workgroup)
    amg_zero!(Lc.x, backend, workgroup)
    wcycle!(levels, lvl + 1, opts, backend, workgroup)
    amg_spmv_add!(L.x, L.P, Lc.x, one(eltype(L.x)), backend, workgroup)

    # Post-smoothing
    apply_level_smoother!(L, opts.post_sweeps, opts, backend, workgroup)
end

# ── Dispatch on cycle type ─────────────────────────────────────────────────────

function run_cycle!(levels::Vector{<:MultigridLevel}, opts::AMG, ::VCycle,
                     backend, workgroup)
    vcycle!(levels, 1, opts, backend, workgroup)
end

function run_cycle!(levels::Vector{<:MultigridLevel}, opts::AMG, ::WCycle,
                     backend, workgroup)
    wcycle!(levels, 1, opts, backend, workgroup)
end
