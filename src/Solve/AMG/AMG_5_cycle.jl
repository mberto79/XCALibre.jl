# ─── Multigrid cycles ─────────────────────────────────────────────────────────

# ── Single V-cycle (recursive) ────────────────────────────────────────────────
# amg_coarse_solve! is defined in AMG_6_api.jl (needs lu_factor from setup)

function vcycle!(levels::Vector, lvl::Int, opts::AMG, backend, workgroup)
    L  = levels[lvl]
    nc = length(levels)

    if lvl == nc
        amg_coarse_solve!(L, backend)
        return
    end

    # Pre-smoothing
    apply_level_smoother!(L, opts.pre_sweeps, opts, backend, workgroup)

    # Compute residual r = b - A x
    amg_residual!(L.r, L.A, L.x, L.b, backend, workgroup)

    # Restrict residual to coarse RHS:  b_c = R * r
    Lc = levels[lvl + 1]
    amg_spmv!(Lc.b, L.R, L.r, backend, workgroup)
    amg_zero!(Lc.x, backend, workgroup)

    # Solve on coarser level
    vcycle!(levels, lvl + 1, opts, backend, workgroup)

    # Prolongate correction:  x += P * x_c
    amg_spmv_add!(L.x, L.P, Lc.x, one(eltype(L.x)), backend, workgroup)

    # Post-smoothing
    apply_level_smoother!(L, opts.post_sweeps, opts, backend, workgroup)
end

# ── W-cycle (recursive, two coarse solves) ────────────────────────────────────

function wcycle!(levels::Vector, lvl::Int, opts::AMG, backend, workgroup)
    L  = levels[lvl]
    nc = length(levels)

    if lvl == nc
        amg_coarse_solve!(L, backend)
        return
    end

    # Pre-smoothing
    apply_level_smoother!(L, opts.pre_sweeps, opts, backend, workgroup)

    # Residual and restrict
    amg_residual!(L.r, L.A, L.x, L.b, backend, workgroup)
    Lc = levels[lvl + 1]
    amg_spmv!(Lc.b, L.R, L.r, backend, workgroup)
    amg_zero!(Lc.x, backend, workgroup)

    # First coarse solve
    wcycle!(levels, lvl + 1, opts, backend, workgroup)

    # Prolongate and update
    amg_spmv_add!(L.x, L.P, Lc.x, one(eltype(L.x)), backend, workgroup)

    # Second residual and restrict
    amg_residual!(L.r, L.A, L.x, L.b, backend, workgroup)
    amg_spmv!(Lc.b, L.R, L.r, backend, workgroup)
    amg_zero!(Lc.x, backend, workgroup)

    # Second coarse solve
    wcycle!(levels, lvl + 1, opts, backend, workgroup)

    # Prolongate correction
    amg_spmv_add!(L.x, L.P, Lc.x, one(eltype(L.x)), backend, workgroup)

    # Post-smoothing
    apply_level_smoother!(L, opts.post_sweeps, opts, backend, workgroup)
end

# ── Dispatch on cycle type ─────────────────────────────────────────────────────

function run_cycle!(levels::Vector, opts::AMG, ::VCycle, backend, workgroup)
    vcycle!(levels, 1, opts, backend, workgroup)
end

function run_cycle!(levels::Vector, opts::AMG, ::WCycle, backend, workgroup)
    wcycle!(levels, 1, opts, backend, workgroup)
end
