# Phase 7 (G1): device-resident, frozen-sparsity coarse refresh for the matrix-free V-cycle.
# _build_mf_ml does a FULL HOST rebuild (re-aggregate + sparse RAP + permute + lu) every call, which
# is setup, not refresh — unusable per timestep. For a transient solve only A_0's values change; the
# aggregation, permutation and every coarse sparsity pattern are frozen. This builds a refresh plan
# ONCE and then updates ALL numeric data on the device without touching structure.
#
# Under Geometric agglomeration P is unsmoothed piecewise-constant with ONE nonzero per row, so the
# Galerkin coarse operator collapses to a scatter-add aggregation:
#   A_coarse[I,J] = (1/√w_I)(1/√w_J) · Σ_{i∈I, j∈J} A_fine[i,j]
# Every fine nonzero maps to exactly one coarse slot. We precompute (dst slot, scale) per fine
# nonzero once; refresh = zero coarse nzval + one atomic scatter-add per fine nonzero, cascaded down
# the hierarchy. This is the device-native RAP for 1nz/row P — it never materializes P/R (so it keeps
# Option B's VRAM win, unlike the cuSPARSE (R·A)·P path) and runs entirely on device.
# Scope: fused_top=0 (every level materialized). A matrix-free coarse level stores no A_l, so the
# cascade cannot read through it; the top-k VRAM lever (Option C) is refreshed separately if needed.

# NEW SECTION: refresh kernels

# Coarse RAP scatter, recomputing dst+scale in-kernel (no stored rap_dst/rap_scale). One thread per
# SOURCE row i: scale = (1/√w_I)(1/√w_J) from the level's transfer factors; dst = binary search of the
# coarse column J in the frozen target row I. Lean refresh — the per-nonzero maps cost more VRAM than
# the per-row P/R they replaced (phase5d regression), and every input here is already resident in the
# MFLevel (row_macro/coarse_pos/inv_sqrt_w) + the materialized target operator (rowptr/colval).
@kernel function _amg_rap_scatter_recompute_kernel!(dst_nz, @Const(src_rowptr), @Const(src_colval),
                                                    @Const(src_nzval), @Const(row_macro),
                                                    @Const(coarse_pos), @Const(inv_sqrt_w),
                                                    @Const(tgt_rowptr), @Const(tgt_colval))
    i = @index(Global)
    T = eltype(dst_nz)
    @inbounds begin
        gi = Int(row_macro[i]); I = Int(coarse_pos[gi]); swi = inv_sqrt_w[gi]
        tlo = Int(tgt_rowptr[I]); thi = Int(tgt_rowptr[I + 1]) - 1
        for s in src_rowptr[i]:(src_rowptr[i + 1] - 1)
            j = Int(src_colval[s]); gj = Int(row_macro[j]); J = Int(coarse_pos[gj])
            lo = tlo; hi = thi; p = tlo                       # binary search J in tgt row I
            while lo <= hi
                mid = (lo + hi) >> 1; cv = Int(tgt_colval[mid])
                if cv == J
                    p = mid; break
                elseif cv < J
                    lo = mid + 1
                else
                    hi = mid - 1
                end
            end
            Atomix.@atomic dst_nz[p] += swi * inv_sqrt_w[gj] * src_nzval[s]
        end
    end
end

# Diagonal inverse from frozen diag_index (idx==0 → unit, like reference _diag_inverse!).
@kernel function _amg_invdiag_kernel!(invdiag, @Const(nzval), @Const(diag_index))
    i = @index(Global)
    T = eltype(invdiag)
    @inbounds begin
        idx = diag_index[i]
        aii = idx == 0 ? one(T) : nzval[idx]
        invdiag[i] = abs(aii) > eps(T) ? inv(aii) : one(T)
    end
end

# Power-iteration seed v[i] = ±1 (sign by parity), matching reference _estimate_lambda_max!.
@kernel function _amg_seed_kernel!(v)
    i = @index(Global)
    T = eltype(v)
    @inbounds v[i] = isodd(i) ? one(T) : -one(T)
end

# Gershgorin row bound out[i] = (Σ_p |nz[p]|)·|invdiag[i]| (scaled Gershgorin floor for ω safety).
@kernel function _amg_abs_rowsum_kernel!(out, @Const(rowptr), @Const(colval), @Const(nzval), @Const(invdiag))
    i = @index(Global)
    T = eltype(out)
    s = zero(T)
    @inbounds begin
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            s += abs(nzval[p])
        end
        out[i] = s * abs(invdiag[i])
    end
end

# NEW SECTION: refresh plan (frozen structure, device-resident)

# Lean refresh plan: NO per-nonzero scatter maps (they cost more VRAM than the P/R they replaced —
# see phase5d regression). The cascade recomputes dst+scale in-kernel from each level's resident
# transfer factors + target sparsity (_amg_rap_scatter_recompute_kernel!). The finest stream gathers
# the incoming (host) values straight into a reusable HOST buffer (one H->D copy into A_perm_1), so no
# device value buffer is stored either. Only the coarsest operator stays device-resident.
mutable struct MFRefreshPlan{HI, HT, MA, HCS, VT}
    finest_value_map::HI      # host: A_perm_1 nz slot -> incoming Am nz slot
    finest_host::HT           # host buffer: A_perm_1 values gathered at the cycle precision (frozen len)
    coarsest_csr::MA          # device coarsest operator (natural order, frozen pattern)
    coarsest_host::HCS        # host coarsest CSR (frozen pattern) for D->H + lu refactor
    omega_nominal::VT         # 1-elt device buffer holding the nominal omega cap (scalar, kept as field)
end

# Assert the frozen coarse pattern is TOTAL for a transfer level (host): every fine nonzero (i,j) maps
# to an existing slot (I,J) in the target. The in-kernel recompute relies on this (binary search must
# hit), so we check it once at build. sparse() can drop exact-zero cancellations (FVM-safe on summed
# conductances, but guarded). No arrays stored — this replaces the per-nonzero map of the old plan.
function _assert_rap_total(fine, target, row_macro, coarse_pos)
    frp = _rowptr(fine); fcv = _colval(fine); n = _m(fine)
    trp = _rowptr(target); tcv = _colval(target)
    @inbounds for i in 1:n
        gi = Int(row_macro[i]); I = Int(coarse_pos[gi])
        lo0 = Int(trp[I]); hi0 = Int(trp[I + 1]) - 1
        for s in frp[i]:(frp[i + 1] - 1)
            j = Int(fcv[s]); gj = Int(row_macro[j]); J = Int(coarse_pos[gj])
            p = searchsortedfirst(view(tcv, lo0:hi0), J) + lo0 - 1
            (p <= hi0 && Int(tcv[p]) == J) || error("frozen coarse pattern missing slot ($I,$J)")
        end
    end
end

# Build the refresh plan from a _build_mf_ml handle (requires fused_top=0). Asserts every transfer
# level's frozen coarse pattern is total (the in-kernel recompute relies on it), then stores only the
# finest stream + coarsest operator — the per-level scatter maps are recomputed on device each refresh.
function build_mf_refresh_plan(handle)
    st = handle.st; T = handle.T; bk = st.backend; M = handle.M
    st.fused_top == 0 || error("G1 refresh requires fused_top=0 (all coarse levels materialized)")
    dev(v) = bk isa CPU ? copy(v) : Adapt.adapt(bk, v)
    A_perms = handle.A_perms; operators = handle.operators; cpis = handle.cpis; aos = handle.aos

    for l in 1:M
        n = _m(A_perms[l]); nc = maximum(handle.aggs[l])
        _, row_macro = _mf_transfer_factors(aos[l], nc, n, T)
        coarse_pos = l < M ? cpis[l + 1] : Int32.(1:nc)
        target = l < M ? A_perms[l + 1] : operators[end]
        _assert_rap_total(A_perms[l], target, row_macro, coarse_pos)
    end

    coarsest = operators[end]
    coarsest_csr = AMGMatrixCSR(dev(_rowptr(coarsest)), dev(_colval(coarsest)),
                                dev(copy(_nzval(coarsest))), _m(coarsest), _n(coarsest))
    coarsest_host = AMGMatrixCSR(copy(_rowptr(coarsest)), copy(_colval(coarsest)),
                                 copy(_nzval(coarsest)), _m(coarsest), _n(coarsest))
    finest_host = zeros(T, length(handle.pvms[1]))
    return MFRefreshPlan(copy(handle.pvms[1]), finest_host,
                         coarsest_csr, coarsest_host, dev(T[st.omega_nominal]))
end

# Device power iteration for λ_max(D⁻¹A) on a frozen operator, reusing level scratch (x/tmp/r/rhs).
# Same math as reference _estimate_lambda_max but seed-agnostic (returns the dominant eigenvalue);
# the Gershgorin floor is permutation-invariant so we do NOT chase the build's natural-order value.
function _lambda_max_device!(lv::MFLevel, bk, wg; iters::Int=5)
    A = lv.A; rp, cv, nz = _rowptr(A), _colval(A), _nzval(A); T = eltype(lv.invdiag)
    v = lv.x; wraw = lv.tmp; w = lv.r; gb = lv.rhs; n = lv.n
    _launch_amg_kernel!(bk, wg, _amg_seed_kernel!, n, v)
    KernelAbstractions.synchronize(bk)
    v ./= sqrt(T(n))
    lambda = one(T)
    for _ in 1:iters
        _launch_amg_kernel!(bk, wg, _amg_csr_matvec_kernel!, n, wraw, rp, cv, nz, v)
        KernelAbstractions.synchronize(bk)
        w .= wraw .* lv.invdiag
        lambda = max(T(norm(w)), eps(T))
        v .= w ./ lambda
    end
    _launch_amg_kernel!(bk, wg, _amg_abs_rowsum_kernel!, n, gb, rp, cv, nz, lv.invdiag)
    KernelAbstractions.synchronize(bk)
    return max(lambda, T(maximum(gb)), one(T))
end

# NEW SECTION: the refresh entry point

# Refresh ALL numeric data of a built MFMLState from a new operator A2 (SAME sparsity as the original
# fine A) with frozen structure: finest values (host gather + one H->D) -> cascaded coarse RAP scatter
# (device, recomputed) -> per-level invdiag + omega (device) -> coarsest LU refactor. No re-aggregation,
# no host RAP, no realloc. Lean: no per-nonzero scatter maps, no device value buffer (see MFRefreshPlan).
function refresh_mf_ml!(st::MFMLState, plan::MFRefreshPlan, A2; omega_nominal=nothing)
    bk = st.backend; wg = 256; M = length(st.levels); T = eltype(plan.finest_host)
    omega_cap = omega_nominal === nothing ? st.omega_nominal : T(omega_nominal)
    # finest: gather the incoming (host) values into A_perm_1 order on host, one H->D copy (no device
    # value buffer, no device gather kernel — the H->D traffic is identical to staging then gathering)
    nz2 = _nzval(_amg_matrix(A2)); fvm = plan.finest_value_map; hb = plan.finest_host
    @inbounds for k in eachindex(hb)
        hb[k] = T(nz2[fvm[k]])
    end
    copyto!(_nzval(st.levels[1].A), hb)
    # cascade coarse operators: zero target, scatter-add from the (already refreshed) fine level,
    # recomputing dst+scale in-kernel from resident transfer factors (one thread per source row)
    for l in 1:M
        lv = st.levels[l]; src = lv.A
        tgt = l < M ? st.levels[l + 1].A : plan.coarsest_csr
        dst_nz = _nzval(tgt)
        _fill_device!(bk, dst_nz, zero(T))
        _launch_amg_kernel!(bk, wg, _amg_rap_scatter_recompute_kernel!, lv.n, dst_nz,
                            _rowptr(src), _colval(src), _nzval(src),
                            lv.row_macro, lv.coarse_pos, lv.inv_sqrt_w,
                            _rowptr(tgt), _colval(tgt))
    end
    # per-level smoother data: invdiag from frozen diag_index, omega from device power iteration
    for l in 1:M
        lv = st.levels[l]
        _launch_amg_kernel!(bk, wg, _amg_invdiag_kernel!, lv.n, lv.invdiag, _nzval(lv.A), lv.diag_index)
        KernelAbstractions.synchronize(bk)
        lambda = _lambda_max_device!(lv, bk, wg)
        lv.omega = min(omega_cap, T(2) - eps(T)) / lambda
    end
    # coarsest: D->H the refreshed coarsest operator into the frozen host pattern, refresh whichever
    # coarse mechanism is active (device dense inverse for GEMV, else refactor host LU)
    KernelAbstractions.synchronize(bk)
    copyto!(_nzval(plan.coarsest_host), Array(_nzval(plan.coarsest_csr)))
    coarse_csc = _csr_to_csc(plan.coarsest_host)
    st.coarse_fac = lu(coarse_csc)
    st.coarse_inv === nothing ||
        (st.coarse_inv = _build_coarse_dense_inv(coarse_csc, bk, T, st.coarse_max_rows))
    return st
end

# NEW SECTION: G1 validation driver

# Frozen-aggregation Galerkin RAP oracle: A1's FROZEN aggregation/permutation/transfers applied to
# A2's values via independent host SpGEMM (sparse(Pᵀ)·A·P). This is the correct oracle for a frozen
# refresh: _geometric_aggregates is VALUE-dependent, so a full rebuild on A2 re-aggregates to a
# different (incompatible) pattern — only on structured grids do A1/A2 happen to coincide. The refresh
# freezes A1's aggregation (standard transient AMG, re-setup periodically), so the oracle must too.
function _frozen_rap_oracle(handle, A2)
    T = handle.T; M = handle.M
    op = _permuted_operator(_amg_matrix(A2), handle.cps[1], handle.cpis[1])[1]  # value-only perm
    ops = Any[op]; omegas = T[]
    for l in 1:M
        n = _m(ops[l]); nc = maximum(handle.aggs[l])
        iw, rm = _mf_transfer_factors(handle.aos[l], nc, n, T)
        cpos = l < M ? handle.cpis[l + 1] : Int32.(1:nc)
        cols = [Int(cpos[rm[k]]) for k in 1:n]; vals = [iw[rm[k]] for k in 1:n]
        Pperm = sparse(collect(1:n), cols, vals, n, nc)
        push!(ops, _amg_matrix(sparse(Pperm') * _csr_to_csc(ops[l]) * Pperm))
        _, invd = _diag_inverse(ops[l]); push!(omegas, _estimate_lambda_max(ops[l], invd))
    end
    return ops, omegas
end

# Validate the device refresh (A1 built -> A2 streamed, frozen structure) against the frozen-aggregation
# oracle on A2's values. A2 must share A1's sparsity (different values) — this tests both the scatter
# arithmetic AND that the frozen pattern is adequate for changed values. Returns the worst relative
# nzval error across every level + the coarsest, plus a separate omega relerr (λ_max drift).
function mf_ml_refresh_error(A1, A2, merge_levels::Integer, backend; pre::Int=2, post::Int=2,
                             omega_nominal=4/3, max_coarse::Integer=64)
    h1 = _build_mf_ml(A1, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                      max_coarse=max_coarse, fused_top=0)
    plan = build_mf_refresh_plan(h1)
    refresh_mf_ml!(h1.st, plan, A2; omega_nominal=omega_nominal)
    ops, lam = _frozen_rap_oracle(h1, A2); T = h1.T; M = length(h1.st.levels)
    relerr = 0.0; omega_relerr = 0.0
    for l in 1:M
        got = Array(_nzval(h1.st.levels[l].A)); ref = _nzval(ops[l])
        length(got) == length(ref) || error("refresh/oracle pattern mismatch at level $l")
        relerr = max(relerr, maximum(abs.(got .- ref)) / max(maximum(abs.(ref)), eps(T)))
        om_ref = min(T(omega_nominal), T(2) - eps(T)) / T(lam[l])
        omega_relerr = max(omega_relerr, abs(h1.st.levels[l].omega - om_ref) / max(abs(om_ref), eps(T)))
    end
    cg = Array(_nzval(plan.coarsest_csr)); cr = _nzval(ops[end])
    relerr = max(relerr, maximum(abs.(cg .- cr)) / max(maximum(abs.(cr)), eps(T)))
    return (relerr=relerr, omega_relerr=omega_relerr, levels=M + 1, n=h1.st.n)
end
