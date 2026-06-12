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
            Atomix.@atomic dst_nz[p] += T(swi * inv_sqrt_w[gj] * src_nzval[s])  # cast: src/dst precision may differ (split precision)
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

# NEW SECTION: composed-map kernels (fused/matrix-free levels have no stored A_l to cascade through)

# One aggregation hop in place: comp_dst (current-level row position) -> next level position, comp_s
# accumulates the 1/sqrt(w) chain. Init comp_dst[i]=i, comp_s[i]=1 then hop per level: the composed
# P_1...P_k keeps ONE nonzero per fine row with row-dependent weight comp_s[i].
@kernel function _amg_compose_hop_kernel!(comp_dst, comp_s, @Const(row_macro), @Const(coarse_pos),
                                          @Const(inv_sqrt_w))
    i = @index(Global)
    @inbounds begin
        g = Int(row_macro[Int(comp_dst[i])])
        comp_dst[i] = coarse_pos[g]
        comp_s[i] *= inv_sqrt_w[g]
    end
end

@kernel function _amg_compose_init_kernel!(comp_dst, comp_s)
    i = @index(Global)
    @inbounds begin
        comp_dst[i] = i
        comp_s[i] = one(eltype(comp_s))
    end
end

# Composed RAP scatter: like _amg_rap_scatter_recompute_kernel! but with per-fine-row dst/scale maps,
# skipping the fused (un-materialized) levels: A_tgt[I,J] = Σ s_i·s_j·A0[i,j]. One thread per fine row.
@kernel function _amg_rap_scatter_composed_kernel!(dst_nz, @Const(src_rowptr), @Const(src_colval),
                                                   @Const(src_nzval), @Const(comp_dst), @Const(comp_s),
                                                   @Const(tgt_rowptr), @Const(tgt_colval))
    i = @index(Global)
    T = eltype(dst_nz)
    @inbounds begin
        I = Int(comp_dst[i]); si = comp_s[i]
        tlo = Int(tgt_rowptr[I]); thi = Int(tgt_rowptr[I + 1]) - 1
        for s in src_rowptr[i]:(src_rowptr[i + 1] - 1)
            j = Int(src_colval[s]); J = Int(comp_dst[j])
            lo = tlo; hi = thi; p = tlo
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
            Atomix.@atomic dst_nz[p] += T(si * comp_s[j] * src_nzval[s])  # cast: split precision
        end
    end
end

# Diagonal of the un-materialized fused operator: d[I] = Σ_{dst(i)=dst(j)=I} s_i·s_j·A0[i,j].
@kernel function _amg_diag_scatter_composed_kernel!(d, @Const(src_rowptr), @Const(src_colval),
                                                    @Const(src_nzval), @Const(comp_dst), @Const(comp_s))
    i = @index(Global)
    T = eltype(d)
    @inbounds begin
        I = Int(comp_dst[i]); si = comp_s[i]
        for s in src_rowptr[i]:(src_rowptr[i + 1] - 1)
            j = Int(src_colval[s])
            Int(comp_dst[j]) == I || continue
            Atomix.@atomic d[I] += T(si * comp_s[j] * src_nzval[s])
        end
    end
end

# Gershgorin row-sum bound for the un-materialized fused operator, CROSS-aggregate terms only:
# rs[I] = Σ_{dst(j)≠I} |s_i·s_j·A0[i,j]|. Within-aggregate terms all land on the diagonal slot, whose
# EXACT assembled value the caller adds (|d_I|), so the bound |d_I| + rs[I] >= Σ_J|A_l[I,J]| is exact
# for M-matrix operators (same-signed off-diagonal contributions, e.g. FVM pressure) and safe otherwise.
@kernel function _amg_absrowsum_scatter_composed_kernel!(rs, @Const(src_rowptr), @Const(src_colval),
                                                         @Const(src_nzval), @Const(comp_dst), @Const(comp_s))
    i = @index(Global)
    T = eltype(rs)
    @inbounds begin
        I = Int(comp_dst[i]); si = comp_s[i]
        for s in src_rowptr[i]:(src_rowptr[i + 1] - 1)
            j = Int(src_colval[s])
            Int(comp_dst[j]) == I && continue
            Atomix.@atomic rs[I] += T(abs(si * comp_s[j] * src_nzval[s]))
        end
    end
end

# d holds the assembled diagonal; invert in place with the reference unit/eps guards.
@kernel function _amg_invert_diag_kernel!(d)
    i = @index(Global)
    T = eltype(d)
    @inbounds begin
        aii = d[i]
        d[i] = abs(aii) > eps(T) ? inv(aii) : one(T)
    end
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
mutable struct MFRefreshPlan{DI, MA, HCS, VT}
    finest_value_map::DI      # device: A_perm_1 nz slot -> incoming A2 nz slot (Int32, gather index)
    coarsest_csr::MA          # device coarsest operator (natural order, frozen pattern)
    coarsest_host::HCS        # host coarsest CSR (frozen pattern) for D->H + lu refactor
    omega_nominal::VT         # 1-elt device buffer holding the nominal omega cap (scalar, kept as field)
    comp_dst::Any             # device Int32 n_fine: composed row map for fused-level scatters (empty if fused_top==0)
    comp_s::Any               # device T n_fine: composed 1/sqrt(w) chain weights (empty if fused_top==0)
    eig::Vector{Any}          # per-level persistent power-iteration eigenvector (lambda warm-start)
    eig_warm::Vector{Bool}    # eig[l] holds a converged previous eigenvector (false until first refresh)
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
    dev(v) = bk isa CPU ? copy(v) : Adapt.adapt(bk, v)
    A_perms = handle.A_perms; operators = handle.operators; cpis = handle.cpis; aos = handle.aos

    # per-level pattern totality also guarantees the composed (fused-skip) scatter is total: the image
    # of each hop is contained in the next level's frozen pattern by induction.
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
    n1 = st.levels[1].n
    comp_dst = st.fused_top > 0 ? dev(zeros(Int32, n1)) : dev(Int32[])
    comp_s = st.fused_top > 0 ? dev(zeros(T, n1)) : dev(T[])
    eig = Any[dev(zeros(eltype(lv.invdiag), lv.n)) for lv in st.levels]
    return MFRefreshPlan(dev(handle.pvms[1]), coarsest_csr, coarsest_host,
                         dev(T[st.omega_nominal]), comp_dst, comp_s, eig, fill(false, M))
end

# Device power iteration for λ_max(D⁻¹A) on a frozen operator, reusing level scratch (tmp/r/rhs).
# Same math as reference _estimate_lambda_max but seed-agnostic (returns the dominant eigenvalue);
# the Gershgorin floor is permutation-invariant so we do NOT chase the build's natural-order value.
# v is the persistent plan eigenvector: warm=true seeds from the previous refresh's eigenvector and
# needs far fewer iterations (the operator drifts slowly between outer iterations).
function _lambda_max_device!(lv::MFLevel, v, warm::Bool, bk, wg; iters::Int=5, warm_iters::Int=2)
    A = lv.A; rp, cv, nz = _rowptr(A), _colval(A), _nzval(A); T = eltype(lv.invdiag)
    wraw = lv.tmp; w = lv.r; gb = lv.rhs; n = lv.n
    if !warm
        _launch_amg_kernel!(bk, wg, _amg_seed_kernel!, n, v)
        KernelAbstractions.synchronize(bk)
        v ./= sqrt(T(n))
    end
    lambda = one(T)
    for _ in 1:(warm ? warm_iters : iters)
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

# λ_max(D⁻¹A_l) for a fused (un-materialized) level: A_l·v via the exact Galerkin chain. The
# Gershgorin floor uses the composed |·| row-sum bound already staged in lv.rhs by the caller.
function _lambda_max_matfree!(st::MFMLState, l::Int, v, warm::Bool, bk, wg; iters::Int=5, warm_iters::Int=2)
    lv = st.levels[l]; T = eltype(lv.invdiag)
    cy = st.levels[1].tmp; cz = st.levels[1].r
    wraw = lv.tmp; w = lv.r; n = lv.n
    if !warm
        _launch_amg_kernel!(bk, wg, _amg_seed_kernel!, n, v)
        KernelAbstractions.synchronize(bk)
        v ./= sqrt(T(n))
    end
    lambda = one(T)
    for _ in 1:(warm ? warm_iters : iters)
        _mf_apply_operator!(wraw, st, l, v, cy, cz, bk, wg)
        KernelAbstractions.synchronize(bk)
        w .= wraw .* lv.invdiag
        lambda = max(T(norm(w)), eps(T))
        v .= w ./ lambda
    end
    lv.rhs .= lv.rhs .* abs.(lv.invdiag)  # rhs holds the composed abs row-sum bound
    KernelAbstractions.synchronize(bk)
    return max(lambda, T(maximum(lv.rhs)), one(T))
end

# NEW SECTION: the refresh entry point

# Refresh ALL numeric data of a built MFMLState from a new operator A2 (SAME sparsity as the original
# fine A) with frozen structure: finest values (host gather + one H->D) -> cascaded coarse RAP scatter
# (device, recomputed) -> per-level invdiag + omega (device) -> coarsest LU refactor. No re-aggregation,
# no host RAP, no realloc. Lean: no per-nonzero scatter maps, no device value buffer (see MFRefreshPlan).
# coarse=false is the light per-timestep refresh (mirrors the reference coarse_refresh_interval
# off-iteration): finest values + finest invdiag only; the coarse cascade, smoother lambdas and the
# coarsest factorization stay frozen until the next full refresh. Preconditioner-only staleness —
# the outer Krylov reads the live A, so correctness is unaffected.
function refresh_mf_ml!(st::MFMLState, plan::MFRefreshPlan, A2; omega_nominal=nothing, coarse::Bool=true)
    bk = st.backend; wg = 256; M = length(st.levels)
    finest_nz = _nzval(st.levels[1].A); T = eltype(finest_nz)
    TC = eltype(st.coarse_rhs)  # coarse-level (>=2) + coarsest storage type (split precision)
    omega_cap = omega_nominal === nothing ? st.omega_nominal : T(omega_nominal)
    # finest: device gather of A2's values into A_perm_1 order (frozen permutation). The CFD system
    # matrix is already device-resident, so one gather kernel replaces the old D->H + host gather +
    # H->D round-trip. adapt is a no-op on a device A2 (zero copy); only a host A2 (validation driver)
    # pays one H->D. _amg_gather_kernel! converts precision in the assignment (finest is F64 == A2 F64).
    fvm = plan.finest_value_map
    # A device-resident A2 (production CFD) is gathered in place — genuinely zero-copy. Only a host Array
    # (the mf_ml_refresh_error validation driver) pays one H->D. Guard on Base Array so no CUDA dep here.
    nz = _nzval(A2)
    src_nz = (bk isa CPU || !(nz isa Array)) ? nz : Adapt.adapt(bk, nz)
    # one-time directive telemetry: confirms the gather source is a native GPU array under production CFD.
    bk isa CPU || @info "AMG gf refresh: finest gather source" type=typeof(src_nz) eltype=eltype(src_nz) zero_copy=(src_nz === nz) maxlog=1
    _launch_amg_kernel!(bk, wg, _amg_gather_kernel!, length(fvm), finest_nz, src_nz, fvm)
    if !coarse
        lvf = st.levels[1]
        _launch_amg_kernel!(bk, wg, _amg_invdiag_kernel!, lvf.n, lvf.invdiag, _nzval(lvf.A), lvf.diag_index)
        KernelAbstractions.synchronize(bk)
        return st
    end
    ft = st.fused_top
    lv1 = st.levels[1]; A1 = lv1.A
    if ft > 0
        # fused (un-materialized) levels: walk the composed maps down the block, refreshing each
        # level's invdiag (diagonal scatter) + omega (matrix-free power iteration), then deliver the
        # values of the first materialized operator below the block in ONE composed scatter from A_0.
        _launch_amg_kernel!(bk, wg, _amg_compose_init_kernel!, lv1.n, plan.comp_dst, plan.comp_s)
        _launch_amg_kernel!(bk, wg, _amg_compose_hop_kernel!, lv1.n, plan.comp_dst, plan.comp_s,
                            lv1.row_macro, lv1.coarse_pos, lv1.inv_sqrt_w)
        for l in 2:(ft + 1)
            lv = st.levels[l]
            _fill_device!(bk, lv.invdiag, zero(eltype(lv.invdiag)))
            _fill_device!(bk, lv.rhs, zero(eltype(lv.rhs)))
            _launch_amg_kernel!(bk, wg, _amg_diag_scatter_composed_kernel!, lv1.n, lv.invdiag,
                                _rowptr(A1), _colval(A1), _nzval(A1), plan.comp_dst, plan.comp_s)
            _launch_amg_kernel!(bk, wg, _amg_absrowsum_scatter_composed_kernel!, lv1.n, lv.rhs,
                                _rowptr(A1), _colval(A1), _nzval(A1), plan.comp_dst, plan.comp_s)
            KernelAbstractions.synchronize(bk)
            lv.rhs .+= abs.(lv.invdiag)  # invdiag still holds the raw assembled diagonal here
            _launch_amg_kernel!(bk, wg, _amg_invert_diag_kernel!, lv.n, lv.invdiag)
            KernelAbstractions.synchronize(bk)
            lambda = _lambda_max_matfree!(st, l, plan.eig[l], plan.eig_warm[l], bk, wg)
            lv.omega = min(omega_cap, T(2) - eps(T)) / T(lambda)
            plan.eig_warm[l] = true
            _launch_amg_kernel!(bk, wg, _amg_compose_hop_kernel!, lv1.n, plan.comp_dst, plan.comp_s,
                                lv.row_macro, lv.coarse_pos, lv.inv_sqrt_w)
        end
        tgt = ft + 2 <= M ? st.levels[ft + 2].A : plan.coarsest_csr
        dst_nz = _nzval(tgt)
        _fill_device!(bk, dst_nz, zero(eltype(dst_nz)))
        _launch_amg_kernel!(bk, wg, _amg_rap_scatter_composed_kernel!, lv1.n, dst_nz,
                            _rowptr(A1), _colval(A1), _nzval(A1), plan.comp_dst, plan.comp_s,
                            _rowptr(tgt), _colval(tgt))
    end
    # cascade the materialized coarse operators: zero target, scatter-add from the (already refreshed)
    # fine level, recomputing dst+scale in-kernel from resident transfer factors (1 thread/source row)
    for l in (ft == 0 ? 1 : ft + 2):M
        lv = st.levels[l]; src = lv.A
        tgt = l < M ? st.levels[l + 1].A : plan.coarsest_csr
        dst_nz = _nzval(tgt)
        _fill_device!(bk, dst_nz, zero(eltype(dst_nz)))
        _launch_amg_kernel!(bk, wg, _amg_rap_scatter_recompute_kernel!, lv.n, dst_nz,
                            _rowptr(src), _colval(src), _nzval(src),
                            lv.row_macro, lv.coarse_pos, lv.inv_sqrt_w,
                            _rowptr(tgt), _colval(tgt))
    end
    # materialized-level smoother data: invdiag from frozen diag_index, omega from warm-started power
    # iteration on the persistent plan eigenvector (fused levels were refreshed in the block above)
    for l in 1:M
        _mf_is_matfree(st, l) && continue
        lv = st.levels[l]
        _launch_amg_kernel!(bk, wg, _amg_invdiag_kernel!, lv.n, lv.invdiag, _nzval(lv.A), lv.diag_index)
        KernelAbstractions.synchronize(bk)
        lambda = _lambda_max_device!(lv, plan.eig[l], plan.eig_warm[l], bk, wg)
        lv.omega = min(omega_cap, T(2) - eps(T)) / T(lambda)  # cap in T; coarse lambda is at level precision
        plan.eig_warm[l] = true
    end
    # coarsest: D->H the refreshed coarsest operator into the frozen host pattern, refresh whichever
    # coarse mechanism is active (device dense inverse for GEMV, else refactor host LU)
    KernelAbstractions.synchronize(bk)
    copyto!(_nzval(plan.coarsest_host), Array(_nzval(plan.coarsest_csr)))
    coarse_csc = _csr_to_csc(plan.coarsest_host)
    st.coarse_fac = lu(coarse_csc)
    st.coarse_inv === nothing ||
        (st.coarse_inv = _build_coarse_dense_inv(coarse_csc, bk, TC, st.coarse_max_rows))
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
                             omega_nominal=4/3, max_coarse::Integer=64, coarse_storage=nothing,
                             fused_top::Integer=0)
    h1 = _build_mf_ml(A1, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                      max_coarse=max_coarse, fused_top=fused_top, coarse_storage=coarse_storage)
    plan = build_mf_refresh_plan(h1)
    refresh_mf_ml!(h1.st, plan, A2; omega_nominal=omega_nominal)
    ops, lam = _frozen_rap_oracle(h1, A2); T = h1.T; M = length(h1.st.levels)
    relerr = 0.0; omega_relerr = 0.0
    for l in 1:M
        lv = h1.st.levels[l]
        if _mf_is_matfree(h1.st, l)
            # no stored values at a fused level: validate the refreshed invdiag against the oracle diag
            _, invd_ref = _diag_inverse(ops[l])
            got_d = Array(lv.invdiag)
            relerr = max(relerr, maximum(abs.(got_d .- invd_ref)) / max(maximum(abs.(invd_ref)), eps(T)))
        else
            got = Array(_nzval(lv.A)); ref = _nzval(ops[l])
            length(got) == length(ref) || error("refresh/oracle pattern mismatch at level $l")
            relerr = max(relerr, maximum(abs.(got .- ref)) / max(maximum(abs.(ref)), eps(T)))
        end
        om_ref = min(T(omega_nominal), T(2) - eps(T)) / T(lam[l])
        omega_relerr = max(omega_relerr, abs(lv.omega - om_ref) / max(abs(om_ref), eps(T)))
    end
    cg = Array(_nzval(plan.coarsest_csr)); cr = _nzval(ops[end])
    relerr = max(relerr, maximum(abs.(cg .- cr)) / max(maximum(abs.(cr)), eps(T)))
    return (relerr=relerr, omega_relerr=omega_relerr, levels=M + 1, n=h1.st.n)
end

# Deployment check (advisor #4): operator-match alone does not prove a REFRESHED mixed-prec state
# converges. Build on A1 -> refresh to A2 -> drive the stationary MF V-cycle on A2 and report iters.
# Mirrors mf_ml_convergence but on the refreshed state (st.levels[1].A holds A2's permuted finest).
function mf_ml_refresh_convergence(A1, A2, merge_levels::Integer, backend; pre::Int=2, post::Int=2,
                                   itmax::Int=200, rtol=1e-8, omega_nominal=4/3, max_coarse::Integer=64,
                                   coarse_storage=nothing, fused_top::Integer=0)
    h1 = _build_mf_ml(A1, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                      max_coarse=max_coarse, fused_top=fused_top, coarse_storage=coarse_storage)
    plan = build_mf_refresh_plan(h1)
    refresh_mf_ml!(h1.st, plan, A2; omega_nominal=omega_nominal)
    st = h1.st; T = h1.T; n = st.n; bk = backend; wg = 256
    Afine = st.levels[1].A; rp, cv, nz = _rowptr(Afine), _colval(Afine), _nzval(Afine)
    dev(v) = bk isa CPU ? copy(v) : Adapt.adapt(bk, v)
    b = dev(rand(T, n)); x = dev(zeros(T, n)); res = dev(zeros(T, n))
    _launch_amg_kernel!(bk, wg, _amg_csr_residual_kernel!, n, res, rp, cv, nz, x, b)
    KernelAbstractions.synchronize(bk)
    r0 = norm(Array(res)); rn = r0; it = 0
    while it < itmax && rn > rtol * r0
        it += 1
        dx = mf_ml_cycle(st, res)
        _launch_amg_kernel!(bk, wg, _amg_add_kernel!, n, x, dx)
        _launch_amg_kernel!(bk, wg, _amg_csr_residual_kernel!, n, res, rp, cv, nz, x, b)
        KernelAbstractions.synchronize(bk)
        rn = norm(Array(res))
    end
    return (converged=(rn <= rtol * r0), iters=it, final_rel=rn / r0, n=n, levels=length(st.levels) + 1)
end
