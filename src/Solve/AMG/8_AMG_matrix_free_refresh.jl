# NEW SECTION

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
            Atomix.@atomic dst_nz[p] += T(swi * inv_sqrt_w[gj] * src_nzval[s])  # T cast: split precision (src/dst may differ)
        end
    end
end

@kernel function _amg_invdiag_kernel!(invdiag, @Const(nzval), @Const(diag_index))
    i = @index(Global)
    T = eltype(invdiag)
    @inbounds begin
        idx = diag_index[i]
        aii = idx == 0 ? one(T) : nzval[idx]
        invdiag[i] = abs(aii) > eps(T) ? inv(aii) : one(T)
    end
end

@kernel function _amg_seed_kernel!(v)
    i = @index(Global)
    T = eltype(v)
    @inbounds v[i] = isodd(i) ? one(T) : -one(T)
end

# NEW SECTION

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
            Atomix.@atomic dst_nz[p] += T(si * comp_s[j] * src_nzval[s])
        end
    end
end

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

# Cross-aggregate terms only; within-aggregate terms land on diagonal (caller adds |d_I|)
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

@kernel function _amg_invert_diag_kernel!(d)
    i = @index(Global)
    T = eltype(d)
    @inbounds begin
        aii = d[i]
        d[i] = abs(aii) > eps(T) ? inv(aii) : one(T)
    end
end

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

# NEW SECTION

mutable struct MatrixFreeRefreshPlan{DI, MA, HCS, VT}
    finest_value_map::DI
    coarsest_csr::MA
    coarsest_host::HCS
    omega_nominal::VT
    comp_dst::Any
    comp_s::Any
    eig::Vector{Any}
    eig_warm::Vector{Bool}
end

# In-kernel binary search requires every fine nonzero to map to an existing coarse slot
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

function build_refresh_plan(handle)
    st = handle.st; T = handle.T; bk = st.backend; M = handle.M
    dev(v) = bk isa CPU ? copy(v) : Adapt.adapt(bk, v)
    A_perms = handle.A_perms; operators = handle.operators; cpis = handle.cpis; aos = handle.aos

    for l in 1:M
        n = _m(A_perms[l]); nc = maximum(handle.aggs[l])
        _, row_macro = transfer_factors(aos[l], nc, n, T)
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
    return MatrixFreeRefreshPlan(dev(handle.pvms[1]), coarsest_csr, coarsest_host,
                         dev(T[st.omega_nominal]), comp_dst, comp_s, eig, fill(false, M))
end

# warm=true reuses previous eigenvector (fewer iters needed as operator drifts slowly)
function _lambda_max_device!(lv::MatrixFreeLevel, v, warm::Bool, bk, wg; iters::Int=5, warm_iters::Int=2)
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

function _lambda_max_matfree!(st::MatrixFreeHierarchy, l::Int, v, warm::Bool, bk, wg; iters::Int=5, warm_iters::Int=2)
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
        apply_fused_operator!(wraw, st, l, v, cy, cz, bk, wg)
        KernelAbstractions.synchronize(bk)
        w .= wraw .* lv.invdiag
        lambda = max(T(norm(w)), eps(T))
        v .= w ./ lambda
    end
    lv.rhs .= lv.rhs .* abs.(lv.invdiag)  # rhs holds composed abs row-sum bound
    KernelAbstractions.synchronize(bk)
    return max(lambda, T(maximum(lv.rhs)), one(T))
end

# NEW SECTION

# coarse=false: light refresh (finest only); staleness is preconditioner-only (outer Krylov reads live A)
function refresh_matrix_free_hierarchy!(st::MatrixFreeHierarchy, plan::MatrixFreeRefreshPlan, A2; omega_nominal=nothing, coarse::Bool=true)
    bk = st.backend; wg = 256; M = length(st.levels)
    finest_nz = _nzval(st.levels[1].A); T = eltype(finest_nz)
    TC = eltype(st.coarse_rhs)  # coarse-level (>=2) + coarsest storage type (split precision)
    omega_cap = omega_nominal === nothing ? st.omega_nominal : T(omega_nominal)
    fvm = plan.finest_value_map
    nz = _nzval(A2)
    src_nz = (bk isa CPU || !(nz isa Array)) ? nz : Adapt.adapt(bk, nz)
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
            lv.rhs .+= abs.(lv.invdiag)  # invdiag still holds the raw diagonal here (inverted next)
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
    for l in 1:M
        is_fused_level(st, l) && continue
        lv = st.levels[l]
        _launch_amg_kernel!(bk, wg, _amg_invdiag_kernel!, lv.n, lv.invdiag, _nzval(lv.A), lv.diag_index)
        KernelAbstractions.synchronize(bk)
        lambda = _lambda_max_device!(lv, plan.eig[l], plan.eig_warm[l], bk, wg)
        lv.omega = min(omega_cap, T(2) - eps(T)) / T(lambda)
        plan.eig_warm[l] = true
    end
    KernelAbstractions.synchronize(bk)
    copyto!(_nzval(plan.coarsest_host), Array(_nzval(plan.coarsest_csr)))
    coarse_csc = _csr_to_csc(plan.coarsest_host)
    st.coarse_fac[] = lu(coarse_csc)
    st.coarse_inv[] === nothing ||
        (st.coarse_inv[] = _build_coarse_dense_inv(coarse_csc, bk, TC, st.coarse_max_rows))
    return st
end
