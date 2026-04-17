# ─── Backend-agnostic sparse/vector KA kernels used by the AMG hierarchy ──────

# ── SpMV: y = A * x  (CSR, row-per-workitem) ──────────────────────────────────

@kernel function _amg_spmv!(y, rowptr, colval, nzval, x)
    row = @index(Global)
    @inbounds begin
        acc = zero(eltype(nzval))
        for nzi in rowptr[row]:(rowptr[row+1] - 1)
            acc += nzval[nzi] * x[colval[nzi]]
        end
        y[row] = acc
    end
end

function amg_spmv!(y, A, x, backend, workgroup)
    nzval, colval, rowptr = get_sparse_fields(A)
    n = length(y)
    kernel! = _amg_spmv!(_setup(backend, workgroup, n)...)
    kernel!(y, rowptr, colval, nzval, x)
end

# ── SpMV accumulate: y += α * A * x ───────────────────────────────────────────

@kernel function _amg_spmv_add!(y, rowptr, colval, nzval, x, alpha)
    row = @index(Global)
    @inbounds begin
        acc = zero(eltype(nzval))
        for nzi in rowptr[row]:(rowptr[row+1] - 1)
            acc += nzval[nzi] * x[colval[nzi]]
        end
        y[row] += alpha * acc
    end
end

function amg_spmv_add!(y, A, x, alpha, backend, workgroup)
    nzval, colval, rowptr = get_sparse_fields(A)
    n = length(y)
    kernel! = _amg_spmv_add!(_setup(backend, workgroup, n)...)
    kernel!(y, rowptr, colval, nzval, x, alpha)
end

# ── AXPY: y = y + α * x ───────────────────────────────────────────────────────

@kernel function _amg_axpy!(y, x, alpha)
    i = @index(Global)
    @inbounds y[i] += alpha * x[i]
end

function amg_axpy!(y, x, alpha, backend, workgroup)
    n = length(y)
    kernel! = _amg_axpy!(_setup(backend, workgroup, n)...)
    kernel!(y, x, alpha)
end

# ── AXPBY: y = α * x + β * y ──────────────────────────────────────────────────

@kernel function _amg_axpby!(y, x, alpha, beta)
    i = @index(Global)
    @inbounds y[i] = alpha * x[i] + beta * y[i]
end

function amg_axpby!(y, x, alpha, beta, backend, workgroup)
    n = length(y)
    kernel! = _amg_axpby!(_setup(backend, workgroup, n)...)
    kernel!(y, x, alpha, beta)
end

# ── COPY: dst = src ────────────────────────────────────────────────────────────

@kernel function _amg_copy!(dst, src)
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

function amg_copy!(dst, src, backend, workgroup)
    n = length(dst)
    kernel! = _amg_copy!(_setup(backend, workgroup, n)...)
    kernel!(dst, src)
end

# ── Type-converting copy (mixed-precision boundaries: fine↔coarse) ─────────────

@kernel function _amg_cast_copy!(dst, src)
    i = @index(Global)
    @inbounds dst[i] = convert(eltype(dst), src[i])
end

function amg_cast_copy!(dst, src, backend, workgroup)
    n = length(dst)
    kernel! = _amg_cast_copy!(_setup(backend, workgroup, n)...)
    kernel!(dst, src)
end

# ── ZERO: v .= 0 ───────────────────────────────────────────────────────────────

@kernel function _amg_zero!(v)
    i = @index(Global)
    @inbounds v[i] = zero(eltype(v))
end

function amg_zero!(v, backend, workgroup)
    n = length(v)
    kernel! = _amg_zero!(_setup(backend, workgroup, n)...)
    kernel!(v)
end

# ── Compute inverse diagonal: Dinv[i] = 1 / A[i,i] (slow: row scan, warp divergence) ────

@kernel function _amg_build_Dinv!(Dinv, rowptr, colval, nzval)
    row = @index(Global)
    @inbounds begin
        d = zero(eltype(nzval))
        for nzi in rowptr[row]:(rowptr[row+1] - 1)
            if colval[nzi] == row
                d = nzval[nzi]
                break
            end
        end
        Dinv[row] = ifelse(d != zero(eltype(nzval)), one(eltype(nzval)) / d, one(eltype(nzval)))
    end
end

function amg_build_Dinv!(Dinv, A, backend, workgroup)
    nzval, colval, rowptr = get_sparse_fields(A)
    n = length(Dinv)
    kernel! = _amg_build_Dinv!(_setup(backend, workgroup, n)...)
    kernel!(Dinv, rowptr, colval, nzval)
end

# ── Fast inverse diagonal using pre-computed diagonal pointer (no row scan) ────

@kernel function _amg_build_Dinv_fast!(Dinv, nzval, diag_ptr)
    row = @index(Global)
    @inbounds begin
        d = nzval[diag_ptr[row]]
        Dinv[row] = ifelse(d != zero(eltype(nzval)), one(eltype(nzval)) / d, one(eltype(nzval)))
    end
end

function amg_build_Dinv!(Dinv, A, diag_ptr::AbstractVector{<:Integer}, backend, workgroup)
    nzval, _, _ = get_sparse_fields(A)
    n = length(Dinv)
    kernel! = _amg_build_Dinv_fast!(_setup(backend, workgroup, n)...)
    kernel!(Dinv, nzval, diag_ptr)
end

# ── Build diagonal pointer on CPU ─────────────────────────────────────────────

function _build_diag_ptr_cpu(A::SparseMatricesCSR.SparseMatrixCSR)
    n   = size(A, 1)
    ptr = zeros(Int32, n)
    @inbounds for i in 1:n
        for nzi in A.rowptr[i]:(A.rowptr[i+1]-1)
            if A.colval[nzi] == i
                ptr[i] = Int32(nzi)
                break
            end
        end
    end
    return ptr
end

# ── Build l1-scaled inverse diagonal: Dinv[i] = 1 / Σ_j |a_ij| ──────────────
# For FVM M-matrices: ρ(D_l1⁻¹A) ≤ 1 by construction, converges with ω=1.

@kernel function _amg_build_l1_Dinv!(Dinv, rowptr, nzval)
    row = @index(Global)
    @inbounds begin
        s = zero(eltype(nzval))
        for nzi in rowptr[row]:(rowptr[row+1] - 1)
            s += abs(nzval[nzi])
        end
        Dinv[row] = ifelse(s > zero(eltype(nzval)), one(eltype(nzval)) / s, one(eltype(nzval)))
    end
end

function amg_build_l1_Dinv!(Dinv, A, backend, workgroup)
    nzval, _, rowptr = get_sparse_fields(A)
    n = length(Dinv)
    kernel! = _amg_build_l1_Dinv!(_setup(backend, workgroup, n)...)
    kernel!(Dinv, rowptr, nzval)
end

# ── Residual r = b - A*x (single pass) ────────────────────────────────────────

@kernel function _amg_residual!(r, rowptr, colval, nzval, x, b)
    row = @index(Global)
    @inbounds begin
        acc = zero(eltype(nzval))
        for nzi in rowptr[row]:(rowptr[row+1] - 1)
            acc += nzval[nzi] * x[colval[nzi]]
        end
        r[row] = b[row] - acc
    end
end

function amg_residual!(r, A, x, b, backend, workgroup)
    nzval, colval, rowptr = get_sparse_fields(A)
    n = length(r)
    kernel! = _amg_residual!(_setup(backend, workgroup, n)...)
    kernel!(r, rowptr, colval, nzval, x, b)
end

# ── L1-Jacobi sweep (fused, ping-pong); full row sum including diagonal ────────

@kernel function _amg_l1jacobi_sweep!(x_new, x, Dinv_l1, rowptr, colval, nzval, b, omega)
    row = @index(Global)
    @inbounds begin
        acc = zero(eltype(nzval))
        for nzi in rowptr[row]:(rowptr[row+1] - 1)
            acc += nzval[nzi] * x[colval[nzi]]   # full row including diagonal
        end
        x_new[row] = x[row] + omega * Dinv_l1[row] * (b[row] - acc)
    end
end

function amg_l1jacobi_sweep!(x_new, x, Dinv_l1, A, b, omega, backend, workgroup)
    nzval, colval, rowptr = get_sparse_fields(A)
    n = length(x)
    kernel! = _amg_l1jacobi_sweep!(_setup(backend, workgroup, n)...)
    kernel!(x_new, x, Dinv_l1, rowptr, colval, nzval, b, omega)
end

amg_norm(v) = norm(v)

# ── Jacobi sweep kernel (used by AMG internal smoothing) ──────────────────────

@kernel function _amg_jacobi_sweep!(x_new, x, Dinv, rowptr, colval, nzval, b, omega)
    row = @index(Global)
    @inbounds begin
        acc = zero(eltype(nzval))
        for nzi in rowptr[row]:(rowptr[row+1] - 1)
            j = colval[nzi]
            if j != row
                acc += nzval[nzi] * x[j]
            end
        end
        x_new[row] = omega * Dinv[row] * (b[row] - acc) + (one(eltype(nzval)) - omega) * x[row]
    end
end

function amg_jacobi_sweep!(x_new, x, Dinv, A, b, omega, backend, workgroup)
    nzval, colval, rowptr = get_sparse_fields(A)
    n = length(x)
    kernel! = _amg_jacobi_sweep!(_setup(backend, workgroup, n)...)
    kernel!(x_new, x, Dinv, rowptr, colval, nzval, b, omega)
end

# ── Correction-form Jacobi update (damped without buffer swap) ───────────────

@kernel function _amg_dinv_axpy!(x, Dinv, r, omega)
    i = @index(Global)
    @inbounds x[i] += omega * Dinv[i] * r[i]
end

function amg_dinv_axpy!(x, Dinv, r, omega, backend, workgroup)
    n = length(x)
    kernel! = _amg_dinv_axpy!(_setup(backend, workgroup, n)...)
    kernel!(x, Dinv, r, omega)
end

# ── D⁻¹-scaled AXPBY (used by Chebyshev smoother direction update) ────────────

@kernel function _amg_dinv_axpby!(y, Dinv, x, alpha, beta)
    i = @index(Global)
    @inbounds y[i] = alpha * Dinv[i] * x[i] + beta * y[i]
end

function amg_dinv_axpby!(y, Dinv, x, alpha, beta, backend, workgroup)
    n = length(y)
    kernel! = _amg_dinv_axpby!(_setup(backend, workgroup, n)...)
    kernel!(y, Dinv, x, alpha, beta)
end

# ── Jacobi correction (on entry x_new holds Ax; on exit, the updated iterate) ─

@kernel function _amg_jacobi_correction!(x_new, x, Dinv, b, omega)
    i = @index(Global)
    @inbounds x_new[i] = x[i] + omega * Dinv[i] * (b[i] - x_new[i])
end

function amg_jacobi_correction!(x_new, x, Dinv, b, omega, backend, workgroup)
    n = length(x_new)
    kernel! = _amg_jacobi_correction!(_setup(backend, workgroup, n)...)
    kernel!(x_new, x, Dinv, b, omega)
end

# ── GPU-resident RAP: Ac = R·A·P (one thread per coarse row, unsmoothed P) ────
# P has 1 nnz/row; agg[j] = colval(P)[j] = coarse aggregate of fine node j.
# Linear search in short Ac rows; all on-device, no PCIe.

@kernel function _amg_rap_row!(
    Ac_nzval, Ac_rowptr, Ac_colval,
    A_nzval,  A_rowptr,  A_colval,
    R_nzval,  R_rowptr,  R_colval,
    agg)
    c1 = @index(Global)
    @inbounds begin
        ac_start = Ac_rowptr[c1]
        ac_end   = Ac_rowptr[c1 + 1] - 1
        TOut = eltype(Ac_nzval)
        # zero the output row
        for k in ac_start:ac_end
            Ac_nzval[k] = zero(TOut)
        end
        # scatter R[c1, i] * A[i, j] into Ac[c1, agg[j]]; cast to TOut to avoid Float64 promotion
        for r_idx in R_rowptr[c1]:(R_rowptr[c1 + 1] - 1)
            i   = R_colval[r_idx]
            riv = TOut(R_nzval[r_idx])
            for a_idx in A_rowptr[i]:(A_rowptr[i + 1] - 1)
                j  = A_colval[a_idx]
                c2 = agg[j]
                v  = riv * TOut(A_nzval[a_idx])
                for k in ac_start:ac_end
                    if Ac_colval[k] == c2
                        Ac_nzval[k] += v
                        break
                    end
                end
            end
        end
    end
end

function amg_rap_update!(Lc, L, backend, workgroup)
    Ac_nzval, Ac_colval, Ac_rowptr = get_sparse_fields(Lc.A)
    A_nzval,  A_colval,  A_rowptr  = get_sparse_fields(L.A)
    R_nzval,  R_colval,  R_rowptr  = get_sparse_fields(L.R)
    agg = _colval(L.P)   # P.colval[j] = coarse aggregate of fine node j
    nc  = length(Lc.Dinv)
    kernel! = _amg_rap_row!(_setup(backend, workgroup, nc)...)
    kernel!(Ac_nzval, Ac_rowptr, Ac_colval,
            A_nzval,  A_rowptr,  A_colval,
            R_nzval,  R_rowptr,  R_colval,
            agg)
end

# ── GPU-resident RAP for smoothed P: Ac = R·A·P (general multi-nnz P) ─────────
# Same thread-per-coarse-row, but P can have multiple nnz/row (smoothed prolongation).

@kernel function _amg_rap_row_smooth!(
    Ac_nzval, Ac_rowptr, Ac_colval,
    A_nzval,  A_rowptr,  A_colval,
    R_nzval,  R_rowptr,  R_colval,
    P_nzval,  P_rowptr,  P_colval)
    c1 = @index(Global)
    @inbounds begin
        ac_start = Ac_rowptr[c1]
        ac_end   = Ac_rowptr[c1 + 1] - 1
        TOut = eltype(Ac_nzval)
        for k in ac_start:ac_end
            Ac_nzval[k] = zero(TOut)
        end
        for r_idx in R_rowptr[c1]:(R_rowptr[c1 + 1] - 1)
            i   = R_colval[r_idx]
            riv = TOut(R_nzval[r_idx])
            for a_idx in A_rowptr[i]:(A_rowptr[i + 1] - 1)
                j   = A_colval[a_idx]
                aiv = riv * TOut(A_nzval[a_idx])
                for p_idx in P_rowptr[j]:(P_rowptr[j + 1] - 1)
                    c2 = P_colval[p_idx]
                    v  = aiv * TOut(P_nzval[p_idx])
                    for k in ac_start:ac_end
                        if Ac_colval[k] == c2
                            Ac_nzval[k] += v
                            break
                        end
                    end
                end
            end
        end
    end
end

function amg_rap_update_smooth!(Lc, L, backend, workgroup)
    Ac_nzval, Ac_colval, Ac_rowptr = get_sparse_fields(Lc.A)
    A_nzval,  A_colval,  A_rowptr  = get_sparse_fields(L.A)
    R_nzval,  R_colval,  R_rowptr  = get_sparse_fields(L.R)
    P_nzval,  P_colval,  P_rowptr  = get_sparse_fields(L.P)
    nc  = length(Lc.Dinv)
    kernel! = _amg_rap_row_smooth!(_setup(backend, workgroup, nc)...)
    kernel!(Ac_nzval, Ac_rowptr, Ac_colval,
            A_nzval,  A_rowptr,  A_colval,
            R_nzval,  R_rowptr,  R_colval,
            P_nzval,  P_rowptr,  P_colval)
end

# ── In-place A·P update into pre-allocated AP (one thread per fine row) ────────
# Fixed sparsity pattern at setup; only nzval updates. Eliminates SpGEMM allocation.

@kernel function _amg_ap_update!(
    AP_nzval, AP_rowptr, AP_colval,
    A_nzval,  A_rowptr,  A_colval,
    P_nzval,  P_rowptr,  P_colval)
    i = @index(Global)   # fine row
    @inbounds begin
        ap_start = AP_rowptr[i]
        ap_end   = AP_rowptr[i + 1] - 1
        TOut = eltype(AP_nzval)
        for k in ap_start:ap_end
            AP_nzval[k] = zero(TOut)
        end
        for a_idx in A_rowptr[i]:(A_rowptr[i + 1] - 1)
            j   = A_colval[a_idx]
            aiv = TOut(A_nzval[a_idx])
            for p_idx in P_rowptr[j]:(P_rowptr[j + 1] - 1)
                c2 = P_colval[p_idx]
                v  = aiv * TOut(P_nzval[p_idx])
                for k in ap_start:ap_end
                    if AP_colval[k] == c2
                        AP_nzval[k] += v
                        break
                    end
                end
            end
        end
    end
end

function amg_ap_update!(AP, L, backend, workgroup)
    AP_nzval, AP_colval, AP_rowptr = get_sparse_fields(AP)
    A_nzval,  A_colval,  A_rowptr  = get_sparse_fields(L.A)
    P_nzval,  P_colval,  P_rowptr  = get_sparse_fields(L.P)
    nf = length(L.Dinv)
    kernel! = _amg_ap_update!(_setup(backend, workgroup, nf)...)
    kernel!(AP_nzval, AP_rowptr, AP_colval,
            A_nzval,  A_rowptr,  A_colval,
            P_nzval,  P_rowptr,  P_colval)
end

# ── In-place R·AP update into pre-allocated Ac (one thread per coarse row) ─────
# Second half of split RAP: scatters R[c1, i] * AP[i, c2] into Ac[c1, c2].

@kernel function _amg_rp_update!(
    Ac_nzval, Ac_rowptr, Ac_colval,
    AP_nzval, AP_rowptr, AP_colval,
    R_nzval,  R_rowptr,  R_colval)
    c1 = @index(Global)   # coarse row
    @inbounds begin
        ac_start = Ac_rowptr[c1]
        ac_end   = Ac_rowptr[c1 + 1] - 1
        TOut = eltype(Ac_nzval)
        for k in ac_start:ac_end
            Ac_nzval[k] = zero(TOut)
        end
        for r_idx in R_rowptr[c1]:(R_rowptr[c1 + 1] - 1)
            i   = R_colval[r_idx]
            riv = TOut(R_nzval[r_idx])
            for ap_idx in AP_rowptr[i]:(AP_rowptr[i + 1] - 1)
                c2 = AP_colval[ap_idx]
                v  = riv * TOut(AP_nzval[ap_idx])
                for k in ac_start:ac_end
                    if Ac_colval[k] == c2
                        Ac_nzval[k] += v
                        break
                    end
                end
            end
        end
    end
end

function amg_rp_update!(Lc, AP, L, backend, workgroup)
    Ac_nzval, Ac_colval, Ac_rowptr = get_sparse_fields(Lc.A)
    AP_nzval, AP_colval, AP_rowptr = get_sparse_fields(AP)
    R_nzval,  R_colval,  R_rowptr  = get_sparse_fields(L.R)
    nc = length(Lc.Dinv)
    kernel! = _amg_rp_update!(_setup(backend, workgroup, nc)...)
    kernel!(Ac_nzval, Ac_rowptr, Ac_colval,
            AP_nzval, AP_rowptr, AP_colval,
            R_nzval,  R_rowptr,  R_colval)
end

