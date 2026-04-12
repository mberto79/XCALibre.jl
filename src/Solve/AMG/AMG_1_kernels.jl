# ─── Backend-agnostic sparse/vector KA kernels used by the AMG hierarchy ──────
# All kernels follow the _setup(backend, workgroup, ndrange) idiom.

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

# ── Type-converting copy: dst[i] = convert(eltype(dst), src[i]) ───────────────
# Handles Float64→Float32 at the fine→coarse restriction boundary and
# Float32→Float64 at the coarse→fine prolongation boundary.

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

# ── Compute inverse diagonal: Dinv[i] = 1 / A[i,i] ───────────────────────────
# Slow fallback: searches each row for the diagonal entry (causes warp divergence on GPU).

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

# ── Fast inverse diagonal using pre-computed diagonal pointer ─────────────────
# diag_ptr[row] holds the 1-based nzval index of the diagonal entry — no row scan, no warp divergence.

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
# Returns Vector{Int32} where ptr[i] is the 1-based nzval index of A[i,i].

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
# Denominator is the l1 row norm (includes diagonal). For FVM M-matrices this
# bounds ρ(D_l1⁻¹A) ≤ 1 by construction, making Jacobi convergent with ω = 1.
# Reference: Baker, Falgout, Kolev, Yang, SISC 2011 (hypre BoomerAMG default).

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

# ── L1-Jacobi sweep (fused, ping-pong) ───────────────────────────────────────
# Full row sum (diagonal included); x_new[i] = x[i] + ω · Dinv_l1[i] · r[i].
# Correct for D_l1⁻¹ = 1/||a_i||_1; converges to the true solution of Ax=b.

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

# ── L2 norm (delegates to LinearAlgebra on CPU; uses dot on GPU via LA) ────────
# Note: LinearAlgebra.norm works on GPU arrays via GPUArrays.jl/CUDA extensions.
amg_norm(v) = norm(v)

# ── Jacobi sweep kernel (used by AMG internal smoothing) ──────────────────────
# One sweep: x_new[i] = ω/a_ii * (b[i] - Σ_{j≠i} a_ij*x[j]) + (1-ω)*x[i]

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

# ── Correction-form Jacobi update: x[i] += ω * Dinv[i] * r[i] ────────────────
# Equivalent to damped Jacobi without a tmp-buffer swap.

@kernel function _amg_dinv_axpy!(x, Dinv, r, omega)
    i = @index(Global)
    @inbounds x[i] += omega * Dinv[i] * r[i]
end

function amg_dinv_axpy!(x, Dinv, r, omega, backend, workgroup)
    n = length(x)
    kernel! = _amg_dinv_axpy!(_setup(backend, workgroup, n)...)
    kernel!(x, Dinv, r, omega)
end

# ── D⁻¹-scaled AXPBY: y[i] = alpha * Dinv[i] * x[i] + beta * y[i] ───────────
# Used by Chebyshev smoother direction update: p = alpha * D⁻¹r + beta * p_old.

@kernel function _amg_dinv_axpby!(y, Dinv, x, alpha, beta)
    i = @index(Global)
    @inbounds y[i] = alpha * Dinv[i] * x[i] + beta * y[i]
end

function amg_dinv_axpby!(y, Dinv, x, alpha, beta, backend, workgroup)
    n = length(y)
    kernel! = _amg_dinv_axpby!(_setup(backend, workgroup, n)...)
    kernel!(y, Dinv, x, alpha, beta)
end

# ── GPU-resident RAP: Ac = R·A·P (one thread per coarse row) ─────────────────
# Exploits unsmoothed aggregation: P has exactly 1 nnz per fine row, so
# agg[j] = _colval(L.P)[j] gives the coarse aggregate of fine node j.
# Ac rows are short (≤ 30 entries for 3D FVM), so the inner linear search
# into Ac_colval has negligible cost and avoids any shared-memory allocation.
# All arrays remain on-device; no PCIe transfer.

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
        # scatter R[c1, i] * A[i, j] into Ac[c1, agg[j]]
        # Cast both operands to TOut so the fine-level Float64 A does not
        # promote the entire scatter to Float64 on the GPU.
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

