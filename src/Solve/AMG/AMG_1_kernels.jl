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
# diag_ptr[row] holds the 1-based nzval index of the diagonal entry.
# Eliminates the row-scan loop and all warp divergence — single indexed load per thread.

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
# Returns a Vector{Int32} of length n where ptr[i] is the 1-based nzval index
# of A[i,i]. Called once at amg_setup! time; the result is transferred to device.

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
# Replaces the two-buffer swap pattern. With r = b - Ax pre-computed, this
# is mathematically identical to damped Jacobi but requires no tmp-buffer swap
# and works cleanly with an immutable MultigridLevel struct.

@kernel function _amg_dinv_axpy!(x, Dinv, r, omega)
    i = @index(Global)
    @inbounds x[i] += omega * Dinv[i] * r[i]
end

function amg_dinv_axpy!(x, Dinv, r, omega, backend, workgroup)
    n = length(x)
    kernel! = _amg_dinv_axpy!(_setup(backend, workgroup, n)...)
    kernel!(x, Dinv, r, omega)
end

# ── In-place Jacobi correction: x[i] += ω · Dinv[i] · (b[i] - Ax[i]) ────────
# Fuses the residual SpMV and diagonal scaling into a single memory pass,
# eliminating the intermediate write and read of the residual vector.
# This is the asynchronous (chaotic) Jacobi: threads from different warps may
# read x values that have already been updated in the same kernel launch.
# Convergence for diagonally-dominant M-matrices is guaranteed by the Chazan–
# Miranker theorem and is the standard approach in GPU AMG implementations.

@kernel function _amg_jacobi!(x, Dinv, rowptr, colval, nzval, b, omega)
    row = @index(Global)
    @inbounds begin
        acc = zero(eltype(nzval))
        for nzi in rowptr[row]:(rowptr[row+1] - 1)
            acc += nzval[nzi] * x[colval[nzi]]
        end
        x[row] += omega * Dinv[row] * (b[row] - acc)
    end
end

function amg_smooth_jacobi!(x, Dinv, A, b, omega, backend, workgroup)
    nzval, colval, rowptr = get_sparse_fields(A)
    n = length(x)
    kernel! = _amg_jacobi!(_setup(backend, workgroup, n)...)
    kernel!(x, Dinv, rowptr, colval, nzval, b, omega)
end

# ── Fused Galerkin product: Ac = R · A · P (one thread per output nonzero) ────
#
# Each thread `out` accumulates the sum
#   Ac.nzval[out] = Σ_p  R.nzval[nzi_R[p]] · A.nzval[nzi_A[p]] · P.nzval[nzi_P[p]]
# for p in plan_rowptr[out] : plan_rowptr[out+1]-1.
#
# All arrays are device-resident: no CPU↔device transfer at update time.
# Works on CPU (multi-threaded via KA CPU backend) and any GPU backend.

@kernel function _amg_galerkin!(nzval_Ac, nzval_A, nzval_R, nzval_P,
                                  plan_rowptr, plan_nzi_R, plan_nzi_A, plan_nzi_P)
    out = @index(Global)
    @inbounds begin
        acc = zero(eltype(nzval_Ac))
        for p in plan_rowptr[out]:(plan_rowptr[out + 1] - 1)
            acc += nzval_R[plan_nzi_R[p]] * nzval_A[plan_nzi_A[p]] * nzval_P[plan_nzi_P[p]]
        end
        nzval_Ac[out] = acc
    end
end

function amg_galerkin!(Ac, A, R, P, plan::GalerkinPlan, backend, workgroup)
    nzval_Ac, _, _ = get_sparse_fields(Ac)
    nzval_A,  _, _ = get_sparse_fields(A)
    nzval_R,  _, _ = get_sparse_fields(R)
    nzval_P,  _, _ = get_sparse_fields(P)
    nnz_ac = length(nzval_Ac)
    kernel! = _amg_galerkin!(_setup(backend, workgroup, nnz_ac)...)
    kernel!(nzval_Ac, nzval_A, nzval_R, nzval_P,
            plan.rowptr, plan.nzi_R, plan.nzi_A, plan.nzi_P)
end
