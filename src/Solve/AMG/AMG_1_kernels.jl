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

# ── Residual r = b - A*x ───────────────────────────────────────────────────────

function amg_residual!(r, A, x, b, backend, workgroup)
    amg_spmv!(r, A, x, backend, workgroup)   # r = A*x
    # r = b - r
    n = length(r)
    kernel! = _amg_axpby!(_setup(backend, workgroup, n)...)
    kernel!(r, b, one(eltype(r)), -one(eltype(r)))  # r = 1*b + (-1)*r
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
