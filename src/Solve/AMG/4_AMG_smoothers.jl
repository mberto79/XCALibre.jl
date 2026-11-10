@kernel function _amg_residual_kernel!(r, Ax, b)
    i = @index(Global)
    @inbounds r[i] = b[i] - Ax[i]
end

@kernel function _amg_fill_kernel!(x, value)
    i = @index(Global)
    @inbounds x[i] = value
end

@kernel function _amg_copy_kernel!(dest, src)
    i = @index(Global)
    @inbounds dest[i] = src[i]
end

# Residual form preserves bits near the fixed point; diagonal-excluded form injects Float32 noise
@kernel function _amg_jacobi_step_kernel!(x_new, x_old, b, rowptr, colval, nzval, invdiag, omega)
    i = @index(Global)
    T = eltype(x_new)
    sigma = zero(T)
    @inbounds begin
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            sigma += nzval[p] * x_old[colval[p]]
        end
        x_new[i] = x_old[i] + omega * invdiag[i] * (b[i] - sigma)
    end
end

@kernel function _amg_weighted_diagonal_correction_kernel!(x, residual, invdiag, omega)
    i = @index(Global)
    @inbounds x[i] += omega * invdiag[i] * residual[i]
end

function _amg_axpy!(hierarchy::AbstractAMGHierarchy, x, a, y)
    _launch_amg_kernel!(hierarchy, _amg_axpy_kernel!, length(x), x, a, y)
    return x
end

function _amg_scale_factor(num::T, den::T) where {T}
    (isfinite(num) && isfinite(den) && den > eps(T)) || return one(T)
    sf = num / den
    return isfinite(sf) ? sf : one(T)
end

@kernel function _amg_extract_diagonal_kernel!(diagonal, inv_diagonal, nzval, diagonal_index)
    i = @index(Global)
    T = eltype(diagonal)
    @inbounds begin
        idx = diagonal_index[i]
        aii = idx == 0 ? one(T) : nzval[idx]
        diagonal[i] = aii
        inv_diagonal[i] = abs(aii) > eps(T) ? inv(aii) : one(T)
    end
end

@kernel function _amg_powiter_seed_kernel!(v, invn)
    i = @index(Global)
    T = eltype(v)
    @inbounds v[i] = (isodd(i) ? one(T) : -one(T)) * invn
end

@kernel function _amg_powiter_scale_kernel!(w, invdiag)
    i = @index(Global)
    @inbounds w[i] *= invdiag[i]
end

@kernel function _amg_powiter_normalize_kernel!(v, w, lambda)
    i = @index(Global)
    @inbounds v[i] = w[i] / lambda
end

@kernel function _amg_gershgorin_kernel!(bound, rowptr, colval, nzval, invdiag)
    i = @index(Global)
    T = eltype(bound)
    acc = zero(T)
    @inbounds for p in rowptr[i]:(rowptr[i + 1] - 1)
        acc += abs(nzval[p])
    end
    @inbounds bound[i] = acc * abs(invdiag[i])
end

@kernel function _amg_chebyshev_first_step_kernel!(x, direction, residual, invdiag, alpha)
    i = @index(Global)
    @inbounds begin
        direction[i] = invdiag[i] * residual[i]
        x[i] += alpha * direction[i]
    end
end

@kernel function _amg_chebyshev_step_kernel!(x, direction, residual, invdiag, alpha, beta)
    i = @index(Global)
    @inbounds begin
        direction[i] = invdiag[i] * residual[i] + beta * direction[i]
        x[i] += alpha * direction[i]
    end
end

function _fill_amg!(hierarchy::AbstractAMGHierarchy, x, value)
    _launch_amg_kernel!(hierarchy, _amg_fill_kernel!, length(x), x, value)
    return x
end

function _copy_amg!(hierarchy::AbstractAMGHierarchy, dest, src)
    _launch_amg_kernel!(hierarchy, _amg_copy_kernel!, length(dest), dest, src)
    return dest
end

function _residual!(hierarchy::AMGHierarchy, r, A::AMGMatrixCSR, x, b)
    _launch_amg_kernel!(hierarchy, _amg_csr_residual_kernel!, _m(A), r, _rowptr(A), _colval(A), _nzval(A), x, b)
    return r
end

function _level_jacobi_omega(smoother::Jacobi, level::AMGLevel)
    T = eltype(level.x)
    lambda_max = max(T(level.lambda_max), one(T))
    # weight is lambda_max-scaled; clamp below 2/lambda_max for SPD stability
    return min(T(smoother.weight), T(2) - eps(T)) / lambda_max
end

function _apply_level_smoother!(hierarchy::AMGHierarchy, smoother, level::AMGLevel, b, loops)
    loops > 0 || return level.x
    _check_smoother_backend(hierarchy.backend, smoother)
    return _apply_level_smoother_impl!(hierarchy, smoother, level, b, loops)
end

_apply_level_smoother_impl!(hierarchy::AMGHierarchy, smoother::Jacobi, level::AMGLevel, b, loops) =
    _amg_jacobi!(hierarchy, smoother, level, level.A, b, loops)

function _amg_jacobi!(hierarchy::AMGHierarchy, smoother::Jacobi, level::AMGLevel, A::AMGMatrixCSR, b, loops)
    omega = _level_jacobi_omega(smoother, level)
    for _ in 1:loops
        _launch_amg_kernel!(
            hierarchy,
            _amg_jacobi_step_kernel!,
            length(level.x),
            level.tmp,
            level.x,
            b,
            _rowptr(A),
            _colval(A),
            _nzval(A),
            level.inv_diagonal,
            omega
        )
        level.x, level.tmp = level.tmp, level.x
    end
    return level.x
end

function _chebyshev_bounds(smoother::Chebyshev, level::AMGLevel)
    T = eltype(level.x)
    lambda_max = max(T(smoother.lambda_scale) * T(level.lambda_max), one(T))
    lambda_min = lambda_max / T(smoother.eig_ratio)
    center = (lambda_max + lambda_min) / T(2)
    radius = (lambda_max - lambda_min) / T(2)
    return center, radius
end

function _apply_level_smoother_impl!(hierarchy::AMGHierarchy, smoother::Chebyshev, level::AMGLevel, b, loops)
    T = eltype(level.x)
    for _ in 1:loops
        center, radius = _chebyshev_bounds(smoother, level)
        alpha = inv(center)
        _residual!(hierarchy, level.tmp, level.A, level.x, b)
        _launch_amg_kernel!(
            hierarchy,
            _amg_chebyshev_first_step_kernel!,
            length(level.x),
            level.x,
            level.direction,
            level.tmp,
            level.inv_diagonal,
            alpha
        )
        for k in 2:smoother.degree
            beta = (T(0.5) * radius * alpha)^2
            alpha = inv(center - beta / alpha)
            _residual!(hierarchy, level.tmp, level.A, level.x, b)
            _launch_amg_kernel!(
                hierarchy,
                _amg_chebyshev_step_kernel!,
                length(level.x),
                level.x,
                level.direction,
                level.tmp,
                level.inv_diagonal,
                alpha,
                beta
            )
        end
    end
    return level.x
end

function _amg_forward_sweep!(x, A::AMGMatrixCSR, b, diagonal, omega)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    T = eltype(x)
    @inbounds for i in 1:_m(A)
        sigma = zero(T)
        aii = diagonal[i]
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            j == i && continue
            sigma += nzval[p] * x[j]
        end
        if !iszero(aii)
            gs_value = (b[i] - sigma) / aii
            x[i] = (one(T) - omega) * x[i] + omega * gs_value
        end
    end
    return x
end

function _amg_backward_sweep!(x, A::AMGMatrixCSR, b, diagonal, omega)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    T = eltype(x)
    @inbounds for i in _m(A):-1:1
        sigma = zero(T)
        aii = diagonal[i]
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[p]
            j == i && continue
            sigma += nzval[p] * x[j]
        end
        if !iszero(aii)
            gs_value = (b[i] - sigma) / aii
            x[i] = (one(T) - omega) * x[i] + omega * gs_value
        end
    end
    return x
end

_apply_sweep!(::_Forward, x, A::AMGMatrixCSR, b, diagonal, omega) = _amg_forward_sweep!(x, A, b, diagonal, omega)

function _apply_sweep!(::_Symmetric, x, A::AMGMatrixCSR, b, diagonal, omega)
    _amg_forward_sweep!(x, A, b, diagonal, omega)
    _amg_backward_sweep!(x, A, b, diagonal, omega)
    return x
end

function _apply_level_smoother_impl!(hierarchy::AMGHierarchy, smoother::SweepSmoother, level::AMGLevel, b, loops)
    order, omega = _sweep_order(smoother), eltype(level.x)(_sweep_omega(smoother))
    for _ in 1:loops
        _apply_sweep!(order, level.x, level.A, b, level.diagonal, omega)
    end
    return level.x
end
