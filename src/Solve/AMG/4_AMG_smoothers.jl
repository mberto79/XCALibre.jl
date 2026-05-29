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

@kernel function _amg_jacobi_step_kernel!(x_new, x_old, b, rowptr, colval, nzval, invdiag, omega)
    i = @index(Global)
    T = eltype(x_new)
    sigma = zero(T)
    @inbounds for p in rowptr[i]:(rowptr[i + 1] - 1)
        j = colval[p]
        j == i && continue
        sigma += nzval[p] * x_old[j]
    end
    @inbounds x_new[i] = (one(T) - omega) * x_old[i] + omega * invdiag[i] * (b[i] - sigma)
end

@kernel function _amg_weighted_diagonal_correction_kernel!(x, residual, invdiag, omega)
    i = @index(Global)
    @inbounds x[i] += omega * invdiag[i] * residual[i]
end

function _fill_amg!(hierarchy::AMGHierarchy, x, value)
    _launch_amg_kernel!(hierarchy, _amg_fill_kernel!, length(x), x, value)
    return x
end

function _copy_amg!(hierarchy::AMGHierarchy, dest, src)
    _launch_amg_kernel!(hierarchy, _amg_copy_kernel!, length(dest), dest, src)
    return dest
end

function _residual!(hierarchy::AMGHierarchy, r, A::AMGMatrixCSR, x, b)
    _matvec!(hierarchy, r, A, x)
    _launch_amg_kernel!(hierarchy, _amg_residual_kernel!, length(r), r, r, b)
    return r
end

function _level_jacobi_omega(smoother::AMGJacobi, level::AMGLevel)
    T = eltype(level.x)
    lambda_max = max(T(level.lambda_max), one(T))
    return min(T(smoother.omega), T(4) / (T(3) * lambda_max))
end

function _apply_level_smoother!(hierarchy::AMGHierarchy, smoother::AMGJacobi, level::AMGLevel, b, loops)
    omega = _level_jacobi_omega(smoother, level)
    for _ in 1:loops
        _launch_amg_kernel!(
            hierarchy,
            _amg_jacobi_step_kernel!,
            length(level.x),
            level.tmp,
            level.x,
            b,
            _rowptr(level.A),
            _colval(level.A),
            _nzval(level.A),
            level.inv_diagonal,
            omega
        )
        level.x, level.tmp = level.tmp, level.x
    end
    return level.x
end

function _chebyshev_weight(smoother::AMGChebyshev, level::AMGLevel, degree_index)
    T = eltype(level.x)
    lambda_max = max(T(smoother.lambda_scale) * T(level.lambda_max), one(T))
    lambda_min = lambda_max / T(smoother.eig_ratio)
    center = (lambda_max + lambda_min) / T(2)
    radius = (lambda_max - lambda_min) / T(2)
    angle = (T(2 * degree_index - 1) * T(pi)) / T(2 * smoother.degree)
    return inv(center - radius * cos(angle))
end

function _apply_level_smoother!(hierarchy::AMGHierarchy, smoother::AMGChebyshev, level::AMGLevel, b, loops)
    for _ in 1:loops
        for k in 1:smoother.degree
            _residual!(hierarchy, level.tmp, level.A, level.x, b)
            omega = _chebyshev_weight(smoother, level, k)
            _launch_amg_kernel!(
                hierarchy,
                _amg_weighted_diagonal_correction_kernel!,
                length(level.x),
                level.x,
                level.tmp,
                level.inv_diagonal,
                omega
            )
        end
    end
    return level.x
end
