function _matvec!(y, A, x)
    mul!(y, A, x)
    return y
end

function _matvec!(y, A::SparseXCSR, x)
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    T = eltype(y)
    for i in 1:_m(A)
        yi = zero(T)
        for p in rowptr[i]:(rowptr[i + 1] - 1)
            yi += nzval[p] * x[colval[p]]
        end
        y[i] = yi
    end
    return y
end

function _residual!(r, A, x, b)
    _matvec!(r, A, x)
    @inbounds for i in eachindex(r)
        r[i] = b[i] - r[i]
    end
    return r
end

function _level_jacobi_omega(smoother::AMGJacobi, level::AMGLevel)
    T = eltype(level.x)
    lambda_max = max(T(level.lambda_max), one(T))
    return min(T(smoother.omega), T(4) / (T(3) * lambda_max))
end

function _apply_level_smoother!(smoother::AMGJacobi, level::AMGLevel, b, loops)
    A = level.A
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    T = eltype(level.x)
    omega = _level_jacobi_omega(smoother, level)
    for _ in 1:loops
        for i in eachindex(level.x)
            sigma = zero(T)
            for p in rowptr[i]:(rowptr[i + 1] - 1)
                j = colval[p]
                j == i && continue
                sigma += nzval[p] * level.x[j]
            end
            level.tmp[i] = (one(T) - omega) * level.x[i] + omega * level.inv_diagonal[i] * (b[i] - sigma)
        end
        copyto!(level.x, level.tmp)
    end
    return level.x
end
