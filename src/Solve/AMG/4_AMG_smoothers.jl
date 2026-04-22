function _residual!(r, A, x, b)
    mul!(r, A, x)
    @inbounds for i in eachindex(r)
        r[i] = b[i] - r[i]
    end
    return r
end

function _apply_level_smoother!(smoother::AMGJacobi, level::AMGLevel, b, loops)
    A = level.A
    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    T = eltype(level.x)
    ω = T(smoother.omega)
    for _ in 1:loops
        for i in eachindex(level.x)
            sigma = zero(T)
            for p in rowptr[i]:(rowptr[i + 1] - 1)
                j = colval[p]
                j == i && continue
                sigma += nzval[p] * level.x[j]
            end
            level.tmp[i] = (one(T) - ω) * level.x[i] + ω * level.inv_diagonal[i] * (b[i] - sigma)
        end
        copyto!(level.x, level.tmp)
    end
    return level.x
end

function _apply_level_smoother!(smoother::AMGChebyshev, level::AMGLevel, b, loops)
    T = eltype(level.x)
    d = max(level.lambda_max * (one(T) + smoother.lower_fraction) / 2, eps(T))
    c = max(level.lambda_max * (one(T) - smoother.lower_fraction) / 2, zero(T))
    p = level.tmp
    fill!(p, zero(T))
    α = inv(d)
    β = zero(T)
    for _ in 1:max(loops, smoother.degree)
        _residual!(level.rhs, level.A, level.x, b)
        for i in eachindex(level.x)
            z = level.inv_diagonal[i] * level.rhs[i]
            p[i] = α * z + β * p[i]
            level.x[i] += p[i]
        end
        β = (c * α / 2) ^ 2
        α = inv(d - β / α)
    end
    return level.x
end
