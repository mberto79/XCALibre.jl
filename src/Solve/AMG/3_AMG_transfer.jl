function _restrict!(coarse_rhs, R, residual)
    mul!(coarse_rhs, R, residual)
    return coarse_rhs
end

function _prolongate_add!(x, P, coarse_x, tmp)
    mul!(tmp, P, coarse_x)
    @inbounds for i in eachindex(x)
        x[i] += tmp[i]
    end
    return x
end
