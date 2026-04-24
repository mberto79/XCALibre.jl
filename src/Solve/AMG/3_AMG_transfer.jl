@kernel function _amg_csr_matvec_kernel!(y, rowptr, colval, nzval, x)
    i = @index(Global)
    acc = zero(eltype(y))
    @inbounds for p in rowptr[i]:(rowptr[i + 1] - 1)
        acc += nzval[p] * x[colval[p]]
    end
    y[i] = acc
end

@kernel function _amg_add_kernel!(x, y)
    i = @index(Global)
    @inbounds x[i] += y[i]
end

function _launch_amg_kernel!(hierarchy::AMGHierarchy, kernel, ndrange, args...)
    ndrange <= 0 && return nothing
    kernel! = kernel(_setup(hierarchy.backend, hierarchy.workgroup, ndrange)...)
    kernel!(args...)
    KernelAbstractions.synchronize(hierarchy.backend)
    return nothing
end

function _matvec!(hierarchy::AMGHierarchy, y, A::AMGMatrixCSR, x)
    _launch_amg_kernel!(hierarchy, _amg_csr_matvec_kernel!, _m(A), y, _rowptr(A), _colval(A), _nzval(A), x)
    return y
end

function _restrict!(hierarchy::AMGHierarchy, coarse_rhs, R, residual)
    _matvec!(hierarchy, coarse_rhs, R, residual)
    return coarse_rhs
end

function _prolongate_add!(hierarchy::AMGHierarchy, x, P, coarse_x, tmp)
    _matvec!(hierarchy, tmp, P, coarse_x)
    _launch_amg_kernel!(hierarchy, _amg_add_kernel!, length(x), x, tmp)
    return x
end
