export set_preconditioner
export update_preconditioner!

set_preconditioner(PT::T, eqn, BCs, config
) where T<:PreconditionerType = 
begin
    mesh = get_phi(eqn).mesh
    discretise!(
        eqn, ConstantScalar(zero(_get_int(mesh))), config)
    apply_boundary_conditions!(eqn, BCs, config)
    P = Preconditioner{T}(eqn.equation.A)
    update_preconditioner!(P, mesh, config)
    return P
end

# update_preconditioner!(
#     P::Preconditioner{NormDiagonal,M,PT,S}
#     ) where {M<:AbstractSparseArray,PT,S} =
# begin
#     A = P.A
#     # (; colptr, m, n, nzval, rowval) = A
#     rowval, colptr, nzval, m ,n = sparse_array_deconstructor_preconditioners(A)
#     storage = P.storage
#     @inbounds for i ∈ 1:m
#         idx_start = colptr[i]
#         idx_next = colptr[i+1]
#         column_vals = @view nzval[idx_start:(idx_next-1)] 
#         storage[i] = 1/norm(column_vals)    
#     end
# end

function update_preconditioner!(P::Preconditioner{NormDiagonal,M,PT,S}, mesh, config) where {M<:AbstractSparseArray,PT,S}
    # backend = _get_backend(mesh)

    (; hardware) = config
    (; backend, workgroup) = hardware
    
    A = P.A
    # (; colptr, m, n, nzval, rowval) = A
    # rowval, colptr, nzval, m ,n = sparse_array_deconstructor_preconditioners(A)
    nzval_array = _nzval(A)
    colptr_array = _colptr(A)
    m_array = _m(A)

    storage = P.storage

    kernel! = update_NormDiagonal!(backend, workgroup)
    kernel!(colptr_array, nzval_array, storage, ndrange = m_array)
    KernelAbstractions.synchronize(backend)
end

@kernel function update_NormDiagonal!(colptr, nzval, storage)
    i = @index(Global)

    # @inbounds begin
        idx_start = colptr[i]
        idx_next = colptr[i+1]
        column_vals = @view nzval[idx_start:(idx_next-1)] 
        norm = norm_static(column_vals)
        storage[i] = 1/norm
    # end
end

# update_preconditioner!(P::Preconditioner{Jacobi,M,PT,S}, mesh) where {M<:AbstractSparseArray,PT,S} =
# begin
#     A = P.A
#     # (; colptr, m, n, nzval, rowval) = A
#     rowval, colptr, nzval, m, n = sparse_array_deconstructor_preconditioners(A)
#     storage = P.storage
#     idx_diagonal = zero(eltype(m)) # index to diagonal element
#     @inbounds for i ∈ 1:m
#         idx_start = colptr[i]
#         idx_next = colptr[i+1]
#         @inbounds for p ∈ idx_start:(idx_next-1)
#             row = rowval[p]
#             if row == i
#                 idx_diagonal = p
#                 break
#             end
#         end
#         storage[i] = 1/abs(nzval[idx_diagonal])
#     end
# end

function update_preconditioner!(P::Preconditioner{Jacobi,M,PT,S}, mesh, config) where {M<:AbstractSparseArray,PT,S}
    # backend = _get_backend(mesh)

    (; hardware) = config
    (; backend, workgroup) = hardware

    A = P.A
    # (; colptr, m, n, nzval, rowval) = A
    # rowval, colptr, nzval, m, n = sparse_array_deconstructor_preconditioners(A)
    rowval_array = _rowval(A)
    colptr_array = _colptr(A)
    nzval_array = _nzval(A)
    m_array = _m(A)

    storage = P.storage
    idx_diagonal = zero(eltype(m_array)) # index to diagonal element

    kernel! = update_Jacobi!(backend, workgroup)
    kernel!(rowval_array, colptr_array, nzval_array, idx_diagonal, storage, ndrange = m_array)
    KernelAbstractions.synchronize(backend)
end

@kernel function update_Jacobi!(rowval, colptr, nzval, idx_diagonal, storage)
    i = @index(Global)

    @inbounds begin
        # idx_start = colptr[i]
        # idx_next = colptr[i+1]
        # @inbounds for p ∈ idx_start:(idx_next-1)
        #     row = rowval[p]
        #     if row == i
        #         idx_diagonal = p
        #         break
        #     end
        # end
        idx_diagonal = spindex(colptr, rowval, i, i)
        storage[i] = 1/abs(nzval[idx_diagonal])
    end
end

update_preconditioner!(P::Preconditioner{LDL,M,PT,S}, mesh) where {M<:AbstractSparseArray,PT,S} =
begin
    nothing
end

update_preconditioner!(P::Preconditioner{ILU0,M,PT,S}, mesh) where {M<:AbstractSparseArray,PT,S} =
begin
    ilu0!(P.storage, P.A)
    nothing
end

update_preconditioner!(P::Preconditioner{DILU,M,PT,S}, mesh) where {M<:AbstractSparseArray,PT,S} =
begin
    update_dilu_diagonal!(P, mesh)
    nothing
end

function sparse_array_deconstructor_preconditioners(arr::SparseArrays.SparseMatrixCSC)
    (; rowval, colptr, nzval, m, n) = arr
    return rowval, colptr, nzval, m ,n
end

function sparse_array_deconstructor_preconditioners(arr::CUDA.CUSPARSE.CuSparseMatrixCSC)
    (; rowVal, colPtr, nzVal, dims) = arr
    return rowVal, colPtr, nzVal, dims[1], dims[2]
end

_m(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.dims[1]
_m(A::SparseArrays.SparseMatrixCSC) = A.m

_n(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.dims[2]
_n(A::SparseArrays.SparseMatrixCSC) = A.n