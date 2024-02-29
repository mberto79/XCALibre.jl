export set_preconditioner
export update_preconditioner!

set_preconditioner(PT::T, eqn, BCs, runtime
) where T<:PreconditionerType = 
begin
    mesh = get_phi(eqn).mesh
    discretise!(
        eqn, ConstantScalar(zero(_get_int(mesh))), runtime)
    apply_boundary_conditions!(eqn, BCs)
    P = Preconditioner{T}(eqn.equation.A)
    update_preconditioner!(P, mesh)
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

function update_preconditioner!(P::Preconditioner{NormDiagonal,M,PT,S}, mesh) where {M<:AbstractSparseArray,PT,S}
    backend = _get_backend(mesh)
    
    A = P.A
    # (; colptr, m, n, nzval, rowval) = A
    rowval, colptr, nzval, m ,n = sparse_array_deconstructor_preconditioners(A)
    storage = P.storage

    kernel! = update_NormDiagonal!(backend)
    kernel!(colptr, nzval, storage, ndrange = m)
end

@kernel function update_NormDiagonal!(colptr, nzval, storage)
    i = @index(Global)

    @inbounds begin
        idx_start = colptr[i]
        idx_next = colptr[i+1]
        column_vals = @view nzval[idx_start:(idx_next-1)] 
        storage[i] = 1/norm(column_vals)
    end
end

update_preconditioner!(
    P::Preconditioner{Jacobi,M,PT,S}
    ) where {M<:AbstractSparseArray,PT,S} =
begin
    A = P.A
    # (; colptr, m, n, nzval, rowval) = A
    rowval, colptr, nzval, m ,n = sparse_array_deconstructor_preconditioners(A)
    storage = P.storage
    idx_diagonal = zero(eltype(m)) # index to diagonal element
    @inbounds for i ∈ 1:m
        idx_start = colptr[i]
        idx_next = colptr[i+1]
        @inbounds for p ∈ idx_start:(idx_next-1)
            row = rowval[p]
            if row == i
                idx_diagonal = p
                break
            end
        end
        storage[i] = 1/abs(nzval[idx_diagonal])
    end
end

update_preconditioner!(
    P::Preconditioner{LDL,M,PT,S}
    ) where {M<:AbstractSparseArray,PT,S} =
begin
    nothing
end

update_preconditioner!(
    P::Preconditioner{ILU0,M,PT,S}
    ) where {M<:AbstractSparseArray,PT,S} =
begin
    ilu0!(P.storage, P.A)
    nothing
end

update_preconditioner!(
    P::Preconditioner{DILU,M,PT,S}
    ) where {M<:AbstractSparseArray,PT,S} =
begin
    update_dilu_diagonal!(P) 
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

function norm_static(arr, p = 2)
    sum = 0
    for i in eachindex(arr)
        val = (abs(arr[i]))^p
        sum += val
    end
    return sum^(1/p)
end