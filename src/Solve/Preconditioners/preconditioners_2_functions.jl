# export set_preconditioner
export update_preconditioner!

# set_preconditioner(PT::T, A) where T<:PreconditionerType = begin
#     Preconditioner{T}(A)
# end

# set_preconditioner(PT::T, eqn, BCs, config
# ) where T<:PreconditionerType = 
# begin
#     phi = get_phi(eqn)
#     mesh = phi.mesh
#     TF = _get_float(mesh)
#     time = zero(TF) # assumes simulation starts at time = 0 (might need generalising)

#     if typeof(phi) <: AbstractVectorField

#         discretise!(
#             eqn, get_phi(eqn)) # should this be float?

#         time = zero(TF)
#         apply_boundary_conditions!(eqn, BCs, XDir(1), time)

#     elseif typeof(phi) <: AbstractScalarField

#         discretise!(eqn, get_phi(eqn)) # should this be float?
#         apply_boundary_conditions!(eqn, BCs, nothing, time)
#     end

#     P = Preconditioner{T}(eqn.equation.A)
#     update_preconditioner!(P, mesh)
#     return P
# end

function update_preconditioner!(P::Preconditioner{NormDiagonal,M,PT,S}, mesh) where {M<:AbstractSparseArray,PT,S}
    # backend = _get_backend(mesh)

    (; hardware) = get_configuration(CONFIG)
    (; backend, workgroup) = hardware
    
    A = P.A
    nzval_array = _nzval(A)
    colptr_array = _rowptr(A)

    storage = P.storage

    ndrange = _m(A)
    kernel! = update_NormDiagonal!(_setup(backend, workgroup, ndrange)...)
    kernel!(colptr_array, nzval_array, storage)
    # KernelAbstractions.synchronize(backend)
end

@kernel function update_NormDiagonal!(rowptr, nzval, storage)
    i = @index(Global)

    # @inbounds begin
        idx_start = rowptr[i]
        idx_next = rowptr[i+1]
        column_vals = @view nzval[idx_start:(idx_next-1)] 
        norm = norm_static(column_vals)
        storage[i] = 1/norm
    # end
end

function update_preconditioner!(P::Preconditioner{Jacobi,M,PT,S}, mesh) where {M<:AbstractSparseArray,PT,S}
    # backend = _get_backend(mesh)

    (; hardware) = get_configuration(CONFIG)
    (; backend, workgroup) = hardware

    A = P.A
    rowval = _colval(A)
    colptr = _rowptr(A)
    nzval = _nzval(A)
    m = _m(A)

    storage = P.storage
    idx_diagonal = zero(eltype(m)) # index to diagonal element

    ndrange = m
    kernel! = update_Jacobi!(_setup(backend, workgroup, ndrange)...)
    kernel!(rowval, colptr, nzval, idx_diagonal, storage)
    # KernelAbstractions.synchronize(backend)
end

@kernel function update_Jacobi!(colval, rowptr, nzval, idx_diagonal, storage)
    i = @index(Global)

    @inbounds begin
        idx_diagonal = spindex(rowptr, colval, i, i)
        storage[i] = 1/abs(nzval[idx_diagonal])
    end
end

# update_preconditioner!(P::Preconditioner{LDL,M,PT,S},  mesh) where {M<:AbstractSparseArray,PT,S} =
# begin
#     nothing
# end


# update_preconditioner!(P::Preconditioner{ILU0,M,PT,S},  mesh) where {M<:AbstractSparseArray,PT,S} =
# begin
#     ilu0!(P.storage, P.A)
#     nothing
# end

update_preconditioner!(P::Preconditioner{DILU,M,PT,S},  mesh) where {M<:AbstractSparseArray,PT,S} =
begin
    update_dilu_diagonal!(P, mesh)
    nothing
end


function sparse_array_deconstructor_preconditioners(arr::SparseMatricesCSR.SparseMatrixCSR)
    (; colval, rowptr, nzval, m, n) = arr
    return colval, rowptr, nzval, m ,n
end

_m(A::SparseMatricesCSR.SparseMatrixCSR) = A.m
_n(A::SparseMatricesCSR.SparseMatrixCSR) = A.n

_m(A::SparseXCSR) = parent(A).m
_n(A::SparseXCSR) = parent(A).n