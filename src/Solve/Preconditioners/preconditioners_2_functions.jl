export set_preconditioner
export update_preconditioner!

set_preconditioner(PT::None, eqn, BCs, runtime
) = 
begin
    None()
end

set_preconditioner(PT::T, eqn, BCs, runtime
) where T<:PreconditionerType = 
begin
    discretise!(
        eqn, ConstantScalar(zero(_get_int(get_phi(eqn).mesh))), runtime)
    apply_boundary_conditions!(eqn, BCs)
    P = Preconditioner{T}(eqn.equation.A)
    update_preconditioner!(P)
    return P
end

update_preconditioner!(
    P::None
    ) =
begin
    nothing
end

update_preconditioner!(
    P::Preconditioner{NormDiagonal,M,PT,S}
    ) where {M<:SparseMatrixCSC,PT,S} =
begin
    A = P.A
    (; colptr, m, n, nzval, rowval) = A
    storage = P.storage
    @inbounds for i ∈ 1:m
        idx_start = colptr[i]
        idx_next = colptr[i+1]
        column_vals = @view nzval[idx_start:(idx_next-1)] 
        storage[i] = 1/norm(column_vals)    
    end
end

update_preconditioner!(
    P::Preconditioner{Jacobi,M,PT,S}
    ) where {M<:SparseMatrixCSC,PT,S} =
begin
    A = P.A
    (; colptr, m, n, nzval, rowval) = A
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
    ) where {M<:SparseMatrixCSC,PT,S} =
begin
    nothing
end

update_preconditioner!(
    P::Preconditioner{ILU0,M,PT,S}
    ) where {M<:SparseMatrixCSC,PT,S} =
begin
    ilu0!(P.storage, P.A)
    nothing
end

update_preconditioner!(
    P::Preconditioner{DILU,M,PT,S}
    ) where {M<:SparseMatrixCSC,PT,S} =
begin
    update_dilu_diagonal!(P) 
    nothing
end