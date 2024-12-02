import LinearAlgebra.ldiv!, LinearAlgebra.\

export ldiv!

function extract_diagonal!(D, Di, A::AbstractSparseArray, config)
    (; hardware) = config
    # (; backend, workgroup) = hardware
    (; nzval, n) = A
    backend = CPU()
    workgroup = cld(n, Threads.nthreads())
    

    kernel! = _extract_diagonal!(backend, workgroup)
    kernel!(D, Di, nzval, ndrange = n)
end

@kernel function _extract_diagonal!(D, Di, nzval)
    i = @index(Global)
    
    @inbounds begin
        D[i] = nzval[Di[i]]
    end
end

function diagonal_indices!(Di, A::SparseMatrixCSC{Tf,Ti}) where {Tf,Ti} 
    (; colptr, n, rowval) = A
    idx_diagonal = zero(Ti) # index to diagonal element
    @inbounds for i ∈ 1:n
        idx_diagonal = spindex_csc(colptr, rowval, i, i)
        Di[i] = idx_diagonal
    end
end

function diagonal_indices!(Di, A::SparseMatrixCSR{N, Tf,Ti}) where {N, Tf,Ti} 
    (; rowptr, n, colval) = A
    idx_diagonal = zero(Ti) # index to diagonal element
    @inbounds for i ∈ 1:n
        idx_diagonal = spindex(rowptr, colval, i, i)
        Di[i] = idx_diagonal
    end
end

function update_dilu_diagonal!(P, mesh, config) # must rename
    # for i ∈ 1:m 
    #     D[i] = A[i,i]
    # end

    # for i ∈ 1:m
    #     for j ∈ (i+1):m
    #         D[j] = D[j] - A[i,j]*A[j,i]/D[i]
    #     end
    # end

    # (; A, storage) = P 
    (; storage) = P 
    A = storage.A
    (; rowptr, colval, nzval, m, n) = storage.A
    (; D, Di) = storage
    extract_diagonal!(D, Di, A, config)
    @inbounds for i ∈ 1:m
        start_index = rowptr[i]
        end_index = rowptr[i+1] - 1
        @inbounds for nzi ∈ start_index:end_index # j ∈ (i+1):m
            j = colval[nzi]
            if j > i
                nzIndex1 = nzi # spindex(rowptr, colval, i, j)
                nzIndex2 = spindex(rowptr, colval, j, i)
                if nzIndex2 !== 0
                    D[j] = D[j] - nzval[nzIndex1]*nzval[nzIndex2]/D[i]
                end
            end
        end
    end
    nothing
end

#### PRECONDITIONER DECOMPOSITION ####
# L = UnitLowerTriangular(Acsc) - I
# U = UnitUpperTriangular(Acsc) - I
# PL = (D_star + L)*D_star_inv
# PU = (D_star + U)
######################################

function forward_substitution!(y, P::DILUprecon{M,V,VI}, b) where {M,V,VI}
    (; A, D, Di) = P
    (; rowptr, colval, nzval, n, m) = A
    for i ∈ 1:n
        start_index = rowptr[i]
        end_index = Di[i] - 1
        sum = 0.0
        for nzi ∈ start_index:end_index
            j = colval[nzi]
            A_ij = nzval[nzi]
            sum += A_ij*y[j]*(1/D[j]) # Aii of P is equal to Di, thus (1/Dj) makes Aii = 1
        end
        y[i] = (b[i] - sum)
    end
end

function backward_substitution!(x, P::DILUprecon{M,V,VI}, y) where {M,V,VI}
    (; A, D, Di) = P
    (; rowptr, colval, nzval, n, m) = A
    for i ∈ n:-1:1
        start_index = Di[i] + 1
        end_index = rowptr[i+1] - 1
        sum = 0.0
        for nzi ∈ start_index:end_index
            j = colval[nzi]
                A_ij = ifelse(i == j, D[i], nzval[nzi])
                sum += A_ij*x[j]
        end
        x[i] = (y[i] - sum)/D[i]
    end
end

ldiv!(x, P::DILUprecon{M,V,VI}, b) where {M<:AbstractSparseArray,V,VI} =
begin
    forward_substitution!(x, P, b)
    backward_substitution!(x, P, x)
end