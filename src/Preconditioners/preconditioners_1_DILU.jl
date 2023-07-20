import LinearAlgebra.ldiv!, LinearAlgebra.\

export extract_diagonal!
export dilu_diagonal2!, sparse_diagonal_indices!
export forward_substitution, backward_substitution
export ldiv!, left_div!



function extract_diagonal!(D, Di, A) 
# function extract_diagonal!(D, A) 
# m, n = size(A)
# for i ∈ 1:m
#     D[i] = A[i,i]
# end


(; colptr, m, n, nzval, rowval) = A
idx_diagonal = zero(eltype(m)) # index to diagonal element
@inbounds for i ∈ 1:m
    D[i] = nzval[Di[i]]
end

end

function sparse_diagonal_indices!(Di, A) 
    (; colptr, m, n, nzval, rowval) = A
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
        Di[i] = idx_diagonal
    end
end

function dilu_diagonal2!(P) # must rename
    # (; A, storage) = P
    # (; colptr, m, n, nzval, rowval) = A
    # (; Di, D) = storage
    # extract_diagonal!(D, Di, A) 
    # for i ∈ 1:m
    #     # for j ∈ 1:(i-1)
    #     for j ∈ (i+1):m
    #         # D[i] -= A[i,j]*A[j,i]/D[j]
    #         D[j] -= A[i,j]*A[j,i]/D[i]
    #     end
    # end

    # Algo 2 No good!!!
    # (; A, storage) = P
    # (; colptr, m, n, nzval, rowval) = A
    # (; Di, D) = storage
    # extract_diagonal!(D, Di, A)
    # T = A*transpose(A)*(1.0./D)
    # D .-= T
    
    # Algo 3
    (; A, storage) = P
    (; colptr, m, n, nzval, rowval) = A
    (; Di, D) = storage
    
    extract_diagonal!(D, Di, A)
    @inbounds for i ∈ 1:n # add  D[j] -= A[i,j]*A[j,i]/D[i] 
        for j = (i+1):n
            pj_start = colptr[i] 
            pj_end = colptr[i+1] - 1
            for p_j ∈ pj_start:pj_end
                rowi = rowval[p_j]
                # if rowi > i 
                #     break
                # end
                active_row = j
                p_ij = p_j
                pi_start = colptr[active_row] 
                pi_end = colptr[active_row+1] - 1 
                for p_i ∈ pi_start:pi_end
                    rowj = rowval[p_i]
                    if rowj == i 
                        p_ji = p_i
                        D[Di[j]] -= A[p_ij]*A[p_ji]/D[Di[i]]
                        break
                    end
                end
            end

        end
    end
    nothing
end

function forward_substitution(P, b)
    (; A, storage) = P
    (; colptr, m, n, nzval, rowval) = A
    (; Di, D) = storage

    x = zeros(eltype(D), m)
    x .= b
    for j ∈ 1:n-1
        c_start = Di[j] + 1
        c_end = colptr[j+1] - 1
        for c_pointer ∈ c_start:c_end
            i = rowval[c_pointer]
            x[i] -= nzval[c_pointer] * x[j] / D[j]
        end
    end
    return x

end

function backward_substitution(P, b)
    (; A, storage) = P
    (; colptr, m, n, nzval, rowval) = A
    (; Di, D) = storage

    # # Algo 1
    # x = zeros(eltype(D), n)
    # x[n] = b[n]/D[n]
    # sum = zero(eltype(A))
    # for i ∈ n-1:-1:1
    #     for j=i+1:n
    #     sum += A[i,j]*x[j]
    #     end
    #     x[i] = (b[i] - sum)/D[i]
    #     sum = zero(eltype(A))
    # end
    # return x

    # Algo 2
    x = zeros(eltype(D), m)
    x .= b./D
    # x .= b.*D
    for j ∈ n-1:-1:1
        c_start = Di[j] + 1
        c_end = colptr[j+1] - 1
        for c_pointer ∈ c_start:c_end
            i = rowval[c_pointer]
            x[j] -= nzval[c_pointer] * x[i]/D[j]
            # x[j] -= nzval[c_pointer] * x[i]*D[j]
        end
    end
    return x
end

function left_div!(x, P, b)
# function left_div!(x, A, D, b)

    (; A, Di, D) = P
    (; colptr, m, n, nzval, rowval) = A

    # Forward substitution

    # # Algo 1
    # x .= b
    # for i ∈ 1:m
    #     for j ∈ 1:(i-1) # needs serious check!
    #         x[i] -= A[i,j]*x[j] # Ci = (1/Di)(bi - Aij*Cj)
    #     end
    #     x[i] /= D[i]
    # end

    # Algo 2
    @inbounds for i ∈ eachindex(x)
        # x[i] = b[i]/D[i]
        # x[i] = b[i]*D[i]
        x[i] = b[i]
    end
    @inbounds for j ∈ 1:n-1
        c_start = Di[j] + 1
        c_end = colptr[j+1] - 1
        for c_pointer ∈ c_start:c_end
            i = rowval[c_pointer]
            # x[i] -= nzval[c_pointer] * x[j]
            x[i] -= nzval[c_pointer] * x[j]/D[j]
            # x[i] -= nzval[c_pointer] * x[j]*D[i]
        end
    end

    # Backward substitution
    # Algo 1
    # x[n] = x[n]/D[n]
    # x[n] = x[n]*D[n]
    # sum = zero(eltype(A))
    # for i ∈ n-1:-1:1
    #     for j=i+1:n
    #     sum += A[i,j]*x[j]
    #     end
    #     # x[i] = (x[i] - sum)/D[i]
    #     x[i] = (x[i] - sum)*D[i]
    #     sum = zero(eltype(A))
    # end

    # Algo 2
    @inbounds for i ∈ eachindex(x)
        x[i] = x[i]/D[i]
        # x[i] = b[i]*D[i]
        # x[i] = x[i]
    end
    @inbounds for j ∈ n-1:-1:1
        c_start = Di[j] + 1
        c_end = colptr[j+1] - 1
        for c_pointer ∈ c_start:c_end
            i = rowval[c_pointer]
            # x[j] -= nzval[c_pointer] * x[i]/D[j]
            x[j] -= nzval[c_pointer] * x[i]/D[j]
        end
    end
    nothing
end

function ldiv!(y, P::DILUprecon{M,V,I}, b
    ) where {M<:SparseMatrixCSC,V,I}
    ###
    left_div!(y, P, b)
    ###   
    nothing
end
