import LinearAlgebra.ldiv!, LinearAlgebra.\

export extract_diagonal!
export dilu_diagonal2!, sparse_diagonal_indices!, sparse_row_indices
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

integer_type(A::SparseMatrixCSC{Tf,Ti}) where {Tf,Ti} = Ti

function sparse_row_indices(A, Di) # upper triangular row indices
    (; colptr, m, n, nzval, rowval) = A
    idx_diagonal = zero(eltype(m)) 
    Ri = Vector{integer_type(A)}[] # pointers to sparse rows
    J = Vector{integer_type(A)}[] # pointers to sparse rows
    @inbounds for i ∈ 1:m
        temp = integer_type(A)[]
        J_temp = integer_type(A)[]
        for j = (i+1):m
            idx_start = colptr[j]
            idx_next = Di[j] #colptr[j+1] - 1
            @inbounds for p ∈ idx_start:idx_next
                row = rowval[p]
                if row == i
                    push!(temp, p) # array of pointers
                    push!(J_temp, j) # column indeces
                end
            end
        end
        push!(Ri, temp)
        push!(J, J_temp)
    end
    return Ri, J
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

    # Algo 2
    # (; A, storage) = P
    # (; colptr, m, n, nzval, rowval) = A
    # (; Di, D) = storage
    # extract_diagonal!(D, Di, A) 
    # sum = 0.0
    # for i ∈ 2:m
    #     for j ∈ 1:(i-1)
    #         sum += A[i,j]*A[j,i]/D[j]
    #     end
    #     D[i] -= sum
    #     sum = 0.0
    # end
    
    # Algo 3
    (; A, storage) = P
    (; colptr, m, n, nzval, rowval) = A
    (; Di, Ri, D) = storage
    
    extract_diagonal!(D, Di, A)

    @inbounds for i ∈ 1:n
        # D[i] = nzval[Di[i]] 
        c_start = Di[i] + 1
        c_end = colptr[i+1] - 1
        r_pointer = Ri[i]
        r_count = 0
        @inbounds for c_pointer ∈ c_start:c_end
            j = rowval[c_pointer]
            r_count += 1
            D[j] -= nzval[c_pointer]*nzval[r_pointer[r_count]]/D[i]
        end
        D[i] = 1.0/D[i] # store inverse
    end
    # D .= 1.0./D # store inverse
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

    (; A, D, Di, Ri, J) = P
    (; colptr, m, n, nzval, rowval) = A

    # Forward substitution

    # # Algo 1
    # x .= b
    # for i ∈ 2:m
    #     for j ∈ 1:(i-1) # needs serious check!
    #         x[i] -= A[i,j]*x[j]/D[j]
    #         x[i] -= A[i,j]*x[j]*D[j]
    #     end
    # end

    # Algo 2
    @inbounds for i ∈ eachindex(x)
        x[i] = b[i]
    end
    @inbounds for j ∈ 1:n
        c_start = Di[j] + 1
        c_end = colptr[j+1] - 1
        @inbounds for c_pointer ∈ c_start:c_end
            i = rowval[c_pointer]
            # x[i] -= nzval[c_pointer]*x[j]/D[j]
            x[i] -= nzval[c_pointer]*x[j]*D[j]
        end
    end

    # Backward substitution
    # Algo 1
    # x .= x./D
    # x .= x.*D
    # for i ∈ (n-1):-1:1
    #     for j ∈ (i+1):n # needs serious check!
    #         x[i] -= A[i,j]*x[j]/D[i]
    #         x[i] -= A[i,j]*x[j]*D[i]
    #     end
    # end

    # Algo 2
    @inbounds for i ∈ eachindex(x)
        # x[i] = x[i]/D[i]
        x[i] = x[i]*D[i]
    end
    for i ∈ (n-1):-1:1
        c_pointers = Ri[i]
        j = J[i]
        for (p_i, p) ∈ enumerate(c_pointers)
            # x[i] -= nzval[p]*x[j[p_i]]/D[i]
            x[i] -= nzval[p]*x[j[p_i]]*D[i]
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
