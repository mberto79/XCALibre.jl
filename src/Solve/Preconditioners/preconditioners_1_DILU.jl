import LinearAlgebra.ldiv!, LinearAlgebra.\

export ldiv!

# extract_diagonal!(D, Di, A::AbstractSparseArray{Tf,Ti}) where {Tf,Ti} =
# begin
#     rowval, colptr, nzval, m ,n = sparse_array_deconstructor_preconditioners(A)
#     @inbounds for i ∈ 1:n
#         D[i] = nzval[Di[i]]
#     end
# end

function extract_diagonal!(D, Di, A::AbstractSparseArray{Tf,Ti}, config) where {Tf,Ti}
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    rowval, colptr, nzval, m ,n = sparse_array_deconstructor_preconditioners(A)

    kernel! = extract_diagonal_kernel!(backend, workgroup)
    kernel!(D, Di, nzval, ndrange = n)
end

@kernel function extract_diagonal_kernel!(D, Di, nzval)
    i = @index(Global)
    
    @inbounds begin
        D[i] = nzval[Di[i]]
    end
end

function diagonal_indices!(Di, A::AbstractSparseArray{Tf,Ti}) where {Tf,Ti} 
    (; colptr, n, rowval) = A
    idx_diagonal = zero(eltype(n)) # index to diagonal element
    @inbounds for i ∈ 1:n
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
# integer_type(A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tf,Ti}) where {Tf,Ti} = Ti

function upper_row_indices(A, Di) # upper triangular row column indices
    (; colptr, n, rowval) = A
    Ri = integer_type(A)[] # column pointers on i-th row
    J = integer_type(A)[] # column indices on i-th row
    upper_indices_IDs = UnitRange{integer_type(A)}[]
    @inbounds for i ∈ 1:n
        R_temp = integer_type(A)[]
        J_temp = integer_type(A)[]
        upper_indices_start = length(Ri)
        offset = 0
        for j = (i+1):n
            idx_start = colptr[j]
            idx_next = Di[j] # access column down to diagonal only
            @inbounds for p ∈ idx_start:idx_next
                row = rowval[p]
                if row == i
                    push!(R_temp, p) # array of pointers
                    push!(J_temp, j) # column indices
                    offset += 1
                end
            end
        end
        upper_indices_end = upper_indices_start + offset
        push!(Ri, R_temp...)
        push!(J, J_temp...)
        if upper_indices_end == upper_indices_start
            IDs_range = UnitRange{integer_type(A)}(upper_indices_start:upper_indices_end)
        else
            IDs_range = UnitRange{integer_type(A)}(upper_indices_start+1:upper_indices_end)
        end
        push!(upper_indices_IDs, IDs_range)
    end
    return Ri, J, upper_indices_IDs
end

# function update_dilu_diagonal!(P, mesh) # must rename
#     # (; A, storage) = P
#     # (; colptr, m, n, nzval, rowval) = A
#     # (; Di, D) = storage
#     # extract_diagonal!(D, Di, A) 
#     # for i ∈ 1:m
#     #     # for j ∈ 1:(i-1)
#     #     for j ∈ (i+1):m
#     #         # D[i] -= A[i,j]*A[j,i]/D[j]
#     #         D[j] -= A[i,j]*A[j,i]/D[i]
#     #     end
#     # end

#     # Algo 2
#     # (; A, storage) = P
#     # (; colptr, m, n, nzval, rowval) = A
#     # (; Di, D) = storage
#     # extract_diagonal!(D, Di, A) 
#     # sum = 0.0
#     # for i ∈ 2:m
#     #     for j ∈ 1:(i-1)
#     #         sum += A[i,j]*A[j,i]/D[j]
#     #     end
#     #     D[i] -= sum
#     #     sum = 0.0
#     # end
    
#     # Algo 3
#     backend = _get_backend(mesh)

#     (; A, storage) = P
#     # (; colptr, n, nzval, rowval) = A
#     rowval, colptr, nzval, m ,n = sparse_array_deconstructor_preconditioners(A)
#     (; Di, Ri, D, upper_indices_IDs) = storage
    
#     extract_diagonal!(D, Di, A, backend)

#     @inbounds for i ∈ 1:n
#         # D[i] = nzval[Di[i]]
#         upper_index_ID = upper_indices_IDs[i] 
#         c_start = Di[i] + 1
#         c_end = colptr[i+1] - 1
#         r_pointer = Ri[upper_index_ID]
#         r_count = 0
#         @inbounds for c_pointer ∈ c_start:c_end
#             j = rowval[c_pointer]
#             r_count += 1
#             D[j] -= nzval[c_pointer]*nzval[r_pointer[r_count]]/D[i]
#         end
#         D[i] = 1/D[i] # store inverse
#     end
#     # D .= 1.0./D # store inverse
#     nothing
# end

function update_dilu_diagonal!(P, mesh, config) # must rename
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
    # backend = _get_backend(mesh)

    (; hardware) = config
    (; backend, workgroup) = hardware

    (; A, storage) = P
    # (; colptr, n, nzval, rowval) = A
    rowval, colptr, nzval, m ,n = sparse_array_deconstructor_preconditioners(A)
    (; Di, Ri, D, upper_indices_IDs) = storage
    
    extract_diagonal!(D, Di, A, config)

    kernel! = update_dilu_diagonal_kernel!(backend, workgroup)
    kernel!(upper_indices_IDs, Di, colptr, Ri, rowval, D, nzval, ndrange = n)
    # D .= 1.0./D # store inverse
    nothing
end


@kernel function update_dilu_diagonal_kernel!(upper_indices_IDs, Di, colptr, Ri, rowval, D, nzval)
    i = @index(Global)
    
    @inbounds begin
        # D[i] = nzval[Di[i]]
        upper_index_ID = upper_indices_IDs[i] 
        c_start = Di[i] + 1 
        c_end = colptr[i+1] - 1
        r_count = 0
        for c_pointer ∈ c_start:c_end
            j = rowval[c_pointer]
            r_count += 1
            r_pointer = Ri[upper_index_ID[r_count]]
            nzval_c = nzval[c_pointer]
            nzval_r = nzval[r_pointer]
            Atomix.@atomic D[j] -= nzval_c*nzval_r/D[i]
        end
        @synchronize
        D[i] = 1/D[i] # store inverse
    end
end

function forward_substitution!(x, P, b)
    (; A, D, Di, Ri, J) = P
    (; colptr, n, nzval, rowval) = A

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
end

function backward_substitution!(x, P, c)
    (; A, D, Di, Ri, J, upper_indices_IDs) = P
    (; colptr, n, nzval, rowval) = A

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
        # x[i] = c[i]/D[i]
        x[i] = c[i]*D[i]
    end
    for i ∈ (n-1):-1:1
        upper_index_ID = upper_indices_IDs[i]
        # FIND A WAY NOT TO USE IF STATEMENTS
        if i > 1
            if upper_indices_IDs[i] == upper_indices_IDs[i-1]
                c_pointers = []
            else
                c_pointers = Ri[upper_index_ID]
            end
        else
            c_pointers = Ri[upper_index_ID]
        end
        j = J[upper_index_ID]
        for (p_i, p) ∈ enumerate(c_pointers)
            # x[i] -= nzval[p]*x[j[p_i]]/D[i]
            x[i] -= nzval[p]*x[j[p_i]]*D[i]
        end
    end
end

ldiv!(x, P::DILUprecon{M,V,I}, b) where {M<:AbstractSparseArray,V,I} =
begin
    forward_substitution!(x, P, b)
    backward_substitution!(x, P, x)
end