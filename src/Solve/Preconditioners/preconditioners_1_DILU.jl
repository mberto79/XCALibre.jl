import LinearAlgebra.ldiv!, LinearAlgebra.\

export ldiv!

# THIS WHOLE IMPLEMENTATION NEEDS TO BE CLEANED UP AND CHECKED FOR CORRECTNESS
# NOTE ADDED ON 2024/11/07 - THIS IS LIKELY BROKEN DUE TO CHANGE TO CSR FORMAT


function extract_diagonal!(D, Di, A::AbstractSparseArray, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    (; nzval, n) = A

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

# integer_type(A::SparseMatrixCSC{Tf,Ti}) where {Tf,Ti} = Ti
# integer_type(A::CUDA.CUSPARSE.CuSparseMatrixCSC{Tf,Ti}) where {Tf,Ti} = Ti

function upper_row_indices(A, Di) # upper triangular row column indices
    (; colptr, n, rowval) = A
    TI = eltype(colptr)
    Ri = TI[] # column pointers on i-th row
    J = TI[] # column indices on i-th row
    upper_indices_IDs = UnitRange{TI}[]
    @inbounds for i ∈ 1:n
        R_temp = TI[]
        J_temp = TI[]
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
            IDs_range = UnitRange{TI}(upper_indices_start:upper_indices_end)
        else
            IDs_range = UnitRange{TI}(upper_indices_start+1:upper_indices_end)
        end
        push!(upper_indices_IDs, IDs_range)
    end
    return Ri, J, upper_indices_IDs
end

function update_dilu_diagonal!(P, mesh, config) # must rename
    # # Algo 1
    # (; A, storage) = P
    # (; m) = A
    # (; D) = storage
    # for i ∈ 1:m 
    #     D[i] = A[i,i]
    # end

    # for i ∈ 1:m
    #     for j ∈ (i+1):m
    #         D[j] = D[j] - A[i,j]*A[j,i]/D[i]
    #     end
    # end

    # Algo 2
    (; A, storage) = P 
    (; rowptr, colval, nzval, m, n) = A
    (; D, Di) = storage
    extract_diagonal!(D, Di, A, config)
    @inbounds for i ∈ 1:m
        @inbounds for j ∈ (i+1):m
            nzIndex1 = spindex(rowptr, colval, i, j)
            nzIndex2 = spindex(rowptr, colval, j, i)
            if nzIndex1 !== 0 && nzIndex2 !== 0
                D[j] = D[j] - nzval[nzIndex1]*nzval[nzIndex2]/D[i]
            end
        end
    end
    nothing
end


# function update_dilu_diagonal!(P, mesh, config) # must rename
#     # (; A, storage) = P
#     # (; rowptr, m, n, nzval, colval) = A
    # (; Di, D) = storage
    # extract_diagonal!(D, Di, A) 
    # for i ∈ 1:m
    #     # for j ∈ 1:(i-1)
    #     for j ∈ (i+1):m
    #         # D[i] -= A[i,j]*A[j,i]/D[j]
    #         D[j] -= A[i,j]*A[j,i]/D[i]
    #     end
    # end

#     # Algo 2
#     # (; A, storage) = P
#     # (; rowptr, m, n, nzval, colval) = A
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
#     # backend = _get_backend(mesh)

#     (; hardware) = config
#     (; backend, workgroup) = hardware

#     (; A, storage) = P
#     # (; rowptr, n, nzval, colval) = A
#     colval, rowptr, nzval, m ,n = sparse_array_deconstructor_preconditioners(A)
#     (; Di, Ri, D, upper_indices_IDs) = storage
    
#     extract_diagonal!(D, Di, A, config)

#     kernel! = update_dilu_diagonal_kernel!(backend, workgroup)
#     kernel!(upper_indices_IDs, Di, rowptr, Ri, colval, D, nzval, ndrange = n)
#     # D .= 1.0./D # store inverse
#     nothing
# end


# @kernel function update_dilu_diagonal_kernel!(upper_indices_IDs, Di, rowptr, Ri, colval, D, nzval)
#     i = @index(Global)
    
#     @inbounds begin
#         # D[i] = nzval[Di[i]]
#         upper_index_ID = upper_indices_IDs[i] 
#         c_start = Di[i] + 1 
#         c_end = rowptr[i+1] - 1
#         r_count = 0
#         for c_pointer ∈ c_start:c_end
#             j = colval[c_pointer]
#             r_count += 1
#             r_pointer = Ri[upper_index_ID[r_count]]
#             nzval_c = nzval[c_pointer]
#             nzval_r = nzval[r_pointer]
#             Atomix.@atomic D[j] -= nzval_c*nzval_r/D[i]
#         end
#         @synchronize
#         D[i] = 1/D[i] # store inverse
#     end
# end

function forward_substitution!(y, P::DILUprecon{M,V,VI,VUR}, b) where {M,V,VI,VUR}
    # # Algo 1
    # m, n = size(PL)
    # for i ∈ 1:m
    #     sum = 0.0
    #     for j ∈ 1:(i-1)
    #         sum += PL[i,j]*y[j]
    #     end
    #     y[i] = (b[i] - sum)
    # end

    # # Algo 2
    (; A, D, Di) = P
    (; rowptr, colval, nzval, n, m) = A
    y .= b
    for i ∈ 1:m
        start_index = rowptr[i]
        end_index = Di[i] - 1
        sum = 0.0
        for nzi ∈ start_index:end_index # needs serious check!
            j = colval[nzi]
            println("i = $i, j = $j, $start_index, $end_index")
            # if j <= i
                A_ij = ifelse(i == j, 1, nzval[nzi])
                sum += A_ij*y[j]
                # y[i] -= A_ij*y[j]
            # end
        end
        y[i] = (b[i] - sum)
    end
end

function backward_substitution!(x, P::DILUprecon{M,V,VI,VUR}, y) where {M,V,VI,VUR}
    # # Algo 1
    # m, n = size(PU)
    # for i ∈ (n):-1:1
    #     sum = 0.0
    #     for j ∈ (i+1):n # needs serious check!
    #         sum += PU[i,j]*x[j]
    #     end
    #     x[i] = (y[i] - sum)/PU[i,i]
    # end

    # Algo 2
    (; A, D, Di) = P
    (; rowptr, colval, nzval, n, m) = A
    for i ∈ n:-1:1
        start_index = Di[i] + 1
        end_index = rowptr[i+1] - 1
        sum = 0.0
        for nzi ∈ start_index:end_index # needs serious check!
            j = colval[nzi]
            if j >= i
                A_ij = ifelse(i == j, D[i], nzval[nzi])
                sum += A_ij*x[j]
            end
        end
        x[i] = (y[i] - sum)/D[i]
    end
end

ldiv!(x, P::DILUprecon{M,V,I}, b) where {M<:AbstractSparseArray,V,I} =
begin
    y = zeros(length(x))
    forward_substitution!(y, P, b)
    backward_substitution!(x, P, y)
end