import LinearAlgebra.ldiv!, LinearAlgebra.\

export extract_diagonal!
export dilu_diagonal1!, dilu_diagonal2!
export forward_substitution, backward_substitution
export ldiv!, left_div!



function extract_diagonal!(D, A) 
    m, n = size(A)
    for i ∈ 1:m
        D[i] = A[i,i]
    end
end

function dilu_diagonal1!(D, A)
    m, n = size(A)
    for i ∈ 1:m
        for j ∈ (i+1):m
            D[j] -= A[j,i]*A[i,j]/D[i]
            # D[j] -= A[j,i]*A[i,j]/A[i,i]
        end
    end
end

function dilu_diagonal2!(D, A)
    m, n = size(A)
    # D[1] = A[1,1] # set first element of D
    for i ∈ 1:m
        sum = 0.0
        for j ∈ 1:(i-1)
            sum += A[i,j]*D[j]*A[j,i]
        end
        D[i] = A[i,i] - sum
    end
end

function forward_substitution(A, D, y)
    m, n = size(A)
    t = zeros(eltype(D), m)
    for i ∈ 1:m
        for j ∈ 1:(i-1) # needs serious check!
            t[i] = (1/D[i])*(y[i] - A[i,j]*t[j])
        end
    end
    return t
end

function backward_substitution(A, D, b)
    m, n = size(A)
    x = zeros(eltype(D), m)
    for i ∈ m:-1:1 # needs serious check!
        for j ∈ (i+1):m
            x[i] = b[i] - A[i,j]*b[j]/D[i]
        end
    end
    return x
end

function left_div!(x, A, D, b)
    # m, n = size(A)
    # c = zeros(eltype(b), n)
    # c[1] = b[1]/D[1]
    # # c[1] = b[1]
    # # for i ∈ 1:m
    # for i ∈ 2:m
    #     for j ∈ 1:(i-1)
    #     # for j ∈ 1:(i-1)
    #         c[i] = (1/D[i])*(b[i] - A[i,j]*c[j])
    #         # c[i] = (b[i] - A[i,j]*c[j])
    #     end
    # end
    # x[m] = c[m] #- (1/D[m])
    # for i ∈ (m-1):-1:1
    #     for j ∈ (i+1):m
    #         x[i] = c[i] - (1/D[i])*(A[i,j]*c[j])
    #     end
    # end
    y = forward_substitution(A, D, b)
    x .= backward_substitution(A, D, y)
    nothing
end

function ldiv!(y, P::DILUprecon{M,V,DD,L,U,DA}, b
    ) where {M<:SparseMatrixCSC,V,DD,L,U,DA}
    ###
    D = P.diagonal
    A = P.A
    left_div!(y, A, D, b)
    ###
    # A = P.A
    # D = P.D
    # Da = P.Da
    # La = P.L 
    # Ua = P.U

    # t1 = (D .+ Ua .- Da)\b
    # t2 = b .+ (Da .- 2.0.*D)*t1
    # t3 = (La .- Da .+ D)\t2
    # y .= D*(t1.+t3)

    # # Initial implementation below

    # # t1 = (D + UpperTriangular(A) - Da)\b
    # # t2 = b + (Da - 2*D)*t1
    # # t3 = (LowerTriangular(A) - Da + D)\t2
    # # y .= D*(t1+t3)
    
    nothing
end
