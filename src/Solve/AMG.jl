export AMG, AMG!

struct AMG end

AMG(a,b) = AMG()

function Jacobi_solver!(res, A, b, rD, itmax, tol)
    # @time rD = inv(Diagonal(A))
    x0 = similar(res)
    x0 .= res
    r = 0.0
    term1 = rD*b
    term2 = I - rD*A
    # rmean = 1/mean(res)
    for i ∈ 1:itmax
        res .= term1 .+ (term2)*x0
        r = abs(mean(b .- mul!(x0, A, res))/mean(res))
        if r <= tol
            # println("Converged! Residual ($i iterations): ", r)
            return
        end
        x0 .= res
    end
end

function restriction(A)
    rowsA = rowvals(A)
    cell_used = zeros(Int64, size(A)[1])
    agglomeration = Vector{Int64}[]
    cellsID = Int64[]
    diag_cell = 0
    for j ∈ 1:size(A)[1]
        cellsID = Int64[]
        for ii ∈ nzrange(A,j)
            indx = rowsA[ii]
            if indx == j # pick up Diagonal
                diag_cell = j
            end
        end
        if cell_used[diag_cell] == 1
            continue
        end
        for i ∈ nzrange(A,j)
            cID = rowsA[i]
            if cell_used[cID] == 0 #&& cID > j+1
                push!(cellsID, cID)
                cell_used[cID] = 1
            end
        end
        push!(agglomeration, cellsID)
    end

    i = Int64[]
    j = Int64[]
    v = Int64[]
    for (ii, cIDs) ∈ enumerate(agglomeration) 
        for jj ∈ cIDs 
            push!(i,ii)
            push!(j,jj)
            push!(v,1)
        end
    end

    return sparse(i, j, v)
end

function AMG!(res, A, b, tol)
    rDA = inv(Diagonal(A))
    R1 = restriction(A)
    Rt1 = transpose(R1)

    A_L1 = R1*A*Rt1
    rDA1 = inv(Diagonal(A_L1))
    # println("Level 1: $(size(A_L1)[1]) cells")
    
    R2 = restriction(A_L1)
    Rt2 = transpose(R2)

    A_L2 = R2*A_L1*Rt2
    rDA2 = inv(Diagonal(A_L2))
    # println("Level 2: $(size(A_L2)[1]) cells")


    R3 = restriction(A_L2)
    Rt3 = transpose(R3)

    A_L3 = R3*A_L2*Rt3
    rDA3 = inv(Diagonal(A_L3))
    # println("Level 3: $(size(A_L3)[1]) cells")


    R4 = restriction(A_L3)
    Rt4 = transpose(R4)

    A_L4 = R4*A_L3*Rt4
    rDA4 = inv(Diagonal(A_L4))
    # println("Level 4: $(size(A_L4)[1]) cells")

    R5 = restriction(A_L4)
    Rt5 = transpose(R5)

    A_L5 = R5*A_L4*Rt5
    rDA5 = inv(Diagonal(A_L5))
    # println("Level 5: $(size(A_L5)[1]) cells")

    R6 = restriction(A_L5)
    Rt6 = transpose(R6)
    A_L6 = R6*A_L5*Rt6
    rDA6 = inv(Diagonal(A_L6))
    # println("Level 6: $(size(A_L6)[1]) cells")
    
    r = similar(b)
    r_L1 = zeros(size(R1)[1])
    r_L2 = zeros(size(R2)[1])
    r_L3 = zeros(size(R3)[1])
    r_L4 = zeros(size(R4)[1])
    r_L5 = zeros(size(R5)[1])
    r_L6 = zeros(size(R6)[1])

    dx = zeros(length(b))
    dx_L1 = zeros(length(r_L1))
    dx_L2 = zeros(length(r_L2))
    dx_L3 = zeros(length(r_L3))
    dx_L4 = zeros(length(r_L4))
    dx_L5 = zeros(length(r_L5))
    dx_L6 = zeros(length(r_L6))
    dx_L1 .= 0.0

    Jacobi_solver!(res, A, b, rDA, 1000, 0.2)
    r .= b .- A*res
    # residual = abs(mean(r))/mean(res)

    s_iter = 1
    f_iter = 50

    for i ∈ 1:750

        r .= b .- A*res

        mul!(r_L1, R1, r)
        mul!(dx_L1, Rt2, dx_L2)
        Jacobi_solver!(dx_L1, A_L1, r_L1, rDA1, s_iter, tol)
        r_L1 .-= A_L1*dx_L1
        
        mul!(r_L2, R2, r_L1)
        mul!(dx_L2, Rt3, dx_L3)
        Jacobi_solver!(dx_L2, A_L2, r_L2, rDA2, s_iter, tol)
        r_L2 .-= A_L2*dx_L2 

        mul!(r_L3, R3, r_L2)
        mul!(dx_L3, Rt4, dx_L4)
        Jacobi_solver!(dx_L3, A_L3, r_L3, rDA3, s_iter, tol)
        r_L3 .-= A_L3*dx_L3 
        
        mul!(r_L4, R4, r_L3)
        mul!(dx_L4, Rt5, dx_L5)
        Jacobi_solver!(dx_L4, A_L4, r_L4, rDA4, s_iter, tol)
        r_L4 .-= A_L4*dx_L4 
        
        mul!(r_L5, R5, r_L4)
        dx_L5 .= 0.0 # R5*dx_L4 # 0.0
        Jacobi_solver!(dx_L5, A_L5, r_L5, rDA5, s_iter, tol)
        r_L5 .-= A_L5*dx_L5 

        mul!(r_L6, R6, r_L5)
        dx_L6 .= 0.0 # R5*dx_L4 # 0.0
        Jacobi_solver!(dx_L6, A_L6, r_L6, rDA6, 2000, 1e-10)
        r_L6 .-= A_L6*dx_L6

        # Up to refined levels 

        dx_L5 .+= Rt6*dx_L6 # correct previous solution
        dx_L4 .+= Rt5*dx_L5 # correct previous solution
        dx_L3 .+= Rt4*dx_L4 # correct previous solution
        dx_L2 .+= Rt3*dx_L3 # correct previous solution
        dx_L1 .+= Rt2*dx_L2 # correct previous solution
        res .+= Rt1*dx_L1 # correct previous solution
        Jacobi_solver!(res, A, b, rDA, f_iter, 0.2)

        residual = abs(mean(r)/mean(res))
        
        if residual < tol
            # println("Converged! Residual: $residual in $i iterations!")
            return
        end
    end
    nothing
end