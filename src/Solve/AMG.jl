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
    rnormb = 1/norm(b)
    for i ∈ 1:itmax
        res .= term1 .+ (term2)*x0
        # r = abs(mean(b .- mul!(x0, A, res))/mean(res))
        # r = mean(sqrt.((b .- mul!(x0, A, res)).^2))*rnormb
        r = mean(sqrt.((b .- mul!(x0, A, res)).^2)) #*rnormb
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
    
    r = similar(b)
    r_L1 = zeros(size(R1)[1])
    r_L2 = zeros(size(R2)[1])
    r_L3 = zeros(size(R3)[1])
    r_L4 = zeros(size(R4)[1])
    r_L5 = zeros(size(R5)[1])

    dx = zeros(length(b))
    dx_L1 = zeros(length(r_L1))
    dx_L2 = zeros(length(r_L2))
    dx_L3 = zeros(length(r_L3))
    dx_L4 = zeros(length(r_L4))
    dx_L5 = zeros(length(r_L5))
    dx_L1 .= 0.0

    
    # residual = abs(mean(r))/mean(res)

    p_iter = 10
    s_iter = 15
    i_iter = 15
    f_iter = 5
    residual = 0.0
    iterations = 500

    for i ∈ 1:iterations

        Jacobi_solver!(res, A, b, rDA, p_iter, 1e-20)
        r .= b .- A*res

        mul!(r_L1, R1, r)
        # mul!(dx_L1, Rt2, dx_L2)
        dx_L1 .= 0.0


        Jacobi_solver!(dx_L1, A_L1, r_L1, rDA1, s_iter, 1e-15)
        mul!(r_L2, R2, r_L1 .- A_L1*dx_L1)
        dx_L2 .= 0.0

        Jacobi_solver!(dx_L2, A_L2, r_L2, rDA2, s_iter, 1e-15)
        mul!(r_L3, R3, r_L2 .- A_L2*dx_L2)
        dx_L3 .= 0.0

        Jacobi_solver!(dx_L3, A_L3, r_L3, rDA3, s_iter, 1e-15)
        mul!(r_L4, R4, r_L3 .- A_L3*dx_L3)
        dx_L4 .= 0.0

        Jacobi_solver!(dx_L4, A_L4, r_L4, rDA4, length(r_L4), 1e-10)
        mul!(r_L5, R5, r_L4 .- A_L4*dx_L4)
        dx_L5 .= 0.0

        Jacobi_solver!(dx_L5, A_L5, r_L5, rDA5, length(r_L5), 1e-10)           
       
        # Up to refined levels 

        dx_L4 .+= Rt5*dx_L5 # correct previous solution
        # Jacobi_solver!(dx_L4, A_L4, r_L4, rDA4, i_iter, 1e-15)

        dx_L3 .+= Rt4*dx_L4 # correct previous solution
        # Jacobi_solver!(dx_L3, A_L3, r_L3, rDA3, i_iter, 1e-15)

        dx_L2 .+= Rt3*dx_L3 # correct previous solution
        # Jacobi_solver!(dx_L2, A_L2, r_L2, rDA2, i_iter, 1e-15)

        dx_L1 .+= Rt2*dx_L2 # correct previous solution
        Jacobi_solver!(dx_L1, A_L1, r_L1, rDA1, i_iter, 1e-15)

        res .+= Rt1*dx_L1 # correct previous solution
        # res .= res .+ 2*Rt1*Rt2*Rt3*Rt4*Rt5*dx_L5
        Jacobi_solver!(res, A, b, rDA, f_iter, 1e-10)
        r .= sqrt.((b .- A*res).^2)
        # residual = abs(mean(r)/norm(b))
        residual = mean(r)
        
        if residual < tol
            # println("Converged! Residual: $residual in $i iterations!")
            return
        end
        # println("Residual: $residual in $i iterations!")
    end
    # println("Not converged! Residual: $residual in $iterations iterations!")
    nothing
end