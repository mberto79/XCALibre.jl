export AMG, AMG!

struct AMG{RT}
    # restrictions::RT
    levels::RT
end

restriction_and_A(A0) = begin
    R1 = restriction(A0)
    A1 = R1*A0*transpose(R1)
    println("Number of cells: $(size(R1)[1])")
    R1, A1
end

AMG(A, b) = begin
    # R1, A1 = restriction_and_A(A)
    # R2, A2 = restriction_and_A(A1)
    # R3, A3 = restriction_and_A(A2)
    # R4, A4 = restriction_and_A(A3)
    # R5, A5 = restriction_and_A(A4)
    # R6, A6 = restriction_and_A(A5)
    # # R7, A7 = restriction_and_A(A6)

    # restrictions = (R1, R2, R3, R4, R5, R6)

    # AMG(restrictions)

    A.nzval .= 1.0

    L0 = Level(A)
    L1 = Level(L0)
    L2 = Level(L1)
    L3 = Level(L2)
    L4 = Level(L3)
    L5 = Level(L4)
    L6 = Level(L5)

    levels = (L0, L1,L2,L3, L4, L5, L6)
    AMG(levels)
end

struct Level{A,RA,R1,P,V, R2}
    A::A
    rDA::RA
    R::R1
    P::P
    bc::V
    xc::V
    RT0::R2
end
Level(A0) = begin
    R0 = restriction(A0)
    RT0 = transpose(R0)
    interpolation = (I - 0.65*inv(Diagonal(A0))*A0)*RT0
    R = transpose(interpolation) #Agg
    P = interpolation # transpose(R)
    A = R*A0*P
    bc = zeros(size(R)[1])
    xc = zeros(size(R)[1])
    # println("Number of cells: $(size(R)[1])")
    Level(A, inv(Diagonal(A)), R, P, bc, xc, RT0)
end
Level(L::Level) = Level(L.A)

update_level!(L::Level, A0) = begin
    (; A, rDA, R, P, RT0) = L
    interpolation = (I - 0.65*inv(Diagonal(A0))*A0)*RT0
    P.nzval .= interpolation.nzval
    # R = transpose(L.P) #Agg
    # P = interpolation # transpose(R)
    A.nzval .= (R*A0*P).nzval
    rDA.diag.nzval .= (inv(Diagonal(A))).diag.nzval
    # bc = zeros(size(R)[1])
    # xc = zeros(size(R)[1])
    # println("Number of cells: $(size(R)[1])")
    # Level(A, inv(Diagonal(A)), R, P, bc, xc, R0)
    nothing
end
update_level!(L1::Level, L0::Level) = update_level!(L1, L0.A)

Level(A0, R0) = begin
    interpolation = (I - 0.65*inv(Diagonal(A0))*A0)*transpose(R0)
    R = transpose(interpolation) #Agg
    P = interpolation # transpose(R)
    A = R*A0*P
    bc = zeros(size(R)[1])
    xc = zeros(size(R)[1])
    # println("Number of cells: $(size(R)[1])")
    Level(A, inv(Diagonal(A)), R, P, bc, xc)
end

Level(L::Level, Agg) = begin
    A0 = L.A
    interpolation = (I - 0.65*inv(Diagonal(A0))*A0)*transpose(Agg)
    R = transpose(interpolation) #Agg
    P = interpolation # transpose(R)
    A = R*A0*P
    # P = (I - 0.4*inv(Diagonal(A0))*A0)*transpose(R)
    bc = zeros(size(R)[1])
    xc = zeros(size(R)[1])
    # println("Number of cells: $(size(R)[1])")
    Level(A, inv(Diagonal(A)), R, P, bc, xc)
end

struct JacobiSolver{T,TX}
    ω::T
    temp::TX
    iter::Int
end
JacobiSolver(ω, x::TX; iter=1) where {T, TX<:AbstractArray{T}} = JacobiSolver{T,TX}(ω, similar(x), iter)
JacobiSolver(x::TX, ω=0.5; iter=1) where {T, TX<:AbstractArray{T}} = JacobiSolver{T,TX}(ω, similar(x), iter)

function (jacobi::JacobiSolver)(A, x, b)
    ω = jacobi.ω
    one = Base.one(eltype(A))
    temp = jacobi.temp
    z = zero(eltype(A))

    for i in 1:jacobi.iter
        @inbounds for col = 1:size(x, 2)
            for i = 1:size(A, 1)
                temp[i, col] = x[i, col]
            end

            for i = 1:size(A, 1)
                rsum = z
                diag = z

                for j in nzrange(A, i)
                    row = A.rowval[j]
                    val = A.nzval[j]

                    diag = ifelse(row == i, val, diag)
                    rsum += ifelse(row == i, z, val * temp[row, col])
                end

                xcand = (one - ω) * temp[i, col] + ω * ((b[i, col] - rsum) / diag)
                x[i, col] = ifelse(diag == 0, x[i, col], xcand)
            end
        end
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

# function (amg::AMG)(res, A, b, tol)
# function AMG!(amg, Rs, res, A, b, tol)
function AMG!(amg, res, A, b, tol, iteration)
    # Rs = amg.restrictions

    # @time begin

    # L0 = Level(A, Rs[1])
    # L1 = Level(L0, Rs[2])
    # L2 = Level(L1, Rs[3])
    # L3 = Level(L2, Rs[4])
    # L4 = Level(L3, Rs[5])
    # L5 = Level(L4, Rs[6])

    # end

    # levels = (L0, L1,L2,L3, L4, L5)

    # @time begin
    # L0 = Level(A)
    # L1 = Level(L0)
    # L2 = Level(L1)
    # L3 = Level(L2)
    # L4 = Level(L3)
    # L5 = Level(L4)
    # L6 = Level(L5)
    # end

    # levels = (L0, L1,L2,L3, L4, L5, L6)

    levels = amg.levels
    update_interval = 20
    if (iteration + update_interval - 1) % update_interval == 0
        update_level!(levels[1], A)
        update_level!(levels[2], levels[1])
        update_level!(levels[3], levels[2])
        update_level!(levels[4], levels[3])
        update_level!(levels[5], levels[4])
        update_level!(levels[6], levels[5])
    end

    smoother0 = JacobiSolver(res, 0.65, iter=3)
    smoother1 = JacobiSolver(levels[1].xc, 0.65, iter=3)
    smoother2 = JacobiSolver(levels[2].xc, 0.65, iter=3)
    smoother3 = JacobiSolver(levels[3].xc, 0.65, iter=3)
    smoother4 = JacobiSolver(levels[4].xc, 0.65, iter=3)
    smoother5 = JacobiSolver(levels[5].xc, 0.65, iter=3)
    smoother6 = JacobiSolver(levels[6].xc, 0.65, iter=3)

    for i ∈ 1:100

        smoother0(A, res, b)
    
        # level 1 corrections
        levels[1].xc .= 0.0
        mul!(levels[1].bc, levels[1].R, b .- A*res)
        smoother1(levels[1].A, levels[1].xc, levels[1].bc)
    
        # level 2 corrections
        levels[2].xc .= 0.0
        mul!(levels[2].bc, levels[2].R, levels[1].bc .- levels[1].A*levels[1].xc)
        smoother2(levels[2].A, levels[2].xc, levels[2].bc)

        # level 3 corrections
        levels[3].xc .= 0.0
        mul!(levels[3].bc, levels[3].R, levels[2].bc .- levels[2].A*levels[2].xc)
        smoother3(levels[3].A, levels[3].xc, levels[3].bc)

        # level 4 corrections
        levels[4].xc .= 0.0
        mul!(levels[4].bc, levels[4].R, levels[3].bc .- levels[3].A*levels[3].xc)
        smoother4(levels[4].A, levels[4].xc, levels[4].bc)

        # level 5 corrections
        levels[5].xc .= 0.0
        mul!(levels[5].bc, levels[5].R, levels[4].bc .- levels[4].A*levels[4].xc)
        smoother5(levels[5].A, levels[5].xc, levels[5].bc)
    
        # Coarsest corrections
        # levels[3].xc .= 0.0
        mul!(levels[6].bc, levels[6].R, levels[5].bc .- levels[5].A*levels[5].xc)
        # smoother3(levels[3].A, levels[3].xc, levels[3].bc)
        levels[6].xc .= levels[6].A\levels[6].bc
    
        # return corrections 

        levels[5].xc .+= levels[6].P*levels[6].xc
        smoother5(levels[5].A, levels[5].xc, levels[5].bc)

        levels[4].xc .+= levels[5].P*levels[5].xc
        smoother4(levels[4].A, levels[4].xc, levels[4].bc)

        levels[3].xc .+= levels[4].P*levels[4].xc
        smoother3(levels[3].A, levels[3].xc, levels[3].bc)
    
    
        levels[2].xc .+= levels[3].P*levels[3].xc
        smoother2(levels[2].A, levels[2].xc, levels[2].bc)
    
    
        levels[1].xc .+= levels[2].P*levels[2].xc
        smoother1(levels[1].A, levels[1].xc, levels[1].bc)
    
    
        res .+= levels[1].P*levels[1].xc
        smoother0(A, res, b)
    
    
        r = norm(b - A*res)/norm(b)
        # r = norm(b - A*res)
        # println("Residual: $r (iteration $i)")
        if r < tol
            # println("Converged! Residual: $r (iteration $i)")
            # return
            break
        end
    
    end
    nothing
end