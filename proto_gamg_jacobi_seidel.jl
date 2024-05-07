using FVM_1D
using LinearAlgebra
using SparseArrays
using Statistics
using Krylov
using AlgebraicMultigrid

mesh_file = "unv_sample_meshes/quad.unv"
mesh_file = "unv_sample_meshes/quad40.unv"
mesh_file = "unv_sample_meshes/quad100.unv"
mesh_file = "unv_sample_meshes/trig40.unv"
mesh_file = "unv_sample_meshes/trig100.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)

k = 100

k = ConstantScalar(k)
T = ScalarField(mesh)
Q = ScalarField(mesh)


T_eqn = (
        Time{Steady}(T)
        - Laplacian{Linear}(k, T) 
        == 
        -Source(Q)
    ) → Equation(mesh)

T = assign(T, 
        Dirichlet(:inlet, 500),
        Dirichlet(:outlet, 0),
        Dirichlet(:bottom, 100),
        Dirichlet(:top, 100)
)

runtime = set_runtime(iterations=1, write_interval=1, time_step=1)
prev = T.values 
discretise!(T_eqn, prev, runtime)
apply_boundary_conditions!(T_eqn, T.BCs)

(; A, b) = T_eqn.equation



# Jacobi solver
function Jacobi_solver!(res, A::T, b, rD, ω, itmax, tol; verbose=false) where T
    # @time rD = inv(Diagonal(A))
    x0 = copy(res)
    residual = 0.0
    # ω = 2/3 # 1
    @inbounds term1 = ω*rD*b
    @inbounds term2 = I - ω*rD*A
    # rnormb = 1/norm(b)
    @inbounds for i ∈ 1:itmax
        # mul!(res, term2, x0)
        res .= term1 .+ term2*x0
        # residual = abs(mean(b .- mul!(x0, A, res))/mean(res))
        # residual = mean(sqrt.((b .- mul!(x0, A, res)).^2))*rnormb
        mul!(x0, A, res)
        x0 .= b .- x0
        residual = norm(x0)
        if residual <= tol
            if verbose
            println("Converged! Residual: $residual ($i iterations): ", residual)
            end
            return
        end
        x0 .= res
    end
    if verbose
        println("No converged! Residual: $residual ($itmax iterations)")
    end
end

struct Jacobi{T,TX}
    ω::T
    temp::TX
    iter::Int
end
Jacobi(ω, x::TX; iter=1) where {T, TX<:AbstractArray{T}} = Jacobi{T,TX}(ω, similar(x), iter)
Jacobi(x::TX, ω=0.5; iter=1) where {T, TX<:AbstractArray{T}} = Jacobi{T,TX}(ω, similar(x), iter)

function (jacobi::Jacobi)(A, x, b)
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

smoother = Jacobi(T.values, 2/3, iter=10)
@time smoother(A, T.values, b)


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
        # i_counter += 1
        for i ∈ nzrange(A,j)
            cID = rowsA[i]
            # if cID == j+1
            #     continue
            # end
            # if cID == j
            #     if cell_used[diag_cell] == 1
            #         continue
            #     end
            # end
            if cell_used[cID] == 0 #&& cID > j+1
                push!(cellsID, cID)
                # push!(i_vals,i_counter)
                # push!(j_vals,cID)
                # push!(v_vals,1)
                cell_used[cID] = 1
            end
        end
        push!(agglomeration, cellsID)
    end

    # i = zeros(Int64, length(agglomeration))
    # j = zeros(Int64, length(agglomeration))
    # v = ones(Int64, length(agglomeration))
    i = Int64[]
    j = Int64[]
    v = Int64[]
    for (ii, cIDs) ∈ enumerate(agglomeration) 
        for jj ∈ cIDs 
            # i[jj] = ii
            # j[jj] = jj
            # i[jj] = ii
            # j[jj] = jj
            push!(i,ii)
            push!(j,jj)
            push!(v,1)
        end
    end

    return sparse(i, j, v)
end

struct Level{A,RA,R,P,V}
    A::A
    rDA::RA
    R::R
    P::P
    bc::V
    xc::V
end
Level(A0) = begin
    Agg = restriction(A0)
    interpolation = (I - 0.65*inv(Diagonal(A0))*A0)*transpose(Agg)
    R = transpose(interpolation) #Agg
    P = interpolation # transpose(R)
    A = R*A0*P
    # P = (I - 0.4*inv(Diagonal(A0))*A0)*transpose(R)
    bc = zeros(size(R)[1])
    xc = zeros(size(R)[1])
    println("Number of cells: $(size(R)[1])")
    Level(A, inv(Diagonal(A)), R, P, bc, xc)
end

Level(L::Level) = begin
    A0 = L.A
    Agg = restriction(A0)
    interpolation = (I - 0.65*inv(Diagonal(A0))*A0)*transpose(Agg)
    R = transpose(interpolation) #Agg
    P = interpolation # transpose(R)
    A = R*A0*P
    # P = (I - 0.4*inv(Diagonal(A0))*A0)*transpose(R)
    bc = zeros(size(R)[1])
    xc = zeros(size(R)[1])
    println("Number of cells: $(size(R)[1])")
    Level(A, inv(Diagonal(A)), R, P, bc, xc)
end
Level(L::Level, x) = begin
    A0 = L.A
    Agg = restriction(A0)
    interpolation = (I - 0.65*inv(Diagonal(A0))*A0)*transpose(Agg)
    R = transpose(interpolation) #Agg
    P = interpolation # transpose(R)
    A = R*A0*P
    # P = (I - 0.4*inv(Diagonal(A0))*A0)*transpose(R)
    bc = zeros(size(R)[1])
    xc = zeros(size(R)[1])
    println("Number of cells: $(size(A)[1])")
    Level(A, inv(Diagonal(A)), R, P, bc, xc)
end

L0 = Level(A)
L1 = Level(L0)
L2 = Level(L1)
L3 = Level(L2)
L4 = Level(L3)
L5 = Level(L4,1)

L5.P
L5.xc
L5.R
L5.rDA
L5.A

L4.P

L5.xc .= [1:length(L5.xc);]./length(L5.xc)
T.values .= L0.P*L1.P*L2.P*L3.P*L4.P*L5.P*L5.xc
write_vtk("Level_5", mesh, ("T", T))

L4.xc .= [1:length(L4.xc);]./length(L4.xc)
T.values .= L0.P*L1.P*L2.P*L3.P*L4.P*L4.xc
write_vtk("Level_4", mesh, ("T", T))

L3.xc .= [1:length(L3.xc);]./length(L3.xc)
T.values .= L0.P*L1.P*L2.P*L3.P*L3.xc
write_vtk("Level_3", mesh, ("T", T))

L2.xc .= [1:length(L2.xc);]./length(L2.xc)
T.values .= L0.P*L1.P*L2.P*L2.xc
write_vtk("Level_2", mesh, ("T", T))

L1.xc .= [1:length(L1.xc);]./length(L1.xc)
T.values .= L0.P*L1.P*L1.xc
write_vtk("Level_1", mesh, ("T", T))

L0.xc .= [1:length(L0.xc);]./length(L0.xc)
# T.values .= (I - 0.5*inv(Diagonal(A))*A)*L0.P*L0.xc
T.values .= L0.P*L0.xc
write_vtk("Level_0", mesh, ("T", T))


levels = (L0,L1,L2,L3,L4,L5)


pre_iter = 3
res_iter = 10
int_iter = 1
post_iter = 3
@time Jacobi_solver!(T.values, A, b, inv(Diagonal(A)), 2/3, 10, 1e-20)
@time Jacobi_solver!(T.values, A, b, inv(Diagonal(A)), 1, 10, 1e-20)

smoother0 = Jacobi(T.values, 0.65, iter=3)
smoother1 = Jacobi(levels[1].xc, 0.65, iter=3)
smoother2 = Jacobi(levels[2].xc, 0.65, iter=3)
smoother3 = Jacobi(levels[3].xc, 0.65, iter=3)
smoother4 = Jacobi(levels[4].xc, 0.65, iter=3)

T.values .= 100.0
xfine = T.values
tol = 1e-5
@time for i ∈ 1:100
    

    # top level smoother
    # Jacobi_solver!(x, A, b, inv(Diagonal(A)), 2/3, pre_iter, 1e-20)
    smoother0(A, xfine, b)

    # level 1 corrections
    levels[1].xc .= 0.0
    mul!(levels[1].bc, levels[1].R, b .- A*xfine)
    # Jacobi_solver!(levels[1].xc, levels[2].A, levels[1].bc, levels[2].rDA, 2/3, res_iter, 1e-20)
    levels[2].A
    levels[1].xc
    smoother1(levels[1].A, levels[1].xc, levels[1].bc)

    # level 2 corrections
    levels[2].xc .= 0.0
    mul!(levels[2].bc, levels[2].R, levels[1].bc .- levels[1].A*levels[1].xc)
    # Jacobi_solver!(levels[2].xc, levels[3].A, levels[2].bc, levels[3].rDA, 2/3, res_iter, 1e-20)
    smoother2(levels[2].A, levels[2].xc, levels[2].bc)

    # level 3 corrections
    levels[3].xc .= 0.0
    mul!(levels[3].bc, levels[3].R, levels[2].bc .- levels[2].A*levels[2].xc)
    # Jacobi_solver!(levels[2].xc, levels[3].A, levels[2].bc, levels[3].rDA, 2/3, res_iter, 1e-20)
    smoother3(levels[3].A, levels[3].xc, levels[3].bc)


    # level 4 corrections
    # levels[3].xc .= 0.0
    mul!(levels[4].bc, levels[4].R, levels[3].bc .- levels[3].A*levels[3].xc)

    levels[4].xc .= levels[4].A\levels[4].bc

    # return corrections 
    levels[3].xc .+= levels[4].P*levels[4].xc
    # Jacobi_solver!(levels[2].xc, levels[3].A, levels[2].bc, levels[3].rDA, 2/3, int_iter, 1e-20)
    smoother3(levels[3].A, levels[3].xc, levels[3].bc)


    levels[2].xc .+= levels[3].P*levels[3].xc
    # Jacobi_solver!(levels[2].xc, levels[3].A, levels[2].bc, levels[3].rDA, 2/3, int_iter, 1e-20)
    smoother2(levels[2].A, levels[2].xc, levels[2].bc)


    levels[1].xc .+= levels[2].P*levels[2].xc
    # Jacobi_solver!(levels[1].xc, levels[2].A, levels[1].bc, levels[2].rDA, 2/3, int_iter, 1e-20)
    smoother1(levels[1].A, levels[1].xc, levels[1].bc)


    xfine .+= levels[1].P*levels[1].xc
    # Jacobi_solver!(x, A, b, inv(Diagonal(A)), 2/3, post_iter, 1e-20)
    smoother0(A, xfine, b)


    r = norm(b - A*xfine)/norm(b)
    # r = norm(b - A*xfine)
    # println("Residual: $r (iteration $i)")
    if r < tol
        println("Converged! Residual: $r (iteration $i)")
        break
    end

end
write_vtk("test", mesh, ("T", T))

function vcycle(mgrid::M, x, b, level, tol) where M

    p_iter = 15
    s_iter = 5
    i_iter = 5
    f_iter = 5

    (; A, rDA, R, P, xc, bc) = mgrid[level]
    mul!(bc, R, b .- A*x)
    xc .= 0.0

    # println("Prelevel $level")

    if level == length(mgrid)
        xc .= A\b
        # Jacobi_solver!(corr, A, b, rDA, 2/3, p_iter, 1e-20)
    else
        xc .= vcycle(mgrid, xc, bc, level + 1, 1e-20) 
    end

    # println("post-level $level")

    x .+= P*xc

    Jacobi_solver!(x, A, b, rDA, 1.5, f_iter, 1e-20)
    x
end

function AMG!(mgrid::M, x, b, tol) where M
    iterations = 100
    residual = 0.0
    x0 = copy(x)
    temp = similar(x0)
    @inbounds for i ∈ 1:iterations
        x0 .= x
        x .= vcycle(mgrid, x, b, 1, tol)
        mul!(temp, A, x)
        temp .= b .- temp
        residual = norm(temp)
        # residual = abs(maximum(x .- x0)/mean(x))
        if residual <= tol
            println("Converged! Residual: $residual ($i iterations): ", residual)
            return
        end
    end
    println("Not converged! Residual: $residual")
end



tol = 0.005
T.values .= 100
@time AMG!(mgrid, T.values, b, tol)
write_vtk("result_AMG", mesh, ("T", T))

T.values .= 100
@time Jacobi_solver!(T.values, A, b, inv(Diagonal(A)), 1, 20000, 1e-3, verbose=true)
write_vtk("result_Jacobi", mesh, ("T", T))

T.values .= 100
JacobiSolver = Jacobi(T.values, 1, iter=20000)
@time JacobiSolver(A, T.values, b)
write_vtk("result_Jacobi_AMG", mesh, ("T", T))


T.values .= 100.0
@time T.values .= AlgebraicMultigrid.solve(
    A, b, SmoothedAggregationAMG(), maxiter = 2000, abstol = 1e-15)
r = norm(b - A*T.values)
write_vtk("result_RugeStubenAMG", mesh, ("T", T))

T.values .= 100.0
@time T.values .= A\b
r = norm(b - A*T.values)
write_vtk("direct_result", mesh, ("T", T))

T.values .= 100.0
@time xs, stats = cg(A, b)
@time xs, stats = bicgstab(A, b)
T.values .= xs
write_vtk("result_Krylov", mesh, ("T", T))