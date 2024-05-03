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

T.values .= 100.0
@time T.values .= A\b
write_vtk("direct_result", mesh, ("T", T))

T.values .= 100.0
solver = CgSolver(A, b)
@time solve!(solver, A, b, T.values)
@time xs, stats = cg(A, b)
@time xs, stats = bicgstab(A, b)
T.values .= xs
write_vtk("result_Krylov", mesh, ("T", T))

# Jacobi solver
function Jacobi_solver!(res, A, b, rD, itmax, tol; verbose=false)
    # @time rD = inv(Diagonal(A))
    x0 = similar(res)
    x0 .= res
    residual = 0.0
    term1 = rD*b
    term2 = I - rD*A
    rnormb = 1/norm(b)
    for i ∈ 1:itmax
        mul!(res, term2, x0)
        res .= term1 .+ res
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

# U = UpperTriangular(A)
# D = Diagonal(A)
# L = LowerTriangular(A)

# rowsA = rowvals(A)
# rowsL = rowvals(L)

# col1 = nzrange(A,2)
# col2 = nzrange(L,99)


# lrows = Int64[]
# for j ∈ 1:size(L)[1]
#     for i ∈ nzrange(L,j)
#         push!(lrows, rowsL[i])
#     end
# end

# cell_used = zeros(Int64, length(mesh.cells))
# agglomeration = Vector{Int64}[]

# cellsID = Int64[]
# for j ∈ 1:size(A)[1]
#     diag_cell= rowsA[nzrange(L,j)[1]]
#     if cell_used[diag_cell] == 0
#         cellsID = Int64[]
#         for i ∈ nzrange(L,j)
#             cID = rowsL[i]
#             if cell_used[cID] == 0
#                 push!(cellsID, cID)
#                 cell_used[cID] = 1
#             end
#         end
#     push!(agglomeration, cellsID)
#     end 
# end

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

R = restriction(A)
Rt = transpose(R)

struct Level{A,RA,R,P,V}
    A::A
    rDA::RA
    R::R
    P::P
    res::V
    corr::V
end
Level(A) = begin
    R = restriction(A)
    P = transpose(R)
    res = zeros(size(R)[1])
    corr = zeros(size(R)[1])
    Level(A, inv(Diagonal(A)), R, P, res, corr)
end

Level(L::Level) = Level(L.R*L.A*L.P)
Level(L::Level, x) = begin
    A = L.R*L.A*L.P
    # R = restriction(A)
    R = 1
    P = transpose(R)
    res = zeros(size(A)[1])
    corr = zeros(size(A)[1])
    Level(A, inv(Diagonal(A)), R, P, res, corr)
end

L0 = Level(A)
L1 = Level(L0)
L2 = Level(L1)
L3 = Level(L2)
L4 = Level(L3)
L5 = Level(L4, 1)
mgrid = [L0,L1,L2,L3,L4,L5]

function vcycle(mgrid, x, b, level, tol)

    p_iter = 20
    s_iter = 5
    i_iter = 5
    f_iter = 5

    # println("level $level")

    l = mgrid[level]
    (; A, rDA, R, P, res, corr) = l

    Jacobi_solver!(x, A, b, rDA, p_iter, 1e-20)
    mul!(res, R, b .- A*x)
    corr .= 0.0

    if level == length(mgrid)
        corr .= A\res
    else
        corr .= vcycle(mgrid, corr, res, level + 1, tol) 
    end

    # println("Post-level $level")


    dcorr = mgrid[level].P*corr
    x .+= dcorr

    Jacobi_solver!(x, A, b, rDA, f_iter, 1e-10)
    x
end

function AMG!(mgrid, x, b, tol)
    iterations = 100
    residual = 0.0
    for i ∈ 1:iterations
        vcycle(mgrid, x, b, 1, tol)
        residual = norm(b .- A*x)
        if residual <= tol
            println("Converged! Residual: $residual ($i iterations): ", residual)
            return
        end
    end
    println("Not converged! Residual: $residual")
end

tol = 10
T.values .= 100
@time AMG!(mgrid, T.values, b, tol)
write_vtk("result_AMG", mesh, ("T", T))

T.values .= 100.0
@time Jacobi_solver!(T.values, A, b, inv(Diagonal(A)), 20000, 0.1, verbose=true)
write_vtk("result_Jacobi", mesh, ("T", T))

T.values .= 100.0
@time T.values .= AlgebraicMultigrid.solve(
    A, b, SmoothedAggregationAMG(), maxiter = 20, abstol = 1e-6)
write_vtk("result_RugeStubenAMG", mesh, ("T", T))

size(b)
t = zeros(10000)
b1 = reshape(t, size(b))

colour = 0
for (i, cluster) ∈ enumerate(agglomeration)
    colour = i
    for ID ∈ cluster
        T.values[ID] = colour
    end
end


write_vtk("coloring", mesh, ("T", T))

lrows