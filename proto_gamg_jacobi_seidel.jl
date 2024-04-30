using FVM_1D
using LinearAlgebra
using SparseArrays

# using FVM_1D.Mesh
# using FVM_1D.Fields
using FVM_1D.ModelFramework
using FVM_1D.Discretise
using FVM_1D.Solve
using FVM_1D.Calculate
using FVM_1D.RANSModels
using FVM_1D.VTK

mesh_file = "unv_sample_meshes/quad.unv"
mesh_file = "unv_sample_meshes/quad40.unv"
mesh_file = "unv_sample_meshes/quad100.unv"
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
        Dirichlet(:outlet, 100),
        Dirichlet(:bottom, 100),
        Dirichlet(:top, 100)
)

runtime = set_runtime(iterations=1, write_interval=1, time_step=1)
prev = T.values 
discretise!(T_eqn, prev, runtime)
apply_boundary_conditions!(T_eqn, T.BCs)

(; A, b) = T_eqn.equation

@time T.values .= A\b

r = sum(b - A*T.values)

write_vtk("direct_result", mesh, ("T", T))

rD = inv(Diagonal(A))

x = similar(T.values)
x .= 100.0

# Jacobi solver
function Jacobi_solver!(res, A, b, itmax, tol)
    rD = inv(Diagonal(A))
    x0 = similar(res)
    x0 .= res
    r = 0.0
    for i ∈ 1:itmax
        res .= rD*b + (I - rD*A)*x0
        x0 .= res
        r = abs(sum(b - A*res))
        if r <= tol
            # println("Converged! Residual ($i iterations): ", r)
            return
        end
    end
    # println("Residual ($itmax iterations): ", r)
end

T.values .= 100
Jacobi_solver!(T.values, A, b, 10000, 1e-5)
write_vtk("result_Jacobi", mesh, ("T", T))

U = UpperTriangular(A)
D = Diagonal(A)
L = LowerTriangular(A)

rowsA = rowvals(A)
rowsL = rowvals(L)

col1 = nzrange(A,2)
col2 = nzrange(L,99)


lrows = Int64[]
for j ∈ 1:size(L)[1]
    for i ∈ nzrange(L,j)
        push!(lrows, rowsL[i])
    end
end

cell_used = zeros(Int64, length(mesh.cells))
agglomeration = Vector{Int64}[]

cellsID = Int64[]
for j ∈ 1:size(A)[1]
    diag_cell= rowsA[nzrange(L,j)[1]]
    if cell_used[diag_cell] == 0
        cellsID = Int64[]
        for i ∈ nzrange(L,j)
            cID = rowsL[i]
            if cell_used[cID] == 0
                push!(cellsID, cID)
                cell_used[cID] = 1
            end
        end
    push!(agglomeration, cellsID)
    end 
end

function restriction(A)
    rowsA = rowvals(A)
    cell_used = zeros(Int64, size(A)[1])
    agglomeration = Vector{Int64}[]
    cellsID = Int64[]
    i_vals = Int64[]
    j_vals = Int64[]
    v_vals = Int64[]
    i_counter = 0
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
        i_counter += 1
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
                push!(i_vals,i_counter)
                push!(j_vals,cID)
                push!(v_vals,1)
                cell_used[cID] = 1
            end
        end
        push!(agglomeration, cellsID)
    end

    return sparse(i_vals, j_vals, v_vals)
end

R = restriction(A)
Rt = transpose(R)

R*b
tol = 1e-6

T.values .= 100
# function AMG!(res, A, b, tol)
    R1 = restriction(A)
    Rt1 = transpose(R)

    A_L1 = R1*A*Rt1
    
    R2 = restriction(A_L1)
    Rt2 = transpose(R2)

    A_L2 = R2*A_L1*Rt2
    
    r = similar(b)
    r_L1 = zeros(size(R1)[1])
    r_L2 = zeros(size(R2)[1])

    dx = zeros(length(b))
    dx_L1 = zeros(length(r_L1))
    dx_L2 = zeros(length(r_L2))

    # for i ∈ 1:1000
    res = T.values
    Jacobi_solver!(res, A, b, 5, tol)
    r .= b - A*res
    residual = abs(sum(r))
    println("Iteration $i, residual: ", residual)
    r_L1 .= R1*(r)
    R1
    # dx_L1 = zeros(length(r_L1))
    dx_L1 .= 0.0
    
    Jacobi_solver!(dx_L1, A_L1, r_L1, 5, tol)

    r_L2 .= R2*(r_L1)
    # dx_L2 = zeros(length(r_L2))
    dx_L2 .= R2*dx_L1
    
    Jacobi_solver!(dx_L2, A_L2, r_L2, 100, 1e-1)

    # dx_L1 .= dx_L1 + Rt2*dx_L2
    dx_L1 .= Rt2*dx_L2
    Jacobi_solver!(dx_L1, A_L1, r_L1, 5, tol)

    dx .= Rt1*dx_L1
    Jacobi_solver!(dx, A, r, 100, 0.001)

    res .= res .+ dx
        if residual < tol
            break
            return res
        end
    # Jacobi_solver!(res, A, b, 10000, tol)
    # end
    # return res
# end

T.values .= 0.0
@time AMG!(T.values, A, b, 1e-6)
write_vtk("result_AMG", mesh, ("T", T))

T.values .= 100
@time Jacobi_solver!(T.values, A, b, 10000, 1e-6)
write_vtk("result_Jacobi", mesh, ("T", T))

colour = 0
for (i, cluster) ∈ enumerate(agglomeration)
    colour = i
    for ID ∈ cluster
        T.values[ID] = colour
    end
end


write_vtk("coloring", mesh, ("T", T))

lrows