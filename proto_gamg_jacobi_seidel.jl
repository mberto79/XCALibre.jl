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

T.values .= A\b

r = sum(b - A*T.values)

write_vtk("result", mesh, ("T", T))

rD = inv(Diagonal(A))

x = similar(T.values)
x .= 100.0

# Jacobi solver
iterations = 100
for i ∈ 1:iterations
    @time T.values .= rD*b + (I - rD*A)*x
    x .= T.values
    write_vtk("result_$i", mesh, ("T", T))
end

U = UpperTriangular(A)
D = Diagonal(A)
L = LowerTriangular(A)