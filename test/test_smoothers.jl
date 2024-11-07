using SparseMatricesCSR

# JacobiSmoother: constructors 

# Direct constructor
n = 10
niters = 5
omega = 1
s = JacobiSmoother(niters, omega, zeros(n))

@test s.loops == niters
@test s.omega == omega 
@test length(s.x_temp) == n

# Construct with mesh object only
test_grids_dir = pkgdir(XCALibre, "test", "grids")
meshFile = joinpath(test_grids_dir, "trig40.unv")
mesh = UNV2D_mesh(meshFile, scale=0.001)
s = JacobiSmoother(mesh)

@test s.loops == 5 # default loops
@test s.omega == 1 # default relaxation factor 
@test length(s.x_temp) == length(mesh.cells)

# Keyword constructor 
s = JacobiSmoother(domain=mesh, loops=niters)
@test s.loops == niters # default loops
@test s.omega ≈ 2/3 # default relaxation factor 
@test length(s.x_temp) == length(mesh.cells)

# JacobiSmoother: solver
i = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5]
j = [1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5]
v = [300, -100, -100, 200, -100, -100, 200, -100, -100, 200, -100, -100, 300]
b = [200*500, 0, 0, 0, 200*100]

A_csr = sparsecsr(i, j, v)
A_check = Array(A_csr)
x_check = A_check\b

config = (;hardware = (;backend=CPU(), workgroup=5))
s = JacobiSmoother(50, 1, zeros(5))
x_test = zeros(5)
XCALibre.Solve.apply_smoother!(s, x_test, A_csr, b, config)
x_test
@test x_check ≈ x_test atol=1e-1