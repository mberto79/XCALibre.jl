using XCALibre
using LinearAlgebra
using Plots

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
# trig, trig40, trig100, quad, quad40, quad100
grid = "trig40.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

phi = ScalarField(mesh)

cell_xyz = getproperty.(mesh.cells, :centre)

phi.values

sin.(cell_xyz.⋅Ref([1,0,0])*π)

phi.values .= sin.(cell_xyz.⋅Ref([1,0,0])*π)

scatter(cell_xyz.⋅Ref([1,0,0]), phi.values)

phi = assign(phi,
    Dirichlet(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:top, 0.0),
    Neumann(:bottom, 0.0)
) 

gradPhi = Grad{Orthogonal}(phi)
phif = FaceScalarField(mesh)

solvers = (
    phi = set_solver(
        phi;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax = 1.0
    ),
)

schemes = (phi = set_schemes(),)
runtime = set_runtime(iterations=1, write_interval=-1, time_step=1)
hardware = set_hardware(backend=CPU(), workgroup=1024)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

grad!(gradPhi, phif, phi, phi.BCs, 0, config)

scatter(cell_xyz.⋅Ref([1,0,0]), gradPhi.result.x.values)

XCALibre.Solvers.limit_gradient!(gradPhi, phi, config)

scatter(cell_xyz.⋅Ref([1,0,0]), gradPhi.result.x.values)
