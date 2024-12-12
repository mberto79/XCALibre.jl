using Plots
using XCALibre

# quad and trig 40 and 100
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "trig100.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU()

phi = ScalarField(mesh)
phif = FaceScalarField(mesh)
gradScheme = Orthogonal
# gradScheme = Midpoint
∇phi = Grad{gradScheme}(phi)

phi = assign(
    phi, 
    Dirichlet(:inlet, 0.0),
    Dirichlet(:outlet, 2.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0),
    )


solvers=nothing
schemes=nothing
runtime=nothing
hardware = set_hardware(backend=backend, workgroup=1024)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

for (cID, cell) ∈ enumerate(mesh.cells)
    cx = cell.centre[1]
    phi.values[cID] = 2*cx
end


grad!(∇phi, phif, phi, phi.BCs, 0.0, config) 

meshData = VTKWriter2D(nothing, nothing)
write_vtk("output", mesh, meshData, ("phi", phi), ("gradPhi", ∇phi.result))
