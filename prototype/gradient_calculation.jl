using Plots
using XCALibre

# quad and trig 40 and 100
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "trig.unv"
grid = "trig40.unv"
# grid = "quad100.unv"

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
    Dirichlet(:outlet, 0.0),
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
    phi.values[cID] = sin(cx*π)
end


grad!(∇phi, phif, phi, phi.BCs, 0.0, config)
limit_gradient!(FaceBased(mesh), ∇phi, phi, config)
# limit_gradient!(MFaceBased(mesh), ∇phi, phi, config)

meshData = VTKWriter2D(nothing, nothing)
write_results("output", mesh, meshData, ("phi", phi), ("gradPhi", ∇phi.result.x))

f(type::Nothing, a, b) = a + b

f(nothing,2,3)

using KernelAbstractions

s = fill!(allocate(CPU(), Float64, (20,)), one(Float64))

k = 0.95
rk = (1.0/k - 1.0)

# vector tests

psi = VectorField(mesh)
psif = FaceVectorField(mesh)
gradScheme = Orthogonal
# gradScheme = Midpoint
∇psi = Grad{gradScheme}(psi)

psi = assign(
    psi, 
    Dirichlet(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
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
    psi.x.values[cID] = sin(cx*π)
end


grad!(∇psi, psif, psi, psi.BCs, 0.0, config)
# limiter = limit_gradient!(FaceBased(), ∇phi, phi, config)
limit_gradient!(MFaceBased(), ∇psi, psi, config)

meshData = VTKWriter2D(nothing, nothing)
write_results("output", mesh, meshData, ("psi", psi), ("gradPsi", ∇psi.result.xx))