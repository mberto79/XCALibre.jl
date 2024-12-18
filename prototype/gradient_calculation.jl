using Plots
using XCALibre

# quad and trig 40 and 100
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "trig.unv"
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
limiter = limit_gradient!(FaceBased(), ∇phi, phi, config)

# phi.values .= limiter

∇phi.result.x.values .*= limiter
∇phi.result.y.values .*= limiter
∇phi.result.z.values .*= limiter

meshData = VTKWriter2D(nothing, nothing)
write_vtk("output", mesh, meshData, ("phi", phi), ("gradPhi", ∇phi.result))

f(type::Nothing, a, b) = a + b

f(nothing,2,3)

using KernelAbstractions

s = fill!(allocate(CPU(), Float64, (20,)), one(Float64))

rk = (1.0/1 - 1.0)