using Plots
using XCALibre

# quad and trig 40 and 100
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "trig40.unv"
# grid = "quad100.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU()

psi = VectorField(mesh)
psif = FaceVectorField(mesh)
gradScheme = Orthogonal
gradScheme = Midpoint
∇psi = Grad{gradScheme}(psi)

psi = assign(
    psi, 
    Dirichlet(:inlet, [0.0,0.0,0.0]),
    Dirichlet(:outlet, [2.0,0.0,0.0]),
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
    psi.x.values[cID] = 2*cx
end


grad_old!(∇psi, psif, psi, psi.BCs, 0.0, config) 
grad!(∇psi, psif, psi, psi.BCs, 0.0, config) 
# limit_gradient!(∇psi, psif, psi, config)

meshData = VTKWriter2D(nothing, nothing)
write_results("output", mesh, meshData, ("psi", psi), ("gradpsixx", ∇psi.result.xx), ("gradpsixy", ∇psi.result.xy))

nu, ρ = 1.48e-5, 1.225
Ux = 60.0
νR = 10
Tu = 0.025
k_inlet = 3/2*(Tu*Ux)^2
ω_inlet = k_inlet/(νR*nu)
k_inlet/ω_inlet