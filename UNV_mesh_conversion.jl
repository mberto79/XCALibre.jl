using Plots
using StaticArrays
using BenchmarkTools

using FVM_1D

using FVM_1D.Mesh2D
using FVM_1D.UNV
using FVM_1D.Plotting
using FVM_1D.Discretise
using FVM_1D.Calculate
using FVM_1D.Models
using FVM_1D.Solvers
using FVM_1D.VTK

using Krylov


# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/backwardFacingStep_2mm.unv"
points, elements, boundaryElements = load(mesh_file, Int64, Float64);
mesh = build_mesh(mesh_file, scale=0.001)

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

UBCs = ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    # Dirichlet(:bottom, [0.0, 0.0, 0.0]),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
    # Neumann(:top, 0.0)
)

uxBCs = (
    Dirichlet(:inlet, velocity[1]),
    Neumann(:outlet, 0.0),
    # Dirichlet(:bottom, 0.0),
    Dirichlet(:wall, 0.0),
    Dirichlet(:top, 0.0)
    # Neumann(:top, 0.0)
)

uyBCs = (
    Dirichlet(:inlet, velocity[2]),
    Neumann(:outlet, 0.0),
    # Dirichlet(:bottom, 0.0),
    Dirichlet(:wall, 0.0),
    Dirichlet(:top, 0.0)
    # Neumann(:top, 0.0)
)

pBCs = (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    # Neumann(:bottom, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

setup_U = SolverSetup(
    iterations  = 1,
    solver      = BicgstabSolver,
    tolerance   = 1e-1,
    relax       = 0.8,
    itmax       = 100,
    rtol        = 1e-1
)

setup_p = SolverSetup(
    iterations  = 1,
    solver      = GmresSolver, #CgSolver, #GmresSolver, #BicgstabSolver,
    tolerance   = 1e-1,
    relax       = 0.2,
    itmax       = 100,
    rtol        = 1e-2
)

GC.gc()

ux = ScalarField(mesh)
uy = ScalarField(mesh)
p = ScalarField(mesh)
U = VectorField(mesh)

iterations = 1000
Rx, U = isimple!(
    mesh, velocity, nu, ux, uy, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    setup_U, setup_p, iterations)

write_vtk("results", mesh, ("U", U), ("p", p))

# plotly(size=(400,400), markersize=1, markerstrokewidth=1)
niterations = length(Rx)
plot(collect(1:niterations), Rx[1:niterations], yscale=:log10)

plot(aspect_ratio=:equal, xlim=(0.0,0.25), ylim=(-0.1, 0.1))
plot_mesh!(mesh)
scatter!(mesh.nodes, aspect_ratio=:equal)

fig = plot(aspect_ratio=:equal, color=:blue, legend=false)
for fID ∈ 1:length(faces)
    p1 = nodes[faces[fID].nodesID[1]].coords
    p2 = nodes[faces[fID].nodesID[2]].coords
    x = [p1[1], p2[1]]
    y = [p1[2], p2[2]]
    plot!(fig, x,y, color=:blue)
end
@show fig