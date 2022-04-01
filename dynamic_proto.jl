using Plots

using FVM_1D.Mesh2D
using FVM_1D.Plotting

n_vertical      = 10
n_horizontal1   = 15
n_horizontal2   = 12

p1 = Point(0.0,0.0,0.0)
p2 = Point(1.0,0.0,0.0)
p3 = Point(1.5,0.0,0.0)
p4 = Point(0.0,1.0,0.0)
# p5 = Point(0.8,0.8,0.0)
p5 = Point(1.0,1.0,0.0)
p6 = Point(1.5,0.7,0.0)
points = [p1,p2,p3,p4,p5,p6]

# Edges in x-direction
e1 = line!(points,1,2,n_horizontal1)
e2 = line!(points,2,3,n_horizontal2)
e3 = line!(points,4,5,n_horizontal1)
e4 = line!(points,5,6,n_horizontal2)

# Edges in y-direction
e5 = line!(points,1,4,n_vertical)
e6 = line!(points,2,5,n_vertical)
e7 = line!(points,3,6,n_vertical)
edges = [e1,e2,e3,e4,e5,e6,e7]

b1 = quad(edges, [1,3,5,6])
b2 = quad(edges, [2,4,6,7])
blocks = [b1,b2]

patch1 = Patch(:inlet,  [5])
patch2 = Patch(:outlet, [7])
patch3 = Patch(:bottom, [1,2])
patch4 = Patch(:top,    [3,4])
patches = [patch1, patch2, patch3, patch4]

builder = MeshBuilder2D(points, edges, patches, blocks)
mesh = generate!(builder)

using FVM_1D.Discretise
using FVM_1D.Models

phiBCs = (
    (dirichlet, :inlet, 100),
    (dirichlet, :outlet, 50),
    (dirichlet, :bottom, 100),
    (dirichlet, :top, 100)
)

phi = ScalarField(mesh)
equation = Equation(mesh)

J = 1.0
phiModel = SteadyDiffusion(Laplacian{Linear}(J, phi), 0.0)
phiModel.terms.term1.sign[1] = 1
generate_boundary_conditions!(mesh, phiModel, phiBCs)

discretise!(equation, phiModel, mesh)
update_boundaries!(equation, mesh, phiModel, phiBCs)
phi.values .= equation.A\equation.b

x(mesh) = [mesh.cells[i].centre[1] for i ∈ 1:length(mesh.cells)]
y(mesh) = [mesh.cells[i].centre[2] for i ∈ 1:length(mesh.cells)]
plotly()
scatter(x(mesh), y(mesh), phi.values, color=:red)

scatter(mesh.nodes, colour=:black)
scatter!(centre2d.(mesh.faces), color=:blue)
scatter!(centre2d.(mesh.cells), color=:red)
fig = plot_mesh!(mesh)

# for boundary ∈ mesh.boundaries
#     for ID ∈ boundary.facesID
#         face = mesh.faces[ID]
#         normal = 0.1*face.normal
#         centre = centre2d(face)
#         fig = quiver!(fig, centre..., quiver=([normal[1]], [normal[2]]), color=:green)
#     end
# end
# @show fig
for face ∈ mesh.faces
    normal = 0.1*face.normal
    centre = centre2d(face)
    fig = quiver!(fig, centre..., quiver=([normal[1]], [normal[2]]), color=:green)
end
@show fig

# for (nodei, node) ∈ enumerate(mesh.nodes)
#     centre = [node.coords[1]], [node.coords[2]]
#     fig = annotate!(fig, centre[1],centre[2], text("$(nodei)", :black, 8))
# end
# for (facei, face) ∈ enumerate(mesh.faces)
#     centre = centre2d(face)
#     fig = annotate!(fig, centre[1],centre[2], text("$(facei)", :blue, 10))
# end

for (celli, cell) ∈ enumerate(mesh.cells)
    centre = centre2d(cell)
    fig = annotate!(fig, centre[1],centre[2], text("$(celli)", :red, 10))
end
@show fig
