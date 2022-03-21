using Plots
using StaticArrays
using LinearAlgebra

using FVM_1D.Mesh2D
using FVM_1D.Plotting

n_vertical      = 3
n_horizontal1   = 5
n_horizontal2   = 4

p1 = Point(0.0,0.0,0.0)
p2 = Point(1.0,0.0,0.0)
p3 = Point(1.5,0.0,0.0)
p4 = Point(0.0,1.0,0.0)
p5 = Point(0.8,0.8,0.0)
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

# @time builder = Wireframe(points, edges, patches, blocks)
# @time mesh = preallocate_meshbuilder(builder)
# @time generate_inner_points!(mesh, builder)
# @time generate_elements!(mesh, builder)
# @time counter = generate_boundary_faces!(mesh, builder)
# @time counter = generate_interface_faces!(counter, mesh, builder)
# @time generate_internal_faces!(counter, mesh, builder)

builder = MeshBuilder2D(points, edges, patches, blocks)

function build(builder)
    mesh = preallocate_mesh(builder)
    generate_inner_points!(mesh, builder)
    generate_elements!(mesh, builder)
    counter = generate_boundary_faces!(mesh, builder)
    counter = generate_interface_faces!(counter, mesh, builder)
    generate_internal_faces!(counter, mesh, builder)
    mesh
end

GC.gc()
@time mesh = build(builder)


scatter(mesh.nodes, colour=:blue)
scatter!(mesh.faces, color=:black)
scatter!(mesh.cells, color=:red)
