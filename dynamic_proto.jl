using Plots
using StaticArrays
using LinearAlgebra

using FVM_1D.Mesh2D
using FVM_1D.Plotting

n_vertical      = 2
n_horizontal1   = 5
n_horizontal2   = 3

# n_vertical      = 200
# n_horizontal1   = 1000
# n_horizontal2   = 8000

p1 = Point(0.0,0.0,0.0)
p2 = Point(1.0,0.0,0.0)
p3 = Point(1.5,0.0,0.0)
p4 = Point(0.0,1.0,0.0)
p5 = Point(0.8,0.8,0.0)
p6 = Point(1.5,0.7,0.0)
points = [p1,p2,p3,p4,p5,p6]

# Edges in x-direction
e1 = Edge(1,2,n_horizontal1)
e2 = Edge(2,3,n_horizontal2)
e3 = Edge(4,5,n_horizontal1)
e4 = Edge(5,6,n_horizontal2)

# Edges in y-direction
e5 = Edge(1,4,n_vertical)
e6 = Edge(2,5,n_vertical)
e7 = Edge(3,6,n_vertical)
edges = [e1,e2,e3,e4,e5,e6,e7]

b1 = Block([1,3,5,6],n_horizontal1,n_vertical)
b2 = Block([2,4,6,7],n_horizontal2,n_vertical)
blocks = [b1,b2]

patch1 = Patch(:inlet, 1, [5])
patch2 = Patch(:outlet, 2, [7])
patch3 = Patch(:bottom, 3, [1,2])
patch4 = Patch(:top, 4, [3,4])
patches = [patch1, patch2, patch3, patch4]

MeshDefinition(points, edges, patches, blocks)   

domain = define_mesh(n_vertical, n_horizontal1, n_horizontal2)
tag_boundaries!(domain)
multiblock = build_multiblock(domain)
counter = generate_boundary_nodes!(multiblock)
counter = generate_internal_edge_nodes!(multiblock, counter)
generate_internal_nodes!(multiblock, counter)
build_elements!(multiblock)

multiblock = nothing
@time multiblock = generate(n_vertical, n_horizontal1, n_horizontal2)
c = centres(multiblock.elements)

fig = plot(multiblock.nodes)
plot!(fig, c; colour=:red)
plot!(fig, multiblock.definition.points, colour=:red)

function linear(edge::Edge{I}) where I
    for pointID ∈ edge.pointsID
        pointID*2
    end
end

@time linear(e1)