using Plots
using StaticArrays
using LinearAlgebra

using FVM_1D.Mesh2D

p1 = Point(0.0,0.0,0.0)
p2 = Point(1.0,1.0,0.0)
p3 = Point(9.0,1.0,0.0)
p4 = Point(10.0,0.0,0.0)

@time block = Block(p1, p4, p2, p3, 40, 30)
@time process!(block)

@time collect_elements!(block)
@time centre!(block)

block.elements[1]
fig = plot(block.nodes[1])
plot!(fig, block.nodes[1], colour=:red)
plot!(fig, block.nodes[2], colour=:red)
plot!(fig, block.nodes[42], colour=:red)
plot!(fig, block.nodes[43], colour=:red)

fig = plot(block.nodes)

struct Patch0{I}
    name::Symbol
    ID::I
end

struct Boundary{I}
    ID::I
    facesID::I
    nodesID::I
    cellsID::I
end

e1 = Edge(p1,p2)
e2 = Edge(p2,p1)
e3 = Edge(p1,p2)
e4 = Edge(p1,p3)

Base.:(==)(e1::Edge{F}, e2::Edge{F}) where F = begin
    if e1 === e2
        return true
    else
        e_temp = Edge(e2.p2, e2.p1)
        if e1 === e_temp
            return true
        end
    end
    return false
end

@time e1 == e4

@time a = [1 2 3; 4 5 6; 7 8 9]
@time a[1,1] = 11
a
@time b = vec(a)

@time points_y1 = fill(Point(0.0,0.0,0.0), 500, 4000)

using FVM_1D.Mesh2D
using FVM_1D.Plotting

function define_mesh()
p1 = Point(0.0,0.0,0.0)
p2 = Point(1.0,0.0,0.0)
p3 = Point(1.5,0.0,0.0)
p4 = Point(0.0,1.0,0.0)
p5 = Point(0.8,0.8,0.0)
p6 = Point(1.5,0.7,0.0)
points = [p1,p2,p3,p4,p5,p6]

# Edges in x-direction
e1 = Edge(1,2,5)
e2 = Edge(2,3,3)
e3 = Edge(4,5,5)
e4 = Edge(5,6,3)

# Edges in y-direction
e5 = Edge(1,4,2)
e6 = Edge(2,5,2)
e7 = Edge(3,6,2)
edges = [e1,e2,e3,e4,e5,e6,e7]

b1 = Block(1,3,5,6,5,2)
b2 = Block(2,4,6,7,3,2)
blocks = [b1,b2]

patch1 = Patch(:inlet, 1, [5])
patch2 = Patch(:outlet, 2, [7])
patch3 = Patch(:bottom, 3, [1,2])
patch4 = Patch(:top, 4, [3,4])
patches = [patch1, patch2, patch3, patch4]

return MeshDefinition(points, edges, patches, blocks)
end

function generate()
domain = define_mesh()
tag_boundaries!(domain)
multiblock = build_multiblock(domain)
generate_boundary_nodes!(multiblock, 1)
return multiblock
end

@time multiblock = generate()


fig = plot(multiblock.nodes)
plot!(fig, multiblock.definition.points, colour=:red)