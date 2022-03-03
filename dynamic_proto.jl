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