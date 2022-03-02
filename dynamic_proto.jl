using Plots
using StaticArrays
using LinearAlgebra

struct Point{F<:AbstractFloat}
    coords::SVector{3,F}
end
Point(x::F, y::F, z::F) where F<:AbstractFloat = Point(SVector{3, F}(x,y,z))

struct Edge{F<:AbstractFloat}
    p1::Point{F}
    p2::Point{F}
end

struct Element{I<:Integer,F<:AbstractFloat}
    nodesID::SVector{4,I}
    centre::SVector{3,F}
end

struct Block{I<:Integer,F<:AbstractFloat}
    p1::Point{F}
    p2::Point{F} 
    p3::Point{F} 
    p4::Point{F} 
    nx::I
    ny::I
    nodes::Vector{Point{F}}
    elements::Vector{Element{I,F}}
end
Block(p1::Point{F},p2::Point{F},p3::Point{F},p4::Point{F},nx::I,ny::I) where {I,F} = begin
    Block(
        p1, p2, p3, p4, nx, ny, 
        fill(Point(0.0, 0.0, 0.0), (nx+1)*(ny+1)),
        fill(Element(SVector{4, I}(0,0,0,0), SVector{3, F}(0.0,0.0,0.0)), (nx)*(ny)))
end

function plot(p::Point; colour=:blue)
    scatter([p.coords[1]], [p.coords[2]], color=colour, legend=false)
end

function plot!(fig, p::Point; colour=:blue)
    scatter!(fig, [p.coords[1]], [p.coords[2]], color=colour, legend=false)
end

function plot(vec::Vector{Point{F}}; colour=:blue) where F<:AbstractFloat
    fig = scatter(
        [vec[1].coords[1]], [vec[1].coords[2]], 
        color=colour, legend=false, markersize=3 #, axis_ratio=:equal
        )
    for i ∈ 2:length(vec)
        scatter!(
            fig, [vec[i].coords[1]], [vec[i].coords[2]], 
            color=colour, markersize=3)
    end
    fig
end

function plot!(fig, vec::Vector{Point{F}}; colour=:blue) where F<:AbstractFloat
    for i ∈ 1:length(vec)
        scatter!(fig, [vec[i].coords[1]], [vec[i].coords[2]], color=colour)
    end
    fig
end



function process!(block::Block{I,F}) where {I,F}
    nx = block.nx; ny = block.ny
    points_y1 = fill(Point(0.0,0.0,0.0), ny+1)
    points_y2 = fill(Point(0.0,0.0,0.0), ny+1)
    discretise_edge!(points_y1, 1, block.p1, block.p3, ny)
    discretise_edge!(points_y2, 1, block.p2, block.p4, ny)
    nodei = 1
    for i ∈ eachindex(points_y1)
        nodei_curr = discretise_edge!(block.nodes, nodei, points_y1[i], points_y2[i], nx)
        nodei = nodei_curr
    end
    nothing
end

function discretise_edge!(
    # points::Vector{Point{F}}, point_index::Integer, e::Edge{F}, ncells::Integer
    points::Vector{Point{F}}, node_idx::Integer, p1::Point{F}, p2::Point{F}, ncells::Integer
    ) where F<:AbstractFloat
    nsegments = ncells - 1
    d = p2.coords - p1.coords
    d_mag = norm(d)
    e1 = d/d_mag
    spacing = d_mag/ncells

    points[node_idx] = p1
    points[node_idx + nsegments + 1] = p2

    j = 1
    for pointi ∈ (node_idx+1):(node_idx + ncells)
        points[pointi] = Point(spacing*e1*(j) + p1.coords)
        j += 1
        node_idx = pointi + 1
    end
    return node_idx
end

function collect_elements!(block::Block)
    element_i = 0
    node_i = 0
    for y_i ∈ 1:block.ny
        for x_i ∈ 1:block.nx
            element_i += 1
            node_i += 1
            block.elements[element_i] = Element(
                SVector{4, Int64}(node_i,node_i+1,node_i+(block.nx+1),node_i+(block.nx+1)+1), 
                SVector{3, Float64}(0.0,0.0,0.0)
            )
            
        end
        node_i += 1
    end
end

function centre!(block::Block{I,F}) where {I,F}
    for i ∈ eachindex(block.elements)
        sum =  SVector{3, F}(0.0,0.0,0.0)
        for id ∈ block.elements[i].nodesID
            sum += block.nodes[id].coords
        end
        centre = sum/4
        block.elements[i] = Element(block.elements[i].nodesID, centre)
    end
end

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

Base.:(==)(e1::Edge{F}, e2::Edge{F}) where F = begin
    a = 2
end