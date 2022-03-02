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

struct Block{I<:Integer,F<:AbstractFloat}
    p1::Point{F}
    p2::Point{F} 
    p3::Point{F} 
    p4::Point{F} 
    nx::I
    ny::I
    nodes::Vector{Point{F}}
end
Block(p1::Point{F},p2::Point{F},p3::Point{F},p4::Point{F},nx::I,ny::I) where {I,F} = begin
    Block(p1, p2, p3, p4, nx, ny, fill(Point(0.0, 0.0, 0.0), (nx+1)*(ny+1)))
end

function plot(p::Point)
    scatter([p.coords[1]], [p.coords[2]], color=:blue, legend=false)
end

function plot!(fig, p::Point)
    scatter!(fig, [p.coords[1]], [p.coords[2]], color=:blue, legend=false)
end

function plot(vec::Vector{Point{F}}) where F<:AbstractFloat
    fig = scatter([vec[1].coords[1]], [vec[1].coords[2]], color=:blue, legend=false)
    for i ∈ 2:length(vec)
        scatter!(fig, [vec[i].coords[1]], [vec[i].coords[2]], color=:blue)
    end
    fig
end

function plot!(fig, vec::Vector{Point{F}}; colour=:blue) where F<:AbstractFloat
    for i ∈ 1:length(vec)
        scatter!(fig, [vec[i].coords[1]], [vec[i].coords[2]], color=colour)
    end
    fig
end

p1 = Point(0.0,0.0,0.0)
p2 = Point(1.0,1.0,0.0)
p3 = Point(9.0,1.0,0.0)
p4 = Point(10.0,0.3,0.0)

@time block = Block(p1, p4, p2, p3, 12, 8)
# Allocate array for all points

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

@time process!(block)
fig = plot(block.nodes)
plot!(fig, pts2)


point_index = 1
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
point_index = 1
e1 = Edge(p1,p2)
@time point_index = discretise_edge(points, point_index, e1, ycells)

e2 = Edge(p4,p3)
@time point_index = discretise_edge(points, point_index, e2, ycells)

e3 = Edge(p2,p3)
@time points3 = discretise_edge(e3,div)

e4 = Edge(p1,p4)
@time points4 = discretise_edge(e4,div)

fig = plot(p1)
fig = plot!(fig , p3)
plot!(fig, points, colour=:red)


struct Patch0{F<:AbstractFloat}
    name::Symbol
    edge::Edge{F}
end

inlet = Patch0(:inlet, Edge(p2,p1))

inlet.edge == e1