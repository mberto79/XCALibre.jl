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

function plot(p::Point)
    scatter([p.coords[1]], [p.coords[2]], color=:blue, legend=false)
end

function plot!(fig, p::Point)
    scatter!(fig, [p.coords[1]], [p.coords[2]], color=:blue, legend=false)
end

function plot(vec::Vector{Point})
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
p4 = Point(10.0,0.0,0.0)

points = Point[p1,p2,p3,p4]
fig = plot(points)

d = p2.coords - p1.coords
d_mag = norm(d)
e1 = d/d_mag

# Allocate array for all points
xcells = 5
ycells = 50
points = fill(Point(0.0,0.0,0.0), (xcells + 1)*(ycells + 1))
point_index = 1
function discretise_edge(
    points::Vector{Point{F}}, point_index::Integer, e::Edge{F}, ncells::Integer
    ) where F<:AbstractFloat
    p1 = e.p1; p2 = e.p2; nsegments = ncells - 1
    d = p2.coords - p1.coords
    d_mag = norm(d)
    e1 = d/d_mag
    spacing = d_mag/ncells
    # points = fill(Point(0.0,0.0,0.0), (nsegments+2))
    j = 1
    for pointi ∈ (point_index+1):(point_index + ncells)
        points[pointi] = Point(spacing*e1*(j) + p1.coords)
        j += 1
        point_index = pointi + 1
    end
    points[point_index] = p1
    points[point_index + nsegments + 1] = p2
    return point_index
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