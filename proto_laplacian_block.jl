using LinearAlgebra
using StaticArrays
import Plots
import XCALibre
using XCALibre.Plotting

struct Point{F} <:XCALibre.Mesh2D.AbstractPoint
    coords::SVector{2,F}
end

function linear_distribution(p1, p2, ncells, j)
    d = p2.coords - p1.coords
    d_mag = norm(d)
    e1 = d/d_mag
    spacing = d_mag/ncells
    return spacing*e1*j + p1.coords
end

nx, ny = 50, 40

p1 = Point(SVector(0.0, 0.0))
p2 = Point(SVector(1.0, 0.0))
p3 = Point(SVector(0.1, 1.0))
p4 = Point(SVector(0.9, 1.0))
points = [p1,p2,p3,p4]

e1 = [Point(linear_distribution(p1, p2, nx, j)) for j ∈ 0:nx]
e2 = [Point(linear_distribution(p3, p4, nx, j)) for j ∈ 0:nx]
e3 = [Point(linear_distribution(p1, p3, ny, j)) for j ∈ 0:ny]
e4 = [Point(linear_distribution(p2, p4, ny, j)) for j ∈ 0:ny]

function internal_nodes(e3::Vector{Point{F}}, e4::Vector{Point{F}}, nx, ny
    ) where {I,F}
    edges = Vector{Point{F}}[]
    for i ∈ 2:ny
        p1, p2 = e3[i], e4[i]
        push!(edges, [Point(linear_distribution(p1, p2, nx, j)) for j ∈ 1:nx-1])
    end
    edges
end

inner_nodes = internal_nodes(e3,e4,nx,ny)

y_matrix = zeros(nx+1,ny+1)
x_matrix = zeros(nx+1,ny+1)

x(p::Point{F}) where {F} = p.coords[1]
y(p::Point{F}) where {F} = p.coords[2]
curvedEdge(edge::Point{F}) where {F} = Point(SVector{2,F}(edge.coords[1], -0.2*sin(pi*edge.coords[1])))

@time ec = curvedEdge.(e1)
@time y_matrix[:,1] = y.(ec)
y_matrix[:,end] = y.(e2)
y_matrix[1,:] = y.(e3)
y_matrix[end,:] = y.(e4)

x_matrix[:,1] = x.(ec)
x_matrix[:,end] = x.(e2)
x_matrix[1,:] = x.(e3)
x_matrix[end,:] = x.(e4)

function initial_guess!(m::Matrix{F}) where F
    rows, cols = size(m)
    for j ∈ 2:cols-1
        δ = (m[end,j] - m[1,j])/(rows-1)
        for i ∈ 1:rows
            m[i,j] = m[1,j] + δ*(i-1)
        end
    end
end

@time initial_guess!(y_matrix)
@time initial_guess!(x_matrix)

update_nodes!(inner_nodes, x_matrix, :x)
update_nodes!(inner_nodes, y_matrix, :y)

boundaryPoints = [e1; e2; e3; e4]
fig = plot(boundaryPoints)
fig = plot!(fig, ec; colour=:red)
plot([ec; inner_nodes...; e2]; colour=:red)

@time for step_number = 1:500
    y_matrix = step!(y_matrix)
    x_matrix = step!(x_matrix)
end

update_nodes!(inner_nodes, x_matrix, :x)
update_nodes!(inner_nodes, y_matrix, :y)

plot([ec; inner_nodes...; e2; e3; e4]; colour=:red)
Plots.xlims!(0.45,0.55)
Plots.ylims!(0.17,0.21)

function update_nodes!(inner_nodes, y_matrix, axis::Symbol) where {I,F}
    nx, ny = size(y_matrix)
    for i ∈ 1:length(inner_nodes)
        nodes = inner_nodes[i]
        for j ∈ 1:length(nodes)
            if axis == :x
                nodes[j] = Point(SVector(y_matrix[j+1,i+1],nodes[j].coords[2]))
            end
            if axis == :y
                nodes[j] = Point(SVector(nodes[j].coords[1],y_matrix[j+1,i+1]))
            end
        end
    end
end

function step!(f::Matrix{Float64})
    # Create a new matrix for our return value
    height, width = size(f)
    # new_f = zeros(height, width)
      
    # For every cell in the matrix
    for x = 1:width
      for y = 1:height
        if (x == 1 || x == width
          || y == 1 || y == height)
          # If we're on the edge,
          # just copy over the value
        #   new_f[y, x] = f[y, x]
          f[y, x] = f[y, x]
        else
          # For interior cells,
          # replace with the average
        #   new_f[y, x] = (
            f[y, x] = (
            f[y-1, x] + f[y+1, x] +
            f[y, x-1] + f[y, x+1]) / 4.0
        end     
      end
    end
    # return new_f
    return f
end