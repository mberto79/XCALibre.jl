module Plotting

using FVM_1D.Mesh2D
using RecipesBase
export plotRecipe

@recipe function plotRecipe(point::Node{F}) where F
    xlabel --> "x [m]"
    ylabel --> "y [m]"
    legend --> false
    [point.coords[1]], [point.coords[2]]
end

x(n::Node{F}) where F = n.coords[1]
y(n::Node{F}) where F = n.coords[2]
z(n::Node{F}) where F = n.coords[3]

@recipe function plotRecipe(points::Vector{Node{F}}) where F
    xlabel --> "x [m]"
    ylabel --> "y [m]"
    legend --> false
    x.(points), y.(points)
end

x(f) = f.centre[1]
y(f) = f.centre[2]
z(f) = f.centre[3]


@recipe function plotRecipe(f::Face2D{I,F}) where {I,F}
    xlabel --> "x [m]"
    ylabel --> "y [m]"
    legend --> false
    [x(f)], [y(f)]
end

# @recipe function plotRecipe(f::Vector{Face2D{I,F}}) where {I,F}
    @recipe function plotRecipe(f::Vector{Face2D{I,F}}) where {I,F}
    xlabel --> "x [m]"
    ylabel --> "y [m]"
    legend --> false
    x.(f), y.(f)
end

@recipe function plotRecipe(f::Cell{I,F}) where {I,F}
    xlabel --> "x [m]"
    ylabel --> "y [m]"
    legend --> false
    [x(f)], [y(f)]
end

# @recipe function plotRecipe(f::Vector{Face2D{I,F}}) where {I,F}
    @recipe function plotRecipe(f::Vector{Cell{I,F}}) where {I,F}
    xlabel --> "x [m]"
    ylabel --> "y [m]"
    legend --> false
    x.(f), y.(f)
end

end