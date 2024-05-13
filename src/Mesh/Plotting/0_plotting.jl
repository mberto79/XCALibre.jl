# module Plotting

# using FVM_1D.Mesh
# using RecipesBase
# export plotRecipe, plot_mesh

# @recipe function plotRecipe(point::Node{F}) where F
#     xlabel --> "x [m]"
#     ylabel --> "y [m]"
#     legend --> false
#     [point.coords[1]], [point.coords[2]]
# end

# @recipe function plotRecipe(points::Vector{Node{F}}) where F
#     xlabel --> "x [m]"
#     ylabel --> "y [m]"
#     legend --> false
#     x.(points), y.(points)
# end

# x(f) = f.centre[1]
# y(f) = f.centre[2]
# z(f) = f.centre[3]

# x(n::Node{F}) where F = n.coords[1]
# y(n::Node{F}) where F = n.coords[2]
# z(n::Node{F}) where F = n.coords[3]


# # @recipe function plotRecipe(f::Face2D{I,F}) where {I,F}
# #     xlabel --> "x [m]"
# #     ylabel --> "y [m]"
# #     legend --> false
# #     [x(f)], [y(f)]
# # end

# @recipe function plotRecipe(nodes::Vector{Node{F}}, faces::Vector{Face2D{I,F}}) where {I,F}
#     xlabel --> "x [m]"
#     ylabel --> "y [m]"
#     legend --> false
#     x_vec = zeros(F, length(faces)+1)
#     y_vec = zeros(F, length(faces)+1)
#     for i ∈ eachindex(faces)
#         nodesID = faces[i].nodesID
#         n1 = nodes[nodesID[1]]
#         n2 = nodes[nodesID[2]]
#         x_vec[i]    = x(n1)
#         y_vec[i]    = y(n1)
#         x_vec[i+1]  = x(n2)
#         y_vec[i+1]  = y(n2)
#     end
#     # [p1[1], p2[1]], [p1[2], p2[2]]
#     x_vec, y_vec
# end

# # @recipe function plotRecipe(f::Vector{Face2D{I,F}}) where {I,F}
#     @recipe function plotRecipe(f::Vector{Face2D{I,F}}) where {I,F}
#     xlabel --> "x [m]"
#     ylabel --> "y [m]"
#     legend --> false
#     x.(f), y.(f)
# end

# @recipe function plotRecipe(f::Cell{I,F}) where {I,F}
#     xlabel --> "x [m]"
#     ylabel --> "y [m]"
#     legend --> false
#     [x(f)], [y(f)]
# end

# # @recipe function plotRecipe(f::Vector{Face2D{I,F}}) where {I,F}
# @recipe function plotRecipe(f::Vector{Cell{I,F}}) where {I,F}
#     xlabel --> "x [m]"
#     ylabel --> "y [m]"
#     legend --> false
#     x.(f), y.(f)
# end

# @userplot plot_mesh
# @recipe function plotRecipe(m::plot_mesh)
#     (; nodes, faces) = m.args[1]
#     for face ∈ faces
#         nodesID = face.nodesID
#         p1 = nodes[nodesID[1]].coords
#         p2 = nodes[nodesID[2]].coords
#         x = [p1[1], p2[1]]
#         y = [p1[2], p2[2]]
#         @series begin
#             seriestype := :line
#             color := :blue
#             legend := false
#             x, y
#         end
#     end
# end

# end # end module