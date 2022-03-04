module Plotting

using FVM_1D.Mesh2D
using Plots
export plot, plot!

function plot(p::AbstractPoint; colour=:blue) #where F
    Plots.scatter([p.coords[1]], [p.coords[2]], color=colour, legend=false)
end

function plot!(fig, p::AbstractPoint; colour=:blue) #where F
    Plots.scatter!(fig, [p.coords[1]], [p.coords[2]], color=colour, legend=false)
end

function plot(vec::Vector{<:AbstractPoint}; colour=:blue) #where F
    fig = Plots.scatter(
        [vec[1].coords[1]], [vec[1].coords[2]], 
        color=colour, legend=false, markersize=3 #, axis_ratio=:equal
        )
    for i ∈ 2:length(vec)
        Plots.scatter!(
            fig, [vec[i].coords[1]], [vec[i].coords[2]], 
            color=colour, markersize=3)
    end
    fig
end

function plot!(fig, vec::Vector{<:AbstractPoint}; colour=:blue) #where F
    for i ∈ 1:length(vec)
        Plots.scatter!(fig, [vec[i].coords[1]], [vec[i].coords[2]], color=colour)
    end
    fig
end

# function plot(edge::Edge{I,F}; colour=:blue) where {I,F}
#     Plots.plot(
#         [edge.p1.coords[1], edge.p2.coords[1]],
#         [edge.p1.coords[2], edge.p2.coords[2]], 
#         color=colour, legend=false
#         )
# end

# function plot!(fig, edge::Edge{I,F}; colour=:blue) where {I,F}
#     Plots.plot!(
#         fig,
#         [edge.p1.coords[1], edge.p2.coords[1]],
#         [edge.p1.coords[2], edge.p2.coords[2]], 
#         color=colour, legend=false
#         )
# end

# function plot(edges::Vector{Edge{I,F}}; colour=:blue) where {I,F}
#     fig = plot(edges[1]; colour)
#     for i ∈ 2:length(edges)
#         plot!(fig, edges[i]; colour)
#     end
#     return fig
# end

# function plot!(fig, edges::Vector{Edge{I,F}}; colour=:blue) where {I,F}
#     for edge ∈ edges
#         plot!(fig, edge; colour)
#     end
#     return fig
# end
end