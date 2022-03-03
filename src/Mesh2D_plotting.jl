export plot, plot!

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