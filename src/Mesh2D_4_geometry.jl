export geometric_centre

function geometric_centre(nodes::Vector{Node{F}}, nodeList::SVector{N, I}) where {I,F,N}
    sum = SVector{3, F}(0.0,0.0,0.0)
        for ID ∈ nodeList
            sum += nodes[ID].coords
        end
    return sum/(length(nodeList))
end