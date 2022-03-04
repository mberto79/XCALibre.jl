export build_elements!, centres

function build_elements!(multiblock::MultiBlock{I,F}) where {I,F}
    nodes = multiblock.nodes
    elements = multiblock.elements
    blocks = multiblock.definition.blocks
    elementi = zero(I)
    for block ∈ blocks
        nx, ny = block.nx, block.ny
        nodesID = block.nodesID
        for j ∈ 1:ny
            for i ∈ 1:nx
                elementi += 1
                n1 = nodesID[i,j]
                n2 = nodesID[i+1,j]
                n3 = nodesID[i+1,j+1]
                n4 = nodesID[i,j+1]
                nodeList = SVector{4, I}(n1,n2,n3,n4)
                p1 = nodes[n1].coords
                p2 = nodes[n2].coords
                p3 = nodes[n3].coords
                p4 = nodes[n4].coords
                centre = geometric_centre((p1,p2,p3,p4))
                elements[elementi] = Element(nodeList, centre)
            end
        end
    end
end

function geometric_centre(points::NTuple{N, SVector{3, F}}) where {N,F}
    sum = SVector{3, F}(0.0,0.0,0.0)
        for point ∈ points
            sum += point
        end
    return sum/(length(points))
end

function centres(elements::Vector{Element{I,F}}) where {I,F}
    [Node(elements[i].centre) for i ∈ 1:length(elements)]
end