export process!, discretise_edge!, collect_elements!, centre!

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