# export process!, discretise_edge!, collect_elements!, centre!
export build_multiblock, tag_boundaries!, generate_boundary_nodes!, generate_internal_edge_nodes!, find_blocks_with_same_edge, generate_internal_nodes!

function tag_as_boundary!(edge::Edge{I}) where I
    Edge(edge.p1, edge.p2, edge.n, true, edge.nodesID)
end

function tag_as_boundary!(point::Point{F}) where F
    Point(point.coords, true, false, 0)
end

function tag_boundaries!(domain::MeshDefinition{I,F}) where {I,F}
    patches = domain.patches; edges = domain.edges; points = domain.points
    for patch ∈ patches
        for id ∈ patch.edgesID
            edges[id] = tag_as_boundary!(edges[id])
            points[edges[id].p1] = tag_as_boundary!(points[edges[id].p1])
            points[edges[id].p2] = tag_as_boundary!(points[edges[id].p2])
        end
    end
end

function build_multiblock(domain::MeshDefinition{I,F}) where {I,F}
    # Calculate total number of cells and nodes
    n_cells = zero(I)
    n_nodes = zero(I)
    for block ∈ domain.blocks
        n_cells += block.nx*block.ny
        n_nodes += (block.nx + 1)*(block.ny + 1)
    end
    for edge ∈ domain.edges
        if !edge.boundary
            n_nodes -= (edge.n + 1)
        end
    end
    elements = fill(Element(zero(I), zero(F)), n_cells)
    nodes = fill(Node(zero(F)), n_nodes)
    return MultiBlock(domain, elements, nodes)
end

function linear_distribution(p1, p2, ncells, j)
    d = p2.coords - p1.coords
    d_mag = norm(d)
    e1 = d/d_mag
    spacing = d_mag/ncells
    return spacing*e1*j + p1.coords
end

function nodes_on_boundary_edge!(
    multiblock::MultiBlock{I,F}, edgeID::I, counter::I) where {I,F}
    edge = multiblock.definition.edges[edgeID]
    ncells = edge.n
    p1 = multiblock.definition.points[edge.p1]
    p2 = multiblock.definition.points[edge.p2]
    for j ∈ 2:(ncells)
        multiblock.nodes[counter] = Node(linear_distribution(p1, p2, ncells, j-1))
        nodes_to_multiple_block_matrix!(multiblock, edgeID, counter, j)
        counter += 1
    end
    return counter
end

function nodes_to_multiple_block_matrix!(
    multiblock::MultiBlock{I,F}, edgeID::I, counter::I, node_index::I) where {I,F}
    for block ∈ multiblock.definition.blocks
        rows, cols = size(block.nodesID)
        if block.edge_x1 == edgeID
            block.nodesID[node_index,1] = counter
            # return
        end
        if block.edge_x2 == edgeID
            block.nodesID[node_index,cols] = counter
            # return
        end
        if block.edge_y1 == edgeID
            block.nodesID[1,node_index] = counter
            # return
        end
        if block.edge_y2 == edgeID
            block.nodesID[rows,node_index] = counter
            # return
        end
    end
end

function nodes_to_single_block_matrix!(
    multiblock::MultiBlock{I,F}, blockID::I, edgeID::I, counter::I, node_index::I
    ) where {I,F}
    block = multiblock.definition.blocks[blockID]

    rows, cols = size(block.nodesID)
    if block.edge_x1 == edgeID
        block.nodesID[node_index,1] = counter
        # return
    end
    if block.edge_x2 == edgeID
        block.nodesID[node_index,cols] = counter
        # return
    end
    if block.edge_y1 == edgeID
        block.nodesID[1,node_index] = counter
        # return
    end
    if block.edge_y2 == edgeID
        block.nodesID[rows,node_index] = counter
        # return
    end
end

function generate_boundary_nodes!(multiblock::MultiBlock{I,F}) where {I,F}
    counter = 1
    for patch ∈ multiblock.definition.patches
        for edgeID ∈ patch.edgesID
            edge = multiblock.definition.edges[edgeID]
            p1 = multiblock.definition.points[edge.p1]
            p2 = multiblock.definition.points[edge.p2]
            if p1.processed
                nodes_to_multiple_block_matrix!(multiblock, edgeID, p1.ID, 1) 
            end
            if p2.processed
                nodes_to_multiple_block_matrix!(multiblock, edgeID, p2.ID, edge.n + 1) 
            end
            if !p1.processed
                multiblock.nodes[counter] = Node(p1.coords)
                multiblock.definition.points[edge.p1] = Point(
                    p1.coords, true, true, counter)
                nodes_to_multiple_block_matrix!(multiblock, edgeID, counter, 1) 
                counter += 1
            end
            if !p2.processed
                multiblock.nodes[counter] = Node(p2.coords)
                multiblock.definition.points[edge.p2] = Point(
                    p2.coords, true, true, counter)
                nodes_to_multiple_block_matrix!(multiblock, edgeID, counter, edge.n + 1) 
                counter += 1
            end
        end
    end
    for patch ∈ multiblock.definition.patches
        for edgeID ∈ patch.edgesID
            counter = nodes_on_boundary_edge!(multiblock, edgeID, counter)
        end
    end
    return counter
end

function generate_internal_edge_nodes!(multiblock::MultiBlock{I,F}, counter::I) where {I,F}
    for edgeID ∈ eachindex(multiblock.definition.edges)
        if !multiblock.definition.edges[edgeID].boundary
            counter = nodes_on_internal_edge!(multiblock, edgeID, counter)
        end
    end
    counter
end

function nodes_on_internal_edge!(multiblock, edgeID, counter)
    edge = multiblock.definition.edges[edgeID]
    ncells = edge.n
    p1 = multiblock.definition.points[edge.p1]
    p2 = multiblock.definition.points[edge.p2]
    for j ∈ 2:(ncells)
        multiblock.nodes[counter] = Node(linear_distribution(p1, p2, ncells, j-1))
        nodes_to_multiple_block_matrix!(multiblock, edgeID, counter, j)
        counter += 1
    end
    return counter
end

function generate_internal_nodes!(multiblock::MultiBlock{I,F}, counter::I) where {I,F}
    for block ∈ multiblock.definition.blocks
        ncells = block.nx
        y1_idx = @view block.nodesID[1,:]
        y2_idx = @view block.nodesID[ncells+1,:]
        
        for i ∈ 2:(length(y1_idx) - 1)
            p1 = multiblock.nodes[y1_idx[i]]
            p2 = multiblock.nodes[y2_idx[i]]
            for j ∈ 2:(ncells)
                multiblock.nodes[counter] = Node(linear_distribution(p1, p2, ncells, j-1))
                # nodes_to_multiple_block_matrix!(multiblock, edgeID, counter, j)
                counter += 1
            end
        end
    end # block loop
end

