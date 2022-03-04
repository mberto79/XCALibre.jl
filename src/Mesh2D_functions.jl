# export process!, discretise_edge!, collect_elements!, centre!
export tag_as_boundary!, tag_boundaries!, build_multiblock
export generate_boundary_nodes!

# function process!(block::Block{I,F}) where {I,F}
#     nx = block.nx; ny = block.ny
#     points_y1 = fill(Point(0.0,0.0,0.0), ny+1)
#     points_y2 = fill(Point(0.0,0.0,0.0), ny+1)
#     discretise_edge!(points_y1, 1, block.p1, block.p3, ny)
#     discretise_edge!(points_y2, 1, block.p2, block.p4, ny)
#     nodei = 1
#     for i ∈ eachindex(points_y1)
#         nodei_curr = discretise_edge!(block.nodes, nodei, points_y1[i], points_y2[i], nx)
#         nodei = nodei_curr
#     end
#     nothing
# end

# function discretise_edge!(
#     # points::Vector{Point{F}}, point_index::Integer, e::Edge{F}, ncells::Integer
#     points::Vector{Point{F}}, node_idx::Integer, p1::Point{F}, p2::Point{F}, ncells::Integer
#     ) where F<:AbstractFloat
#     nsegments = ncells - 1
#     d = p2.coords - p1.coords
#     d_mag = norm(d)
#     e1 = d/d_mag
#     spacing = d_mag/ncells

#     points[node_idx] = p1
#     points[node_idx + nsegments + 1] = p2

#     j = 1
#     for pointi ∈ (node_idx+1):(node_idx + ncells)
#         points[pointi] = Point(spacing*e1*(j) + p1.coords)
#         j += 1
#         node_idx = pointi + 1
#     end
#     return node_idx
# end

# function collect_elements!(block::Block)
#     element_i = 0
#     node_i = 0
#     for y_i ∈ 1:block.ny
#         for x_i ∈ 1:block.nx
#             element_i += 1
#             node_i += 1
#             block.elements[element_i] = Element(
#                 SVector{4, Int64}(node_i,node_i+1,node_i+(block.nx+1),node_i+(block.nx+1)+1), 
#                 SVector{3, Float64}(0.0,0.0,0.0)
#             )
            
#         end
#         node_i += 1
#     end
# end

# function centre!(block::Block{I,F}) where {I,F}
#     for i ∈ eachindex(block.elements)
#         sum =  SVector{3, F}(0.0,0.0,0.0)
#         for id ∈ block.elements[i].nodesID
#             sum += block.nodes[id].coords
#         end
#         centre = sum/4
#         block.elements[i] = Element(block.elements[i].nodesID, centre)
#     end
# end

function tag_as_boundary!(edge::Edge{I}) where I
    Edge(edge.p1, edge.p2, edge.n, true, edge.nodesID)
end

function tag_as_boundary!(point::Point{F}) where F
    Point(point.coords, true, false)
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

function nodes_on_edge!(
    multiblock::MultiBlock{I,F}, edgeID::Integer, counter::Integer
    ) where {I,F}
    edge = multiblock.definition.edges[edgeID]
    ncells = edge.n
    p1 = multiblock.definition.points[edge.p1]
    p2 = multiblock.definition.points[edge.p2]
    d = p2.coords - p1.coords
    d_mag = norm(d)
    e1 = d/d_mag
    spacing = d_mag/ncells

    if !p1.processed
        multiblock.nodes[counter] = Node(p1.coords)
        multiblock.definition.points[edge.p1] = Point(p1.coords, true, true)
        counter += 1
    end

    # j = 1
    for j ∈ 1:(ncells-1)
        multiblock.nodes[counter] = Node(spacing*e1*j + p1.coords)
        # j += 1
        counter += 1
    end

    if !p2.processed
        multiblock.nodes[counter] = Node(p2.coords)
        p2 = multiblock.definition.points[edge.p2] = Point(p2.coords, true, true)
        counter += 1
    end
    return counter
end

function update_block_matrix!(
    multiblock::MultiBlock{I,F}, edgeID::Integer, counter::Integer
    ) where {I,F}
    edge = multiblock.definition.edges[edgeID]
    ncells = edge.n
    for block ∈ multiblock.definition.blocks
        if block.edge_x1 == edgeID
            @views block.nodesID[:,1] = [i for i ∈ counter:(counter + ncells)]
            println("Here")
        end
        if block.edge_x2 == edgeID
            nothing
        end
        if block.edge_y1 == edgeID
            nothing
        end
        if block.edge_y2 == edgeID
            nothing
        end
    end
end

function generate_boundary_nodes!(
    multiblock::MultiBlock{I,F}, counter::Integer) where {I,F}
    # block = multiblock.definition.blocks[1]
    edges = multiblock.definition.edges
    # patch = multiblock.definition.patches[1]
    for patch ∈ multiblock.definition.patches
        for edgeID ∈ patch.edgesID
            counter = nodes_on_edge!(multiblock, edgeID, counter)
            update_block_matrix!(multiblock, edgeID, counter)
        end
    end
end

function generate_internal_nodes!()
    nothing
end

