export line!, quad
export build!
export find_edge_in_blocks!

function build!(builder::MeshBuilder2D{I,F}) where {I,F}
    mesh = preallocate_mesh(builder)
    generate_inner_points!(mesh, builder)
    generate_elements!(mesh, builder)
    counter = generate_boundary_faces!(mesh, builder)
    counter = generate_interface_faces!(counter, mesh, builder)
    generate_internal_faces!(counter, mesh, builder)
    mesh
end

function generate_internal_faces!(
    facei::I, mesh::Mesh2{I,F}, builder::MeshBuilder2D{I,F}
    ) where {I,F}
    (; blocks) = builder
    (; nodes, faces) = mesh
    for block ∈ blocks
        nx, ny = block.nx, block.ny
        nodesID = block.nodesID
        for yi = 1:ny-1
            for xi = 1:nx
                p1_ID = nodesID[xi,yi+1]
                p2_ID = nodesID[xi+1,yi+1]
                centre = geometric_centre(nodes, SVector{2,I}(p1_ID,p2_ID))
                face = faces[facei]
                face = @set face.nodesID = SVector{2,I}(p1_ID, p2_ID)
                faces[facei] = @set face.centre =  centre
                block.facesID_NS[xi,yi+1] = facei
                facei += 1
            end
        end
        for xi = 1:nx-1
            for yi = 1:ny
                p1_ID = nodesID[xi+1,yi]
                p2_ID = nodesID[xi+1,yi+1]
                centre = geometric_centre(nodes, SVector{2,I}(p1_ID,p2_ID))
                face = faces[facei]
                face = @set face.nodesID = SVector{2,I}(p1_ID, p2_ID)
                faces[facei] = @set face.centre =  centre
                block.facesID_EW[xi+1,yi] = facei
                facei += 1
            end
        end
    end
    facei
end

function generate_interface_faces!(
    facei::I, mesh::Mesh2{I,F}, builder::MeshBuilder2D{I,F}
    ) where {I,F}
    (; edges, blocks) = builder
    (; nodes, faces) = mesh
    blockPair = [Block(zero(I)) for _ ∈ 1:2] #fill(Block(zero(I)), 2)
    edgeIndexPair = zeros(I,2)
    for (edgeID, edge) ∈ enumerate(edges)
        if !edge.boundary
            find_edge_in_blocks!(blockPair, edgeIndexPair, blocks, edgeID)
            nodesID = edge.nodesID
            for pointi ∈ 1:(length(nodesID) - 1)
                p1_ID = nodesID[pointi]
                p2_ID = nodesID[pointi+1]
                centre = geometric_centre(nodes, SVector{2,I}(p1_ID,p2_ID))
                face = faces[facei]
                face = @set face.nodesID = SVector{2,I}(p1_ID, p2_ID)
                faces[facei] = @set face.centre =  centre
                for (i, block) ∈ enumerate(blockPair)
                    index_to_block_edge!(block, edgeIndexPair[i], pointi, facei)
                end
                facei += 1
            end

        end
    end
    facei
end

function find_edge_in_blocks!(blockPair, edgeIndexPair, blocks, edgeID::I) where I
    counter = zero(I)
    for block ∈ blocks
        for edgei ∈ eachindex(block.edgesID)
            if block.edgesID[edgei] == edgeID
                counter += 1
                blockPair[counter] = block
                edgeIndexPair[counter] = edgei
            end
        end
    end
end

function generate_boundary_faces!(
    mesh::Mesh2{I,F}, builder::MeshBuilder2D{I,F}
    ) where {I,F}
    (; edges, patches, blocks) = builder
    (; nodes, faces, boundaries) = mesh
    facei = one(I)
    for patchi ∈ eachindex(patches)
        patch = patches[patchi]
        i = 1
        for edgeID ∈ patch.edgesID
            block, edgei = locate_boundary_in_blocks(blocks, edgeID)
            nodesID = edges[edgeID].nodesID
            for pointi ∈ 1:(length(nodesID) - 1)
                p1_ID = nodesID[pointi]
                p2_ID = nodesID[pointi+1]
                # push!(boundary.nodesID, p1_ID)
                centre = geometric_centre(nodes, SVector{2,I}(p1_ID,p2_ID))
                face = faces[facei]
                face = @set face.nodesID = SVector{2,I}(p1_ID, p2_ID)
                faces[facei] = @set face.centre =  centre
                index_to_block_edge!(block, edgei, pointi, facei)
                boundaries[patchi].facesID[i] = facei
                facei += 1
                i += 1
            end
        end
        # boundaries[patchi] = @set boundary.name = patch.name
    end
    facei
end

function index_to_block_edge!(block, edgei, pointi::I, facei::I) where I
        if edgei == 1
            block.facesID_NS[pointi,1] = facei
        elseif edgei == 2
            block.facesID_NS[pointi,end] = facei
        elseif edgei == 3
            block.facesID_EW[1, pointi] = facei
        elseif edgei == 4
            block.facesID_EW[end, pointi] = facei
        end
end

function locate_boundary_in_blocks(blocks::Vector{Block{I}}, edgeID::I) where I<:Integer
    for block ∈ blocks
        for edgei ∈ eachindex(block.edgesID)
            if block.edgesID[edgei] == edgeID
                return block, edgei
            end
        end
    end
    nothing
end

# function centre(cell)
#     Node(cell.centre)
# end

function generate_elements!(
    mesh::Mesh2{I,F}, builder::MeshBuilder2D{I,F}
    ) where {I,F}
    (; blocks) = builder
    (; nodes, cells) = mesh
    celli = zero(I)
    for block ∈ blocks
        nx, ny = block.nx, block.ny
        nodesID = block.nodesID
        for yi ∈ 1:ny
            for xi ∈ 1:nx
                celli += 1
                n1 = nodesID[xi,yi]
                n2 = nodesID[xi+1,yi]
                n3 = nodesID[xi+1,yi+1]
                n4 = nodesID[xi,yi+1]
                # nodeList = SVector{4, I}(n1,n2,n3,n4)
                nodeList = [n1,n2,n3,n4] # Can this allocation be removed?
                centre = geometric_centre(nodes, nodeList)
                block.elementsID[xi,yi] = celli
                cell = cells[celli]
                cell = @set cell.nodesID = nodeList
                cell = @set cell.centre = centre
                cells[celli] = cell
            end
        end
    end
end

function generate_inner_points!(
    mesh::Mesh2{I,F}, builder::MeshBuilder2D{I,F}
    ) where {I,F}
    points_count = total_edge_points(builder)
    (; points, edges, blocks) = builder
    for block ∈ blocks
        edgeID1 = block.edgesID[3] 
        edgeID2 = block.edgesID[4] 
        edge_y1 = edges[edgeID1] # "y-dir" edge 1
        edge_y2 = edges[edgeID2] # "y-dir" edge 2
        for yi ∈ 2:block.ny
            pID1 = edge_y1.nodesID[yi]
            pID2 = edge_y2.nodesID[yi]
            p1 = points[pID1]
            p2 = points[pID2]
            δx, normal = linear_distribution(p1, p2, block.nx)
            for xi ∈ 2:block.nx
                points_count += 1
                mesh.nodes[points_count] = Node(p1.coords + δx(xi-1, 1)*normal)
                block.nodesID[xi, yi] = points_count
            end
        end
    end
end

function preallocate_mesh(builder::MeshBuilder2D{I,F}) where {I,F}
    nodes = [Node(zero(F)) for _ ∈ 1:total_points(builder)]
    cells = [Cell(I, F) for _ ∈ 1:total_elements(builder)]
    tag_boundary_edges(builder)
    faces = [Face2D(I,F) for _ ∈ 1:total_faces(builder)]
    boundaries = preallocate_boundaries(builder)
    # Copy existing edge points to new points vector
    for i ∈ eachindex(builder.points)
        nodes[i] = builder.points[i]
    end
    Mesh2(cells, faces, boundaries, nodes)
end

function preallocate_boundaries(builder::MeshBuilder2D{I,F}) where {I,F}
    (; edges, patches) = builder
    boundaries = Boundary{I}[]
    for patchi ∈ eachindex(patches)
        patch = patches[patchi]
        ncells = zero(I)
        for edgeID ∈ patch.edgesID
            ncells += edges[edgeID].ncells
        end
        nfaces = ncells
        # nnodes = ncells + 1
        face_nodesID = [zeros(I,2) for _ ∈ 1:ncells]
        push!(
            boundaries, 
            Boundary(
                # patch.name, zeros(I, nnodes), zeros(I, nfaces), zeros(I,ncells)
                patch.name, face_nodesID, zeros(I, nfaces), zeros(I,ncells)
                ))
    end
    boundaries
end

function total_faces(builder::MeshBuilder2D{I,F}) where{I,F}
    (; blocks, edges) = builder
    nfaces = zero(I)
    for block ∈ blocks
        nx = block.nx
        ny = block.ny
        faces_NS = (nx)*(ny+1)
        faces_EW = (nx+1)*(ny)
        nfaces += faces_NS + faces_EW # (nx+1)*(ny+1)
    end
    for edge ∈ edges # find internal edges and remove from total
        if !edge.boundary
            nfaces -= edge.ncells
        end
    end
    nfaces
end

function tag_boundary_edges(builder::MeshBuilder2D{I,F}) where{I,F}
    (; patches, edges) = builder
    for patch ∈ patches
        for ID ∈ patch.edgesID
            edge = edges[ID]
            edges[ID] = @set edge.boundary = true
        end
    end
end

function total_edge_points(builder::MeshBuilder2D{I,F}) where {I,F}
    length(builder.points)
end

function total_points(builder::MeshBuilder2D{I,F}) where {I,F}
    edge_points = total_edge_points(builder)
    inner_points = zero(I)
    for block ∈ builder.blocks
        inner_points += block.inner_points
    end
    edge_points + inner_points
end

function total_elements(builder::MeshBuilder2D{I,F}) where {I,F}
    total_elements = zero(I)
    for block ∈ builder.blocks
        total_elements += block.nx*block.ny
    end
    total_elements
end

function line!(pts::Vector{Node{F}}, p1_index::I, p2_index::I, ncells::I) where {I,F}
    nodesID = fill(zero(I), ncells+1)
    nodesID[1] = p1_index
    nodesID[end] = p2_index

    p1 = pts[p1_index]
    p2 = pts[p2_index]
    δx, normal = linear_distribution(p1, p2, ncells)
    for j ∈ 2:ncells
        push!(pts, Node(p1.coords + δx(j-1, 1)*normal) )
        nodesID[j] = length(pts)
    end
    return Edge(nodesID, ncells, false)
end

function linear_distribution(p1::Node{F}, p2::Node{F}, ncells::I) where {I,F}
    d = p2.coords - p1.coords
    d_mag = norm(d)
    normal = d/d_mag
    δx(i, β) = (d_mag/ncells)*i
    δx, normal
end

function line!(pts::Vector{Node{F}}, p1_index::I, p2_index::I, ncells::I, ratio::Number) where {I,F}
    nodesID = fill(zero(I), ncells+1)
    nodesID[1] = p1_index
    nodesID[end] = p2_index

    p1 = pts[p1_index]
    p2 = pts[p2_index]
    δx, normal = symmetric_tanh_distribution(p1, p2, ncells)
    for j ∈ 2:ncells
        push!(pts, Node(p1.coords + δx(j-1, ratio) * normal))
        nodesID[j] = length(pts)
    end
    return Edge(nodesID, ncells, false)
end

function symmetric_tanh_distribution(
    p1::Node{F}, p2::Node{F}, ncells::I) where {I,F}
    d = p2.coords - p1.coords
    d_mag = norm(d)
    normal = d/d_mag
    δx(i, β) = begin
        η = (i)/(ncells)
        (d_mag/2)*(1.0 - tanh(β*(1-2*η))/tanh(β))
    end
    δx, normal
end

function quad(edges::Vector{Edge{I}}, edgesID::Vector{I}) where I
    IDs = SVector{4,I}(edgesID)
    nx = edges[IDs[1]].ncells
    ny = edges[IDs[3]].ncells
    nodesID =  zeros(I, nx+1, ny+1) # Matrix to hold pointID information
    @. nodesID[:,1]      = edges[IDs[1]].nodesID
    @. nodesID[:,end]    = edges[IDs[2]].nodesID
    @. nodesID[1,:]      = edges[IDs[3]].nodesID
    @. nodesID[end,:]    = edges[IDs[4]].nodesID
    elementsID = zeros(I, nx, ny)
    facesID_NS = zeros(I, nx, ny+1)
    facesID_EW = zeros(I, nx+1, ny)
    inner_points = (nx+1-2)*(ny+1-2)
    Block(IDs, nx, ny, nodesID, elementsID, facesID_NS, facesID_EW, inner_points, true)
end

# function curve(pts::Vector{Node{F}}, p1_index::I, p2_index::I, ncells::I) where {I,F}
#     nodesID = fill(zero(I), ncells+1)
#     # points = fill(Node(zero(F)), ncells-1)
#     nodesID[1] = p1_index
#     nodesID[end] = p2_index

#     p1 = pts[p1_index]
#     p2 = pts[p2_index]

#     # points[1] = p1; points[end] = p2

#     d = p2.coords - p1.coords
#     d_mag = norm(d)
#     e1 = d/d_mag
#     spacing = d_mag/ncells
#     for j ∈ 2:ncells
#     # for j ∈ eachindex(points)
#         # points[j] = Node(spacing*e1*j + p1.coords)
#         push!(pts, Node(p1.coords +
#             [spacing*e1*(j-1),
#             # spacing*e1[2]*(j-1),
#             spacing*e1[2]*(j-1) + 0.25*sin(pi*spacing*e1[1]*(j-1)),
#             spacing*e1[3]*(j-1)]))
#         nodesID[j] = length(pts)

#     end
#     return Edge(nodesID, ncells, false)
# end