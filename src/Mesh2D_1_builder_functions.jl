export line!, quad
export centre
export preallocate_meshbuilder, generate_inner_points!, generate_elements!
export generate_boundary_faces!, generate_interface_faces!, generate_internal_faces!

function generate_internal_faces!(
    facei::I, builder::MeshBuilder2D{I,F}, mesh_info::Wireframe{I,F}
    ) where {I,F}
    blocks = mesh_info.blocks
    points = builder.points
    faces = builder.faces
    for block ∈ blocks
        nx, ny = block.nx, block.ny
        pointsID = block.pointsID
        for yi = 1:ny-1
            for xi = 1:nx
                p1_ID = pointsID[xi,yi+1]
                p2_ID = pointsID[xi+1,yi+1]
                centre = geometric_centre(points, SVector{2,I}(p1_ID,p2_ID))
                faces[facei] = Face2D(SVector{2,I}(p1_ID, p2_ID), centre)
                # index_to_NS_matrix!(block, xi, yi+1, facei)
                block.facesID_NS[xi,yi+1] = facei
                facei += 1
            end
        end
        for xi = 1:nx-1
            for yi = 1:ny
                p1_ID = pointsID[xi+1,yi]
                p2_ID = pointsID[xi+1,yi+1]
                centre = geometric_centre(points, SVector{2,I}(p1_ID,p2_ID))
                faces[facei] = Face2D(SVector{2,I}(p1_ID, p2_ID), centre)
                # index_to_NS_matrix!(block, row::I, col::I, facei::I)
                block.facesID_EW[xi+1,yi] = facei
                facei += 1
            end
        end
    end
    facei
end

function index_to_NS_matrix!(block, row::I, col::I, facei::I) where I<:Integer
    NS = block.facesID_NS # North-South Matrix
    NS[row,col] = facei
    nothing
end

function index_to_EW_matrix!()
    nothing
end

function generate_interface_faces!(
    facei::I, builder::MeshBuilder2D{I,F}, mesh_info::Wireframe{I,F}
    ) where {I,F}
    blocks = mesh_info.blocks
    edges = mesh_info.edges
    points = builder.points
    faces = builder.faces
    blockPair = fill(Block(zero(I)), 2)
    edgeIndexPair = zeros(I,2)
    for (edgeID, edge) ∈ enumerate(edges)
        if !edge.boundary
            find_edge_in_blocks!(blockPair, edgeIndexPair, blocks, edgeID)
            pointsID = edge.pointsID
            for pointi ∈ 1:(length(pointsID) - 1)
                p1_ID = pointsID[pointi]
                p2_ID = pointsID[pointi+1]
                centre = geometric_centre(points, SVector{2,I}(p1_ID,p2_ID))
                faces[facei] = Face2D(SVector{2,I}(p1_ID, p2_ID), centre)
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
    builder::MeshBuilder2D{I,F}, mesh_info::Wireframe{I,F}
    ) where {I,F}
    blocks = mesh_info.blocks
    patches = mesh_info.patches
    edges = mesh_info.edges
    points = builder.points
    faces = builder.faces
    facei = one(I)
    for patch ∈ patches
        for edgeID ∈ patch.edgesID
            block, edgei = locate_boundary_in_blocks(blocks, edgeID)
            pointsID = edges[edgeID].pointsID
            for pointi ∈ 1:(length(pointsID) - 1)
                p1_ID = pointsID[pointi]
                p2_ID = pointsID[pointi+1]
                centre = geometric_centre(points, SVector{2,I}(p1_ID,p2_ID))
                faces[facei] = Face2D(SVector{2,I}(p1_ID, p2_ID), centre)
                index_to_block_edge!(block, edgei, pointi, facei)
                facei += 1
            end
        end
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

function centre(element)
    Point(element.centre)
end

function generate_elements!(
    builder::MeshBuilder2D{I,F}, mesh_info::Wireframe{I,F}
    ) where {I,F}
    blocks = mesh_info.blocks
    points = builder.points
    elements = builder.elements
    elementi = zero(I)
    for block ∈ blocks
        nx, ny = block.nx, block.ny
        pointsID = block.pointsID
        for yi ∈ 1:ny
            for xi ∈ 1:nx
                elementi += 1
                n1 = pointsID[xi,yi]
                n2 = pointsID[xi+1,yi]
                n3 = pointsID[xi+1,yi+1]
                n4 = pointsID[xi,yi+1]
                nodeList = SVector{4, I}(n1,n2,n3,n4)
                centre = geometric_centre(points, nodeList)
                block.elementsID[xi,yi] = elementi
                elements[elementi] = Element(nodeList, centre)
            end
        end
    end
end

function geometric_centre(points::Vector{Point{F}}, nodeList::SVector{N, I}) where {I,F,N}
    sum = SVector{3, F}(0.0,0.0,0.0)
        for ID ∈ nodeList
            sum += points[ID].coords
        end
    return sum/(length(nodeList))
end

function generate_inner_points!(
    builder::MeshBuilder2D{I,F}, mesh_info::Wireframe{I,F}
    ) where {I,F}
    points_count = total_edge_points(mesh_info)
    points = mesh_info.points
    edges = mesh_info.edges
    blocks = mesh_info.blocks
    for block ∈ blocks
        edgeID1 = block.edgesID[3] 
        edgeID2 = block.edgesID[4] 
        edge_y1 = edges[edgeID1] # "y-dir" edge 1
        edge_y2 = edges[edgeID2] # "y-dir" edge 2
        for yi ∈ 2:block.ny
            pID1 = edge_y1.pointsID[yi]
            pID2 = edge_y2.pointsID[yi]
            p1 = points[pID1]
            p2 = points[pID2]
            spacing, normal = linear_distribution(p1, p2, block.nx)
            for xi ∈ 2:block.nx
                points_count += 1
                builder.points[points_count] = Point(p1.coords + spacing*normal*(xi-1))
                block.pointsID[xi, yi] = points_count
            end
        end
    end
end

function preallocate_meshbuilder(mesh_info::Wireframe{I,F}) where {I,F}
    points = fill(Point(zero(F)), total_points(mesh_info))
    elements = fill(Element(zero(I), zero(F)), total_elements(mesh_info))
    tag_boundary_edges(mesh_info)
    faces = fill(Face2D(I,F), total_faces(mesh_info))
    # Copy existing edge points to new points vector
    for i ∈ eachindex(mesh_info.points)
        points[i] = mesh_info.points[i]
    end
    MeshBuilder2D(points, elements, faces)
end

function total_faces(mesh_info::Wireframe{I,F}) where{I,F}
    blocks = mesh_info.blocks
    edges = mesh_info.edges
    nfaces = zero(I)
    for block ∈ blocks
        nx = block.nx
        ny = block.ny
        faces_NS = (nx)*(ny+1)
        faces_EW = (ny+1)*(nx)
        nfaces += faces_NS + faces_EW
    end
    for edge ∈ edges # find internal edges and remove from total
        if !edge.boundary
            nfaces -= 2*edge.ncells
        end
    end
    nfaces
end

function tag_boundary_edges(mesh_info::Wireframe{I,F}) where{I,F}
    patches = mesh_info.patches
    edges = mesh_info.edges
    for patch ∈ patches
        for ID ∈ patch.edgesID
            edges[ID] = tag_as_boundary(edges[ID])
        end
    end
end

function tag_as_boundary(edge::Edge{I}) where I
    Edge(edge.pointsID, edge.ncells, true)
end

function total_edge_points(mesh_info::Wireframe{I,F}) where {I,F}
    length(mesh_info.points)
end

function total_points(mesh_info::Wireframe{I,F}) where {I,F}
    edge_points = total_edge_points(mesh_info)
    inner_points = zero(I)
    for block ∈ mesh_info.blocks
        inner_points += block.inner_points
    end
    edge_points + inner_points
end

function total_elements(mesh_info::Wireframe{I,F}) where {I,F}
    total_elements = zero(I)
    for block ∈ mesh_info.blocks
        total_elements += block.nx*block.ny
    end
    total_elements
end

function line!(pts::Vector{Point{F}}, p1_index::I, p2_index::I, ncells::I) where {I,F}
    pointsID = fill(zero(I), ncells+1)
    pointsID[1] = p1_index
    pointsID[end] = p2_index

    p1 = pts[p1_index]
    p2 = pts[p2_index]
    spacing, normal = linear_distribution(p1, p2, ncells)
    for j ∈ 2:ncells
        push!(pts, Point(p1.coords + spacing*normal*(j-1)))
        pointsID[j] = length(pts)
    end
    return Edge(pointsID, ncells, false)
end

function linear_distribution(p1::Point{F}, p2::Point{F}, ncells::I) where {I,F}
    d = p2.coords - p1.coords
    d_mag = norm(d)
    normal = d/d_mag
    spacing = d_mag/ncells
    spacing, normal
end

function quad(edges::Vector{Edge{I}}, edgesID::Vector{I}) where {I,F}
    IDs = SVector{4,I}(edgesID)
    nx = edges[IDs[1]].ncells
    ny = edges[IDs[3]].ncells
    pointsID =  zeros(I, nx+1, ny+1) # Matrix to hold pointID information
    pointsID[:,1]      = edges[IDs[1]].pointsID
    pointsID[:,end]    = edges[IDs[2]].pointsID
    pointsID[1,:]      = edges[IDs[3]].pointsID
    pointsID[end,:]    = edges[IDs[4]].pointsID
    elementsID = zeros(I, nx, ny)
    facesID_NS = zeros(I, nx, ny+1)
    facesID_EW = zeros(I, nx+1, ny)
    inner_points = (nx+1-2)*(ny+1-2)
    Block(IDs, nx, ny, pointsID, elementsID, facesID_NS, facesID_EW, inner_points, true)
end

# function curve(pts::Vector{Point{F}}, p1_index::I, p2_index::I, ncells::I) where {I,F}
#     pointsID = fill(zero(I), ncells+1)
#     # points = fill(Point(zero(F)), ncells-1)
#     pointsID[1] = p1_index
#     pointsID[end] = p2_index

#     p1 = pts[p1_index]
#     p2 = pts[p2_index]

#     # points[1] = p1; points[end] = p2

#     d = p2.coords - p1.coords
#     d_mag = norm(d)
#     e1 = d/d_mag
#     spacing = d_mag/ncells
#     for j ∈ 2:ncells
#     # for j ∈ eachindex(points)
#         # points[j] = Point(spacing*e1*j + p1.coords)
#         push!(pts, Point(p1.coords +
#             [spacing*e1*(j-1),
#             # spacing*e1[2]*(j-1),
#             spacing*e1[2]*(j-1) + 0.25*sin(pi*spacing*e1[1]*(j-1)),
#             spacing*e1[3]*(j-1)]))
#         pointsID[j] = length(pts)

#     end
#     return Edge(pointsID, ncells, false)
# end