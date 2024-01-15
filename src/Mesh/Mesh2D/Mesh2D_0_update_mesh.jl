import FVM_1D.UNV2

export update_mesh_format

update_mesh_format(mesh::UNV2.Mesh2; integer=Int64, float=Float64) = begin
    @info "Update to new mesh format (temporary solution)"

    # Pre-allocate memory for mesh entities

    boundaries = Vector{Boundary{Symbol, Vector{integer}}}(
        undef, length(mesh.boundaries)
        )
    cells = Vector{Cell{float,SVector{3,float},UnitRange{integer}}}(
        undef, length(mesh.cells)
        )
    faces = Vector{Face2D{float,SVector{2,integer},SVector{3,float},UnitRange{integer}}}(
        undef, length(mesh.faces)
        )
    nodes = Vector{Node{SVector{3,float}, UnitRange{integer}}}(
        undef, length(mesh.nodes)
        )

    # PROCESSING BOUNDARIES

    for (i, b) ∈ enumerate(mesh.boundaries)
        boundaries[i] = Boundary(b.name, b.facesID, b.cellsID)
    end

    # PROCESSING NODES 

    # Determine array size to hold node_cells information 
    n = zero(integer)
    for node ∈ mesh.nodes 
        n += length(node.neighbourCells)
    end
    node_cells = Vector{integer}(undef, n)

    # Copy neighbourCells indices to node_cells array and build new Node type
    start_index = one(integer)
    for (ni, node) ∈ enumerate(mesh.nodes)
        n_neighbours = length(node.neighbourCells)
        range = start_index:(start_index + n_neighbours - one(integer))
        nodes[ni] = Node(node.coords, range)
        node_cells[range] .= node.neighbourCells
        start_index += n_neighbours
    end
    
    # PROCESSING CELLS

    # Calculate array size needed for cell node and face data
    nnodes = zero(integer)
    nfaces = zero(integer)
    for cell ∈ mesh.cells
        (; nodesID, facesID) = cell 
        nnodes += length(nodesID)
        nfaces += length(facesID)
    end

    # Pre-allocate arrays 
    cell_nodes = Vector{integer}(undef, nnodes)
    cell_faces = Vector{integer}(undef, nfaces)
    cell_neighbours = Vector{integer}(undef, nfaces)
    cell_nsign = Vector{integer}(undef, nfaces)

    ni = one(integer) # node index counter
    fi = one(integer) # face index counter
    for (i, cell) ∈ enumerate(mesh.cells)
        (;nodesID, facesID, neighbours, nsign) = cell

        # collect node data
        nodes_range = ni:(ni + length(nodesID) - 1)
        for nodeID ∈ nodesID
            cell_nodes[ni] = nodeID 
            ni += 1
        end

        # collect face data
        faces_range = fi:(fi + length(facesID) - 1) 
        for j ∈ eachindex(facesID)
            cell_faces[fi] = facesID[j]
            cell_neighbours[fi] = neighbours[j]
            cell_nsign[fi] = nsign[j]
            fi += 1
        end

        # cell assignment
        cells[i] = Cell(
            cell.centre,
            cell.volume,
            nodes_range,
            faces_range
        )
    end

    # PROCESSING FACES

    # Calculate array size needed for face node data
    nnodes = zero(integer)
    for face ∈ mesh.faces
        (; nodesID) = face 
        nnodes += length(nodesID)
    end

    # Pre-allocate arrays 
    face_nodes = Vector{integer}(undef, nnodes)

    ni = one(integer) # node index counter
    for (i, face) ∈ enumerate(mesh.faces)
        (;nodesID) = face
        # node array loop
        nodes_range = ni:(ni + length(nodesID) - 1)
        for nodeID ∈ nodesID
            face_nodes[ni] = nodeID 
            ni += 1
        end

        # face assignment
        faces[i] = Face2D(
            nodes_range,
            face.ownerCells,
            face.centre,
            face.normal,
            face.e,
            face.area,
            face.delta,
            face.weight
        ) 
    end

    # CONSTRUCT FINAL MESH (MESH2)
    Mesh2(
        cells,
        cell_nodes,
        cell_faces,
        cell_neighbours,
        cell_nsign,
        faces,
        face_nodes,
        boundaries,
        nodes,
        node_cells
    )
end