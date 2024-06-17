export mesh2_from_UNV

mesh2_from_UNV(mesh; integer=Int64, float=Float64) = begin
    boundaries = Vector{Boundary{Vector{integer}}}(undef, length(mesh.boundaries))
    cells = Vector{Cell{integer,float}}(undef, length(mesh.cells))
    faces = Vector{Face2D{integer,float}}(undef, length(mesh.faces))
    nodes = Vector{Node{Vector{integer},float}}(undef, length(mesh.nodes))

    for (i, b) ∈ enumerate(mesh.boundaries)
        boundaries[i] = Boundary{Vector{integer}}(b.name, b.facesID, b.cellsID)
    end

    for (i, n) ∈ enumerate(mesh.nodes)
        nodes[i] = Node(n.coords, n.neighbourCells)
    end

    # PROCESSING CELLS
    # Calculate array size needed for nodes and faces
    nnodes = zero(integer)
    nfaces = zero(integer)
    for cell ∈ mesh.cells
        (; nodesID, facesID) = cell 
        nnodes += length(nodesID)
        nfaces += length(facesID)
    end

    # Initialise arrays 
    cell_nodes = Vector{integer}(undef, nnodes)
    cell_faces = Vector{integer}(undef, nfaces)
    cell_neighbours = Vector{integer}(undef, nfaces)
    cell_nsign = Vector{integer}(undef, nfaces)

    ni = 1 # node counter
    fi = 1 # face counter
    for (i, cell) ∈ enumerate(mesh.cells)
        (;nodesID, facesID, neighbours, nsign) = cell
        # node array loop
        nodes_range = ni:(ni + length(nodesID) - 1) #SVector{2,integer}(length(nodesID), ni)
        for nodeID ∈ nodesID
            cell_nodes[ni] = nodeID 
            ni += 1
        end
        # cell array loop
        # faces_range = SVector{2,integer}(length(facesID), fi)
        faces_range = fi:(fi + length(facesID) - 1) 
        for j ∈ eachindex(facesID)
            cell_faces[fi] = facesID[j]
            cell_neighbours[fi] = neighbours[j]
            cell_nsign[fi] = nsign[j]
            fi += 1
        end

        # cell assignment
        cells[i] = Cell{integer,float}(
            cell.centre,
            cell.volume,
            nodes_range,
            faces_range
        ) |> cu
    end

    # PROCESSING FACES
    # Calculate array size needed for all face nodes
    nnodes = zero(integer)
    for face ∈ mesh.faces
        (; nodesID) = face 
        nnodes += length(nodesID)
    end

    # Initialise arrays 
    face_nodes = Vector{integer}(undef, nnodes)

    ni = 1 # node counter
    for (i, face) ∈ enumerate(mesh.faces)
        (;nodesID) = face
        # node array loop
        nodes_range = ni:(ni + length(nodesID) - 1) #SVector{2,integer}(length(nodesID), ni)
        for nodeID ∈ nodesID
            face_nodes[ni] = nodeID 
            ni += 1
        end

        # face assignment
        faces[i] = Face2D{integer,float}(
            nodes_range,
            face.ownerCells,
            face.centre,
            face.normal,
            face.e,
            face.area,
            face.delta,
            face.weight
        ) |> cu
    end

    Mesh2{Vector{Cell{integer,float}}, Vector{integer}, Vector{Face2D{integer,float}}, Vector{Boundary{Vector{integer}}}, Vector{Node{Vector{integer},float}}}(
        cells,
        cell_nodes,
        cell_faces,
        cell_neighbours,
        cell_nsign,
        faces,
        face_nodes,
        boundaries,
        nodes,
    ) |> cu
end