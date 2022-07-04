export build_mesh

function build_mesh(meshFile; scaleFactor=1.0, TI=Int64, TF=Float64)
    stats = @timed begin
    println("Loading mesh...")
    points, elements, boundaryFaces = load(meshFile, TI, TF);
    println("File read successfully")
    if scaleFactor != 1
        scalePoints!(points, scaleFactor)
    end
    println("Generating mesh connectivity...")
    nodes, faces, cells = generate(points, elements, boundaryFaces)
    # nodes, cells, faces, boundaries = connect(points, elements, boundaryFaces)
    # preprocess!(nodes, faces, cells, boundaries)
    # mesh = Mesh.FullMesh(nodes, faces, cells, boundaries)
    end
    # println("Done! Execution time: ", @sprintf "%.6f" stats.time)
    # println("Mesh ready!")
    return nodes, faces, cells
end

function generate(points, elements, boundaryFaces)
    bfaces = total_boundary_faces(boundaryFaces)
    first_element = bfaces + 1
    nodes = generate_nodes(first_element, points, elements);
    faces = generate_faces(first_element, elements, nodes, boundaryFaces);
    cells = generate_cells(first_element, elements, faces, nodes)
    face_cell_connectivity!(cells, faces, nodes)
    # facesRaw, boundaries = generate_faces(elements, nodes, boundaryFaces);
    # faces = face_connectivity(facesRaw);
    # cells = generate_cells(points, elements, faces);
    # return nodes, cells, faces, boundaries
    return nodes, faces, cells
end


# Lower level functions

function scalePoints!(points::Vector{Point{TF}}, scaleFactor) where TF
    for point ∈ points
        point.xyz = point.xyz*scaleFactor
    end
end

function total_boundary_faces(boundaries::Vector{UNV.Boundary{TI}}) where TI
    sum = zero(TI)
    @inbounds for boundary ∈ boundaries
        sum += length(boundary.elements)
    end
    return sum
end

function generate_nodes(first_element, points::Vector{Point{TF}}, elements) where TF
   nodes = Node{TF}[]
   for i ∈ 1:length(points)
       point = points[i].xyz
       push!(nodes, Node(point))
   end
   cellID = 0 # counter for cells
   for i ∈ first_element:length(elements) 
           cellID += 1
           for nodeID ∈ elements[i].vertices
               push!(nodes[nodeID].neighbourCells, cellID)
           end
   end
   return nodes
end

function generate_faces(
    first_element, 
    elements::Vector{UNV.Element{TI}}, 
    nodes::Vector{Node{TF}}, 
    boundaryFaces
    ) where {TI,TF}
    # Generate all faces from element and point information
    faces = Face2D{TI,TF}[]

    # Start with boundary faces (stored in "elements")
    for i ∈ 1:(first_element-1) # loop over elements stored before the first element
        face = Face2D(TI,TF)
        vertex1 = elements[i].vertices[1]
        vertex2 = elements[i].vertices[2]
        if vertex1 < vertex2
            face = @set face.nodesID = SVector{2,TI}(vertex1, vertex2)
            push!(faces, face)
            continue
        elseif vertex1 > vertex2 
            face = @set face.nodesID = SVector{2,TI}(vertex2, vertex1)
            push!(faces, face)
            continue
        else
            throw("Boundary elements are inconsistent: possible mesh corruption")
        end
    end

    # Now build faces for cell-elements (will generate some duplicate faces)
    for i ∈ first_element:length(elements)
        face = Face2D(TI,TF)
        vertices = elements[i].vertices
        nvertices = length(vertices)
        for vi ∈ 1:nvertices
            vertex1 = elements[i].vertices[vi]
            # Check that vi+1 is in bounds - otherwise use the first vertex
            if vi+1 > nvertices
                vertex2 = elements[i].vertices[1]
            else
                vertex2 = elements[i].vertices[vi+1] 
            end
            if vertex1 < vertex2
                face = @set face.nodesID = SVector{2,TI}(vertex1, vertex2)
                push!(faces, face)
                continue
            elseif vertex1 > vertex2 
                face = @set face.nodesID = SVector{2,TI}(vertex2, vertex1)
                push!(faces, face)
                continue
            else
                throw("Boundary elements are inconsistent: possible mesh corruption")
            end
        end
    end
    unique!(faces) # remove duplicates
    return faces
end

function generate_cells(
    first_element, elements, faces::Vector{Face2D{TI, TF}}, nodes
    ) where {TI,TF}
    cells = Cell{TI,TF}[]
    for i ∈ first_element:length(elements)
        cell = Cell(TI,TF)
        nodesID = elements[i].vertices
        for nodeID ∈ nodesID
            push!(cell.nodesID, nodeID)
        end
        push!(cells, cell)
    end
    return cells
end

function face_cell_connectivity!(cells, faces::Vector{Face2D{TI, TF}}, nodes) where {TI,TF}
    for fID ∈ eachindex(faces)
        nodesID = faces[fID].nodesID
        node1 = nodesID[1]
        node2 = nodesID[2]
        neighbours1 = nodes[node1].neighbourCells
        neighbours2 = nodes[node2].neighbourCells
        ownerCells = TI[0,0] # Array for storing cells that have same nodes
        owner_counter = zero(TI) # counter to track which node has been allocated (2D only)
        # Loop to find nodes that share the same neighboring cells (only works for 2D faces)
        for neighbour1 ∈ neighbours1
            for neighbour2 ∈ neighbours2
                if neighbour1 == neighbour2
                    owner_counter += 1
                    ownerCells[owner_counter] = neighbour1
                end
            end
        end
        face = faces[fID]
        face = @set face.ownerCells = SVector{2, TI}(ownerCells)
        faces[fID] = face
        # If no face allocated in the second entry, it's a a boundary face -> don't add
        if ownerCells[2] != 0
            for ownerCell ∈ ownerCells
                push!(cells[ownerCell].facesID, fID)
            end
            push!(cells[ownerCells[1]].neighbours, ownerCells[2])
            push!(cells[ownerCells[2]].neighbours, ownerCells[1])
        end
    end
end
