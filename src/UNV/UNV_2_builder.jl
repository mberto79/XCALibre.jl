export build_mesh

function build_mesh(meshFile; scale=1.0, TI=Int64, TF=Float64)
    stats = @timed begin
    println("Loading mesh...")
    points, elements, boundaryElements = load(meshFile, TI, TF);
    println("File read successfully")
    if scale != one(typeof(scale))
        scalePoints!(points, scale)
    end
    println("Generating mesh...")
    bfaces = total_boundary_faces(boundaryElements)
    cells, faces, nodes, boundaries = generate(points, elements, boundaryElements, bfaces)
    println("Building connectivity...")
    connect!(cells, faces, nodes, boundaries, bfaces)
    # preprocess!(nodes, faces, cells, boundaries)
    # mesh = Mesh.FullMesh(nodes, faces, cells, boundaries)
    end
    println("Done! Execution time: ", @sprintf "%.6f" stats.time)
    println("Mesh ready!")
    return cells, faces, nodes, boundaries
end

function generate(points::Vector{Point{TF}}, elements, boundaryElements, bfaces) where TF
    first_element = bfaces + 1
    nodes = generate_nodes(first_element, elements, points)
    faces = generate_faces(first_element, elements, TF)
    cells = generate_cells(first_element, elements, TF)
    boundaries = generate_boundaries(boundaryElements, elements)
    return cells, faces, nodes, boundaries
end

function connect!(cells, faces, nodes, boundaries, bfaces)
    face_cell_connectivity!(cells, faces, nodes)
    boundary_connectivity!(boundaries, faces, bfaces)
end

function process_geometry!(cells, faces, nodes, boundaries)
    nothing
end

# GENERATION FUNCTIONS

function scalePoints!(points::Vector{Point{TF}}, scaleFactor) where TF
    for i ∈ eachindex(points)
        point = points[i]
        point = @set point.xyz = point.xyz*scaleFactor
        points[i] = point
    end
end

function total_boundary_faces(boundaryElements::Vector{BoundaryLoader{TI}}) where TI
    sum = zero(TI)
    @inbounds for boundary ∈ boundaryElements
        sum += length(boundary.elements)
    end
    return sum
end

function generate_nodes(first_element, elements, points::Vector{Point{TF}}) where TF
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

function generate_faces(first_element, elements::Vector{Element{TI}}, TF) where {TI}
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

function generate_cells(first_element, elements::Vector{Element{TI}}, TF) where {TI}
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

function generate_boundaries(
    boundaryElements::Vector{BoundaryLoader{TI}}, elements
    ) where TI
    boundaries = Boundary{TI}[]
    for boundaryElement ∈ boundaryElements
        name = Symbol(boundaryElement.name)
        boundary = Boundary(name, TI[], TI[], TI[])
        for elementID ∈ boundaryElement.elements
            nodesID = elements[elementID].vertices
            push!(boundary.nodesID, nodesID...)
            unique!(boundary.nodesID)
        end
        push!(boundaries, boundary)
    end
    return boundaries
end

# CONNECTIVITY FUNCTIONS

function face_cell_connectivity!(cells, faces::Vector{Face2D{TI, TF}}, nodes) where {TI,TF}
    ownerCells = TI[0,0] # Array for storing cells that have same nodes
    for fID ∈ eachindex(faces)
        ownerCells .= zero(TI)
        nodesID = faces[fID].nodesID
        node1 = nodesID[1]
        node2 = nodesID[2]
        neighbours1 = nodes[node1].neighbourCells
        neighbours2 = nodes[node2].neighbourCells
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
        else
            # for consistency make ownerCells equal for boundary faces
            face = faces[fID]
            face = @set face.ownerCells = SVector{2, TI}(ownerCells[1], ownerCells[1])
            faces[fID] = face
        end
    end
end

function boundary_connectivity!(
    boundaries::Vector{Boundary{TI}}, faces, bfaces
    ) where TI
    facedef = SVector{2,TI}(0,0)
    for boundary ∈ boundaries 
        nodesID = boundary.nodesID
        for i ∈ 2:length(nodesID)
            id1 = nodesID[i-1]
            id2 = nodesID[i]
            if id1 < id2 
                facedef = SVector{2,TI}(id1,id2)
            else
                facedef = SVector{2,TI}(id2,id1)
            end
            for fID ∈ 1:bfaces 
                face = faces[fID]
                if facedef == face.nodesID
                    push!(boundary.facesID, fID)
                    push!(boundary.cellsID, face.ownerCells[1])
                end
            end
        end
    end
end
