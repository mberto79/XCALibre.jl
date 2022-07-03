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
    nodes = connect(points, elements, boundaryFaces)
    # nodes, cells, faces, boundaries = connect(points, elements, boundaryFaces)
    # preprocess!(nodes, faces, cells, boundaries)
    # mesh = Mesh.FullMesh(nodes, faces, cells, boundaries)
    end
    # println("Done! Execution time: ", @sprintf "%.6f" stats.time)
    # println("Mesh ready!")
    return nodes #mesh
end



function connect(points, elements, boundaryFaces)
    bfaces = total_boundary_faces(boundaryFaces)
    first_element = bfaces + 1
    nodes = generate_nodes(first_element, points, elements);
    faces = generate_faces(first_element, elements, nodes, boundaryFaces);
    # facesRaw, boundaries = generate_faces(elements, nodes, boundaryFaces);
    # faces = face_connectivity(facesRaw);
    # cells = generate_cells(points, elements, faces);
    # return nodes, cells, faces, boundaries
    return nodes, faces
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
            if vi+1 < nvertices
                vertex2 = elements[i].vertices[vi+1] 
            else
                vertex2 = elements[i].vertices[1]
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


    # # Collect boundary faces
    # faces = Mesh.OrderedFace[]
    # boundaryFaces = Mesh.Boundary[]
    # totalBoundaryFaces = 1
    # for (facei, boundaryFace) ∈ enumerate(boundaries)
    #     tempBoundary = Mesh.Boundary()
    #     tempBoundary.name = boundaryFace.name
    #     tempBoundary.ID = facei 
    #     tempBoundary.nFaces = length(boundaryFace.elements)
    #     tempBoundary.startFace = totalBoundaryFaces
    #     totalBoundaryFaces += tempBoundary.nFaces
    #     for element ∈ boundaryFace.elements
    #         nodesID = elements[element].vertices
    #         push!(tempBoundary.nodesID, nodesID)

    #         #= There is an offset of "firstElement" to ignore boundary faces. 
    #         In 3D will need loop over index e.g. nodesID[i]? =#
    #         offset = firstElement - 1 # needed to shift loop to match elements count
    #         for celli ∈ (nodes[nodesID[1]].neighbourCellsID)
    #             testCondition = sum(
    #                 (elements[celli .+ offset].vertices.==nodesID[1])
    #                 .+ 
    #                 (elements[celli .+ offset].vertices.==nodesID[2])
    #                 ) 
    #             if testCondition == 2
    #                 tempFace = Mesh.OrderedFace(nodesID, celli)
    #                 push!(faces, tempFace)
    #                 push!(tempBoundary.cellsID, celli)
    #                 break
    #             end
            
    #         end
    #     end
    #     push!(boundaryFaces, tempBoundary)
    # end
    
    # # Collect and order faces for all cells
    # for (celli, element) ∈ enumerate(elements[firstElement:end])
    #     nNodes = length(element.vertices)
    #     for nodei ∈ 1:nNodes 
    #         if  nodei == nNodes
    #             tempFace = Mesh.OrderedFace(
    #                 [element.vertices[1], element.vertices[nNodes]], celli
    #                 )
    #             push!(faces, tempFace)
    #             break
    #         end
    #         tempFace = Mesh.OrderedFace(
    #             [element.vertices[nodei], element.vertices[nodei+1]], celli
    #             )
    #         push!(faces, tempFace)
    #     end
    # end
    # # Assign array of facesID built from startFace and nFaces (may be removed?)
    # for i ∈ 1:length(boundaryFaces); Mesh.Boundary(boundaryFaces[i]); end
    return faces
end