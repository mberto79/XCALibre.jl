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

function scalePoints!(points::Vector{Point{TF}}, scaleFactor) where TF
    for point ∈ points
        point.xyz = point.xyz*scaleFactor
    end
end

function connect(points, elements, boundaryElements)
    bfaces = total_boundary_faces(boundaryElements)
    first_element = bfaces + 1
    nodes = generate_nodes(first_element, points, elements);
    # facesRaw, boundaries = generate_faces(elements, nodes, boundaryFaces);
    # faces = face_connectivity(facesRaw);
    # cells = generate_cells(points, elements, faces);
    # return nodes, cells, faces, boundaries
    return nodes
end


# Lower level functions
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
   # nodesID = Int32[]
   cellID = 0
#    firstCell = findfirst(x-> x>2, getproperty.(elements, :vertexCount))
   for i ∈ first_element:length(elements) 
           cellID += 1
           # nodesID = elements[i].vertices
           # for ID ∈ nodesID 
           for ID ∈ elements[i].vertices
            #    push!(nodes[ID].neighbourCellsID, cellID)
           end
   end
   return nodes
end