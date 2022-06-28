export build_mesh

function generate_faces(elements, nodes, boundaries)
    # Define initial offset of elements to ignore (e.g boundary faces)
    firstElement = findfirst(x-> x>2, getproperty.(elements, :vertexCount))

    # Collect boundary faces
    faces = Mesh.OrderedFace[]
    boundaryFaces = Mesh.Boundary[]
    totalBoundaryFaces = 1
    for (facei, boundaryFace) ∈ enumerate(boundaries)
        tempBoundary = Mesh.Boundary()
        tempBoundary.name = boundaryFace.name
        tempBoundary.ID = facei 
        tempBoundary.nFaces = length(boundaryFace.elements)
        tempBoundary.startFace = totalBoundaryFaces
        totalBoundaryFaces += tempBoundary.nFaces
        for element ∈ boundaryFace.elements
            nodesID = elements[element].vertices
            push!(tempBoundary.nodesID, nodesID)

            #= There is an offset of "firstElement" to ignore boundary faces. 
            In 3D will need loop over index e.g. nodesID[i]? =#
            offset = firstElement - 1 # needed to shift loop to match elements count
            for celli ∈ (nodes[nodesID[1]].neighbourCellsID)
                testCondition = sum(
                    (elements[celli .+ offset].vertices.==nodesID[1])
                    .+ 
                    (elements[celli .+ offset].vertices.==nodesID[2])
                    ) 
                if testCondition == 2
                    tempFace = Mesh.OrderedFace(nodesID, celli)
                    push!(faces, tempFace)
                    push!(tempBoundary.cellsID, celli)
                    break
                end
            
            end
        end
        push!(boundaryFaces, tempBoundary)
    end
    
    # Collect and order faces for all cells
    for (celli, element) ∈ enumerate(elements[firstElement:end])
        nNodes = length(element.vertices)
        for nodei ∈ 1:nNodes 
            if  nodei == nNodes
                tempFace = Mesh.OrderedFace(
                    [element.vertices[1], element.vertices[nNodes]], celli
                    )
                push!(faces, tempFace)
                break
            end
            tempFace = Mesh.OrderedFace(
                [element.vertices[nodei], element.vertices[nodei+1]], celli
                )
            push!(faces, tempFace)
        end
    end
    # Assign array of facesID built from startFace and nFaces (may be removed?)
    for i ∈ 1:length(boundaryFaces); Mesh.Boundary(boundaryFaces[i]); end
    return faces, boundaryFaces
end

function first_next(orderedFaces, uindex)
    indexFirst = findfirst(x->x==view(orderedFaces, uindex)[1], orderedFaces)
    indexNextOut = findnext(x->x==view(orderedFaces, uindex)[1], orderedFaces,indexFirst+1)
        if indexNextOut === nothing
            indexNext = indexFirst
        else
            indexNext = indexNextOut
        end
    return Int32(indexFirst), Int32(indexNext)
end

# There is definitely room for improvement here!
function face_connectivity(facesRaw) # unfinished
    uniqueidx(v) = unique(i -> v[i], eachindex(v))
    orderedFaces = getproperty.(facesRaw,:ids)
    uniqueIndices = uniqueidx(orderedFaces)
    indexFirst::Int32 = 0
    indexNext::Int32 = 0
    # indexNextOut::Union{Nothing,Int32} = 0
    indexNextOut::Int32 = 0
    tempFaces = Mesh.Face[Mesh.Face() for _ ∈ 1:length(uniqueIndices)]
    @inbounds for (facei, uindex) ∈ enumerate(uniqueIndices)
        # tempFace = Mesh.Face()
        tempFace = tempFaces[facei]

        indexFirst, indexNext = first_next(orderedFaces, uindex)
        
        tempFace.nodesID = facesRaw[indexFirst].ids

        tempFace.ownerCells[1] = facesRaw[indexFirst].owner
        tempFace.ownerCells[2] = facesRaw[indexNext].owner
        tempFace.ID = facei
        # push!(faces, tempFace)
    end
    
    return tempFaces #faces 
end

function generate_nodes(points, elements)
     #=
    NOTE:
    This check (x-> x>2) does not work for 3D (new logic needed).
    Probably using more information about the element type
    from the UNV format documentation 
    =#
    # nodes = [Node(0.0) for _ = 1:length(points)]
    nodes = Node[]
    for i ∈ 1:length(points)
        # nodes[i].xyz = points[i].xyz
        point = points[i].xyz
        push!(nodes, Node(point))
    end
    # nodesID = Int32[]
    cellID = 0
    firstCell = findfirst(x-> x>2, getproperty.(elements, :vertexCount))
    for i ∈ firstCell:length(elements) 
            cellID += 1
            # nodesID = elements[i].vertices
            # for ID ∈ nodesID 
            for ID ∈ elements[i].vertices
                push!(nodes[ID].neighbourCellsID, cellID)
            end
    end
    return nodes
end

function generate_cells(points, elements, faces)
    #=
    NOTE:
    The "findfirst" check (e.g. x-> x>2) does not work for 3D (new logic needed).
    Probably using information about the element type from the UNV format documentation 
    =#
    cells = Mesh.Cell[]
    # cellID = 0
    firstCell = findfirst(x-> x>2, getproperty.(elements, :vertexCount))
    for i ∈ firstCell:length(elements) 
            # cellID += 1
            tempCell = Mesh.Cell()
            tempCell.ID = i - (firstCell - 1)
            tempCell.nodesID = elements[i].vertices
            tempCell.centre = midPoint(getproperty.(points[tempCell.nodesID], :xyz))
            push!(cells, tempCell) 
    end

    # Assign facesID that make up a cell
    faceNeighbourCells = [faces[i].ownerCells for i ∈ 1:length(faces)]
    for (facei, neighbourCellIDs) ∈ enumerate(faceNeighbourCells)
        push!(cells[neighbourCellIDs[1]].facesID, facei)
        if neighbourCellIDs[1] != neighbourCellIDs[2]
            push!(cells[neighbourCellIDs[2]].facesID, facei)
        end
    end

    # Find and assign neighbour cells (loop over facesID and their owners)
    for (celli, cell) ∈ enumerate(cells)
        for faceID ∈ cell.facesID
            for ownerCell ∈ faces[faceID].ownerCells
                if ownerCell != celli
                    push!(cells[celli].neighbours, ownerCell)
                end
            end
        end
    end

    return cells
end # Function end

function connect(points, elements, boundaryFaces)
    # points, elements, boundaries = load(meshFile);
    nodes = generate_nodes(points, elements);
    facesRaw, boundaries = generate_faces(elements, nodes, boundaryFaces);
    faces = face_connectivity(facesRaw);
    cells = generate_cells(points, elements, faces);
    return nodes, cells, faces, boundaries
end

function scalePoints!(points::Vector{Point}, scaleFactor)
    for point ∈ points
        point.xyz = point.xyz*scaleFactor
    end
    return points
end

function preprocess!(nodes, faces, cells, boundaries)
    for n ∈ nodes
        Mesh.nodeWeight!(n, cells)
    end

    for f ∈ faces
        Mesh.faceCentre!(f, nodes)
        Mesh.faceTangent(f, nodes)
        Mesh.faceArea!(f)
        ftn = Mesh.faceUnitTangent(f)
        Mesh.faceNormal!(f, ftn)
        Mesh.delta!(f, cells)
        Mesh.faceWeight!(f, cells)
    end

    # Correct delta at boundary faces
    for boundary ∈ boundaries
        Mesh.delta!(boundary, faces, cells)
    end

    for c ∈ cells
        Mesh.cellNormalsCheck!(c, faces)
        Mesh.cellVolume!(c, faces)
    end
end

function build_mesh(meshFile; scaleFactor=1, TI=Int64, TF=Float64)
    stats = @timed begin
    println("Loading mesh...")
    points, elements, boundaryFaces = load(meshFile, TI, TF);
    println("File read successfully")
    if scaleFactor != 1
        scalePoints!(points, scaleFactor)
    end
    println("Generating mesh connectivity...")
    nodes, cells, faces, boundaries = connect(points, elements, boundaryFaces)
    preprocess!(nodes, faces, cells, boundaries)
    # mesh = Mesh.FullMesh(nodes, faces, cells, boundaries)
    end
    println("Done! Execution time: ", @sprintf "%.6f" stats.time)
    println("Mesh ready!")
    return nothing #mesh
end