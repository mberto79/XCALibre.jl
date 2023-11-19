export build_mesh3D

function build_mesh3D(unv_mesh; scale=1.0, TI=Int64, TF=Float64)
    stats = @timed begin
    println("Loading mesh...")
    points,edges,faces,volumes,boundaryElements = load(unv_mesh, TI, TF);
    println("File read successfully")
    if scale != one(typeof(scale))
        scalePoints!(points, scale)
    end
    println("Generating mesh...")
    bfaces = total_boundary_faces(boundaryElements)
    cells, faces, nodes, boundaries = generate(points, elements, boundaryElements, bfaces)
    println("Building connectivity...")
    #connect!(cells, faces, nodes, boundaries, bfaces)
    mesh = Mesh3(cells,cell_nodes,cell_faces,cell_neighbours,cell_nsign,faces,face_nodes,boundaries,nodes)
    #process_geometry!(mesh)
    end
    println("Done! Execution time: ", @sprintf "%.6f" stats.time)
    println("Mesh ready!")
    return mesh
end

function generate(points::Vector{Point{TF}},boundaryElements) where TF
    nodes=generate_nodes(points)
    faces=generate_faces(faces)
    cells=generate_cells(volumes)
    boundaries=generate_boundaries(boundaryElements)
    return cells,faces,nodes,boundaries
end

function process_geometry!(mesh::Mesh3{TI,TF}) where {TI,TF}
    (; cells, faces, nodes) = mesh
    cell_centres!(cells, nodes)
    cell_centres!(faces, nodes)
    geometry!(mesh)
end

function scalePoints!(points::Vector{Point{TF}}, scaleFactor) where TF
    @inbounds for i ∈ eachindex(points)
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

function generate_nodes(volumes, points::Vector{Point{TF}}) where TF
    nodes = Node{Int64, TF}[]
    @inbounds for i ∈ 1:length(points)
        point = points[i].xyz
        push!(nodes, Node(point))
    end
    cellID = 0 # counter for cells
    @inbounds for i ∈ 1:length(volumes) 
            cellID += 1
            @inbounds for nodeID ∈ volumes[i].volumes
                push!(nodes[nodeID].neighbourCells, cellID)
            end
    end
    return nodes
end

function generate_faces(faces)
    faces1=Face3D{TI,TF}[]

    @inbounds for i=1:length(faces)
        face = Face3D(TI,TF)
        #vertex1 = faces[elementi].faces[1]
        #vertex2 = faces[elementi].faces[2]
        #vertex3 = faces[elementi].faces[3]
        faceindex1=faces[elementi].faceindex[1]
        faceindex2=faces[elementi+1].faceindex[1]
        face=@set face.nodes_range=UnitRange(faceindex1,faceindex2)
    end



end