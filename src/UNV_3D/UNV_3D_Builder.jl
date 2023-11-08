export build_3Dmesh

function build_3Dmesh(meshFile; scale=1.0, TI=Int64,TF=Float64)
    stats = @timed begin
        println("Loading mesh...")
        points,vertices,faces,groups,boundaryElements,elements=load(meshFile,TI,TF)
        println("File Read successfully")
        if scale !=one(typeof(scale))
            scalePoints!(points,scale)
        end
        println("Generating Mesh...")
        bfaces=
        cells,faces,nodes,boundaries=generate()
        println("Building connectivity...")
        connect!()
        mesh=
        process_geometry!(mesh)
    end
    println("Done! Generation Time: ",@sprintf "%.6f" stats.time)
    println("Mesh Generated!")
    return mesh
end

function generate(points,vertices,faces,groups,boundaryElements,bfaces)
    first_element =
    nodes=
    faces=
    cells=
    boundaries
    return cells,faces,nodes,boundaries
end

function connect!(cells,faces,nodes,boundaries,bfaces)
    face_cell_connectivity!()
    boundary_connectivity!()
end

function process_geometry!()
    (cells,faces,nodes)=mesh 
    cell_centres!()
    cell_centres!()
    geometry!()
end

#Support FUNCTIONS

function scalePoints!(points::Vector{Point{TF}}, scaleFactor) where TF
    @inbounds for i ∈ eachindex(points)
        point = points[i]
        point = @set point.xyz = point.xyz*scaleFactor
        points[i] = point
    end
end

function total_boundary_faces(boundaryElements) where TI
    sum=zero(TI)
    @inbounds for boundary ∈ boundaryElements
        sum += length(boundary.elements)
    end
    return sum
end

function first_2D_elements(vertices)
    element_index=zero(Int64)
    element_index=length(vertices)
    return element_index
end

#Generate FUNCTIONS

function generate_nodes(elements,points)
    nodes = Node[]
   @inbounds for i ∈ 1:length(points)
       point1 = points[i].xyz
       push!(nodes, Node(point1))
   end
   cellID = 0 # counter for cells
   @inbounds for i ∈ 1:length(elements) 
           cellID += 1
           @inbounds for nodeID ∈ elements[i].elements
               push!(nodes[nodeID].neighbourCells, cellID)
           end
   end
   return nodes
end

