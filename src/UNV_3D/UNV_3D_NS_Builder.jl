export build_mesh3D

function build_mesh3D(unv_mesh)
    stats= @timed begin
    println("Loading UNV File...")
    points,edges,faces,volumes,boundaryElements=load(unv_mesh)
    println("File Read Successfully")
    println("Generating Mesh...")
    nodes=generate_nodes(points)
    face_nodes=Vector{Int}(generate_face_nodes(faces))
    cell_nodes=Vector{Int}(generate_cell_nodes(volumes))
    cell_faces=Vector{Int}(generate_cell_faces(faces))
    cell_nodes_range=generate_cell_nodes_range(volumes)
    face_nodes_range=generate_face_nodes_range(faces)
    cell_faces_range=generate_faces_range(volumes,faces)
    faces_area=generate_face_area(nodes,faces)
    centre_of_faces=centre_of_face(nodes,faces)
    centre_of_cells=centre_of_cell(nodes,volumes)
    volume_of_cells=volume_of_cell(nodes,volumes)
    faces_normal=face_normal(nodes,faces)
    boundaries=generate_boundaries(boundaryElements)
    cells=generate_cells(volumes,centre_of_cells,volume_of_cells,cell_nodes_range,cell_faces_range)
    faces=generate_faces(faces,face_nodes_range,centre_of_faces,faces_normal,faces_area)

    mesh=Mesh3[]

    cell_neighbours=Vector{Int}(undef,1)
    cell_nsign=Vector{Int}(undef,1)

    push!(mesh,Mesh3(cells,cell_nodes,cell_faces,cell_neighbours,cell_nsign,faces,face_nodes,boundaries,nodes))

    end
    println("Done! Execution time: ", @sprintf "%.6f" stats.time)
    println("Mesh ready!")
    return mesh
end


#function build_mesh3D(unv_mesh)
    #stats = @timed begin
    #println("Loading mesh...")
    #points,edges,faces,volumes,boundaryElements = load(unv_mesh);
    #println("File read successfully")

    #if scale != one(typeof(scale))
     #   scalePoints!(points, scale)
    #end

    #cells, faces, nodes, boundaries = generate(points,faces,volumes,boundaryElements)

    #println("Generating geometry...")
    #face_nodes,face_nodes_range,faces_area,centre_of_face,faces_normal=process_faces(faces,nodes)
    #cell_nodes,cell_faces,cell_nodes_range,cell_faces_range,centre_of_cells,volume_of_cells=process_cells(faces,volumes)
    #println("Geometry processed!")

    #println("Generating mesh...")

    

    #println("Building connectivity...")
    #connect!(cells, faces, nodes, boundaries, bfaces)
    #mesh = Mesh3()
    #process_geometry!(mesh)

    #mesh=Mesh3(cells,cell_nodes,cell_faces,cell_neighbours,cell_nsign,faces,face_nodes,boundaries,nodes)
    #println("Mesh Generated!")

    #end
    #println("Done! Execution time: ", @sprintf "%.6f" stats.time)
    #println("Mesh ready!")
    #return mesh
#end

#Generate
function generate(points,faces,volumes,boundaryElements)
    nodes=generate_nodes(points)
    faces=generate_faces(faces)
    cells=generate_cells(volumes)
    boundaries=generate_boundaries(boundaryElements)
    return nodes,faces,cells,boundaries
end

function process_faces(faces,nodes)
    face_nodes=generate_face_nodes(faces)
    face_nodes_range=generate_face_nodes_range(faces)
    faces_area=generate_face_area(nodes,faces)
    centre_of_face=centre_of_face(nodes,faces)
    faces_normal=face_normal(nodes,faces)
    return face_nodes,face_nodes_range,faces_area,centre_of_face,faces_normal
end

function process_cells(faces,volumes)
    cell_nodes=generate_cell_nodes(volumes)
    cell_faces=generate_cell_faces(faces)
    cell_nodes_range=generate_cell_nodes_range(volumes)
    cell_faces_range=generate_faces_range(volumes,faces)
    centre_of_cells=centre_of_cell(nodes,volumes)
    volume_of_cells=volume_of_cell(nodes,volumes)
    return cell_nodes,cell_faces,cell_nodes_range,cell_faces_range,centre_of_cells,volume_of_cells
end

#Generate nodes
function generate_nodes(points)
    nodes=Node{Int64,Float64}[]
    @inbounds for i ∈ 1:length(points)
        point=points[i].xyz
        push!(nodes,Node(point))
    end
    return nodes
end

#Generate Faces
function generate_face_nodes(faces)
    face_nodes=[] #Vector(undef,length(faces)*3)
    for n=1:length(faces)
        for i=1:3
            push!(face_nodes,faces[n].faces[i])
        end
    end
    return face_nodes
end

#Generate cells
function generate_cell_nodes(volumes)
    cell_nodes=[]
    for n=1:length(volumes)
        for i=1:4
            push!(cell_nodes,volumes[n].volumes[i])
        end
    end
    return cell_nodes
end

function generate_cell_faces(faces)
    cell_faces=[]
    for n=1:length(faces)
        #for i=1:4
            push!(cell_faces,faces[n].faceindex)
        #end
    end
    return cell_faces
end

#Nodes Range
function generate_cell_nodes_range(volumes)
    cell_nodes_range=UnitRange(0,0)
    store=[]
    for i=1:length(volumes)
        cell_nodes_range=UnitRange(volumes[i].volumes[1],volumes[i].volumes[end])
        push!(store,cell_nodes_range)
    end
    return store
end

function generate_face_nodes_range(face_nodes)
    face_nodes_range=UnitRange(0,0)
    store=[]
    for i=1:length(face_nodes)
        face_nodes_range=UnitRange((3*i-2),3*i)
        push!(store,face_nodes_range)
    end
    return store
end

#Faces Range
function generate_faces_range(volumes,faces)
    cell_faces_range=UnitRange(0,0)
    store=[]
    @inbounds for i=1:length(volumes)
        cell_faces_range=UnitRange(faces[(4*i)-3].faceindex,faces[4*i].faceindex)
        push!(store,cell_faces_range)
    end
    return store
end

#Face Area
function generate_face_area(nodes,faces)
    face_area=0
    store=[]
    @inbounds for i=1:length(faces)
        n1=faces[i].faces[1]
        n2=faces[i].faces[2]
        n3=faces[i].faces[3]
        A=(nodes[n1].coords)
        B=(nodes[n2].coords)
        C=(nodes[n3].coords)

        AB=[B[1]-A[1],B[2]-A[2],B[3]-A[3]]
        AC=[C[1]-A[1],C[2]-A[2],C[3]-A[3]]

        face_area=0.5*norm(cross(AB,AC))
        push!(store,face_area)
    end
    #face_area
    return store
end

function centre_of_face(nodes,faces)
    x_centre=0
    y_centre=0
    z_centre=0
    centre=0
    store=[]
    for i=1:length(faces)
        n1=faces[i].faces[1]
        n2=faces[i].faces[2]
        n3=faces[i].faces[3]
        x_centre = (nodes[n1].coords[1] + nodes[n2].coords[1] + nodes[n3].coords[1]) / 3
        y_centre = (nodes[n1].coords[2]+nodes[n2].coords[2]+nodes[n3].coords[2]) / 3
        z_centre = (nodes[n1].coords[3]+nodes[n2].coords[3]+nodes[n3].coords[3]) / 3
        centre=SVector(x_centre,y_centre,z_centre)
        push!(store,centre)
    end
    return store
end

#Cell centre
function centre_of_cell(nodes,volumes)
    x_centre=0
    y_centre=0
    z_centre=0
    centre=0
    centre_store=[]
    @inbounds for i=1:length(volumes)
        n1=volumes[i].volumes[1]
        n2=volumes[i].volumes[2]
        n3=volumes[i].volumes[3]
        n4 =volumes[i].volumes[4]
        x_centre = (nodes[n1].coords[1]+nodes[n2].coords[1]+nodes[n3].coords[1]+nodes[n4].coords[1]) / 4
        y_centre = (nodes[n1].coords[2]+nodes[n2].coords[2]+nodes[n3].coords[2]+nodes[n4].coords[2]) / 4
        z_centre = (nodes[n1].coords[3]+nodes[n2].coords[3]+nodes[n3].coords[3]+nodes[n4].coords[3]) / 4
        centre=SVector{3,Float64}(x_centre,y_centre,z_centre)
        push!(centre_store,centre)
    end
    return centre_store #(x_centre, y_centre, z_centre)
end

#Cell Volume
function volume_of_cell(nodes,volumes)
    AB=0
    AC=0
    AD=0
    volume_store=[]
    for i=1:length(volumes)
        n1=volumes[i].volumes[1]
        n2=volumes[i].volumes[2]
        n3=volumes[i].volumes[3]
        n4=volumes[i].volumes[4]
        A=(nodes[n1].coords)
        B=(nodes[n2].coords)
        C=(nodes[n3].coords)
        D=(nodes[n4].coords)
        AB=[B[1]-A[1],B[2]-A[2],B[3]-A[3]]
        AC=[C[1]-A[1],C[2]-A[2],C[3]-A[3]]
        AD=[D[1]-A[1],D[2]-A[2],D[3]-A[3]]
        volume=abs(dot(AB, cross(AC, AD))) / 6
        push!(volume_store,volume)
    end
    return volume_store
end

#Face Normals
function face_normal(nodes,faces)
    cross_product=0
    normal=0
    store=[]
    for i=1:length(faces)
        n1=faces[i].faces[1]
        n2=faces[i].faces[2]
        n3=faces[i].faces[3]
        A=(nodes[n1].coords)
        B=(nodes[n2].coords)
        C=(nodes[n3].coords)
        AB=[B[1]-A[1],B[2]-A[2],B[3]-A[3]]
        AC=[C[1]-A[1],C[2]-A[2],C[3]-A[3]]
        cross_product = cross(AB, AC)
        normal=SVector{3,Float64}(normalize(cross_product))
        push!(store,normal)
    end
    return store
end

#Generate Boundary
function generate_boundaries(boundaryElements)
    boundaries=Boundary{Int64}[]
    for i=1:length(boundaryElements)
        push!(boundaries,Boundary(Symbol(boundaryElements[i].name),boundaryElements[i].elements,Vector{Int}(undef,1)))
    end
    return boundaries
end

#Generate cells
function generate_cells(volumes,centre_of_cells,volume_of_cells,cell_nodes_range,cell_faces_range)
    cells=Cell{Int64,Float64}[]
    for i=1:length(volumes)
        push!(cells,Cell(centre_of_cells[i],volume_of_cells[i],cell_nodes_range[i],cell_faces_range[i]))
    end
    return cells
end

function generate_faces(faces,face_nodes_range,centre_of_faces,faces_normal,faces_area)
    faces3D=Face3D{Int64,Float64}[]

    ownerCells=SVector(1,2)
    e=SVector{3,Float64}(1,2,3)
    delta=0.1
    weight=0.1

    for i=1:length(faces)
        push!(faces3D,Face3D(face_nodes_range[i],ownerCells,centre_of_faces[i],faces_normal[i],e,faces_area[i],delta,weight))
    end
    return faces3D
end