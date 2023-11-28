unv_mesh="src/UNV_3D_NewStructure/tetra_singlecell.unv"

using StaticArrays
using LinearAlgebra
using Setfield
using Adapt
include("UNV_3D_NS_Types.jl")


#UNV Reader New Structure

#For Tetrahedral Cell Types

pointindx=0
elementindx=0
volumeindx=0
boundaryindx=0

#Defining Arrays with Structs
points=Point[]
edges=Edge[]
faces=Face[]
volumes=Volume[]
boundaryElements=BoundaryElement[]
elements=Element[]

#Defining Arrays for data collection
#Points
point=[]
pointindex=[]

#Vertices
edge=[]
edgeCount=[]
edgeindex=[]

#Faces
face=[]
faceindex=[]
faceCount=[]

#Volumes
volume=[]
volumeindex=[]
volumeCount=[]

#bc
boundary=[]
boundarys=[]
boundaryindex=[]
currentBoundary=0
boundaryNumber=0


#Splits UNV file into sections
for (indx,line) in enumerate(eachline(unv_mesh))
    sline=split(line)

    #Points = 2411
    if sline[1]=="2411" && length(sline)==1
        pointindx=indx
    end
    #Elements = 2412
    if sline[1]=="2412" && length(sline)==1
        elementindx=indx
    end
    #BC=2467
    if sline[1]== "2467" && length(sline)==1
        boundaryindx=indx
    end
end

#Check
pointindx
elementindx
boundaryindx

#Extracting Data from UNV file
for (indx,line) in enumerate(eachline(unv_mesh))
    sline=split(line)
    #Points
    if indx>pointindx && indx<elementindx && length(sline)==4 && parse(Float64,sline[4])==11
        pointindex=parse(Int64,sline[1])
        continue
    end

    if length(sline)==3 && indx>pointindx && indx<elementindx
        point=[parse(Float64,sline[i]) for i=1:length(sline)]
        push!(points,Point(SVector{3,Float64}(point)))
        continue
    end

    #Elements
    #Vertices
    if length(sline)==6 && parse(Int64,sline[end])==2
        edgeCount=parse(Int,sline[end])
        edgeindex=parse(Int,sline[1])
        continue
    end

    if length(sline)==2 && indx>elementindx
        edge=[parse(Int,sline[i]) for i=1:length(sline)]
        push!(edges,Edge(edgeindex,edgeCount,edge))
        push!(elements,Element(edgeindex,edgeCount,edge))
        continue
    end

    #Faces
    if length(sline)==6 && parse(Int64,sline[end])==3
        faceCount=parse(Int,sline[end])
        faceindex=parse(Int,sline[1])
        continue
    end

    if length(sline)==3 && indx>elementindx && parse(Int,sline[end]) ≠ 1
        face=[parse(Int,sline[i]) for i=1:length(sline)]
        push!(faces,Face(faceindex-edgeindex,faceCount,face))
        push!(elements,Element(faceindex,faceCount,face))
        continue
    end

    #Volumes
    if length(sline)==6 && parse(Int,sline[2])==111
        volumeCount=parse(Int,sline[end])
        volumeindex=parse(Int,sline[1])
        volumeindx=indx
        continue
    end

    if length(sline)==4 && indx>elementindx
        volume=[parse(Int,sline[i]) for i=1:length(sline)]
        push!(volumes,Volume(volumeindex-faceindex,volumeCount,volume))
        push!(elements,Element(volumeindex,volumeCount,volume))
        continue
    end

    #Boundary
    if length(sline)==1 && indx>boundaryindx && typeof(tryparse(Int64,sline[1]))==Nothing
        boundaryindex=sline[1]
        currentBoundary=currentBoundary+1
        newBoundary=BoundaryElement(0)
        push!(boundaryElements, newBoundary)
        boundaryNumber=boundaryNumber+1
        boundaryElements[currentBoundary].boundaryNumber=currentBoundary
        boundaryElements[currentBoundary].name=boundaryindex
        continue
    end

    if length(sline)==8 && indx>boundaryindx && parse(Int64,sline[2])!=0
        boundary=[parse(Int,sline[i]) for i=1:length(sline)]
        push!(boundarys,(boundaryindex,boundary))
        push!(boundaryElements[currentBoundary].elements,parse(Int64,sline[2]))
        push!(boundaryElements[currentBoundary].elements,parse(Int64,sline[6]))
        continue
    end

end

pointindex
point
points

edgeCount
edgeindex
edge
edges

faceCount
faceindex
face
faces

volumeindx
volumeCount
volumeindex
volume
volumes


boundaryindex
boundaryindx
boundary
boundarys

boundaryElements
boundaryNumber
currentBoundary
elements



#Generate nodes
function generate_nodes(points)
    nodes=Node{Int64,Float64}[]
    @inbounds for i ∈ 1:length(points)
        point=points[i].xyz
        push!(nodes,Node(point))
    end
    return nodes
end

nodes=generate_nodes(points)




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

face_nodes=Vector{Int}(generate_face_nodes(faces))


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

cell_nodes=Vector{Int}(generate_cell_nodes(volumes))

function generate_cell_faces(faces)
    cell_faces=[]
    for n=1:length(faces)
        #for i=1:4
            push!(cell_faces,faces[n].faceindex)
        #end
    end
    return cell_faces
end

cell_faces=Vector{Int}(generate_cell_faces(faces))

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

cell_nodes_range=generate_cell_nodes_range(volumes)

#function generate_face_nodes_range(faces)
  #  face_nodes_range=UnitRange(0,0)
   # for i=1:length(faces)
   #     face_nodes_range=UnitRange(faces[i].faces[1],faces[i].faces[end])
   # end
   # return face_nodes_range
#end

#generate_face_nodes_range(faces)

function generate_face_nodes_range(face_nodes)
    face_nodes_range=UnitRange(0,0)
    store=[]
    for i=1:length(face_nodes)
        face_nodes_range=UnitRange((3*i-2),3*i)
        push!(store,face_nodes_range)
    end
    return store
end

face_nodes_range=generate_face_nodes_range(faces)

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

cell_faces_range=generate_faces_range(volumes,faces)

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

faces_area=generate_face_area(nodes,faces)
#Face centre

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

centre_of_faces=centre_of_face(nodes,faces)

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
centre_of_cells=centre_of_cell(nodes,volumes)


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

volume_of_cells=volume_of_cell(nodes,volumes)

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
faces_normal=face_normal(nodes,faces)

#Generate function

function generate(points,faces,volumes)
    nodes=generate_nodes(points)
    face_nodes=generate_face_nodes(faces)
    cell_nodes=generate_cell_nodes(volumes)
    cell_faces=generate_cell_faces(faces)

    return nodes,face_nodes,cell_nodes,cell_faces
end


#Generate Boundary
function generate_boundaries(boundaryElements)
    boundaries=Boundary{Int64}[]
    for i=1:length(boundaryElements)
        push!(boundaries,Boundary(Symbol(boundaryElements[i].name),boundaryElements[i].elements,Vector{Int}(undef,1)))
    end
    return boundaries
end

boundaries=generate_boundaries(boundaryElements)




#push!(cells,Cell(centre,volume,cell_nodes_range,cell_faces_range))

#Generate cells
function generate_cells(volumes)
    cells=Cell{Int64,Float64}[]
    for i=1:length(volumes)
        push!(cells,Cell(centre_of_cells[i],volume_of_cells[i],cell_nodes_range[i],cell_faces_range[i]))
    end
    return cells
end

cells=generate_cells(volumes)

#Generate Faces



function generate_faces(faces)
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

faces=generate_faces(faces)

#Generate Mesh3

mesh=Mesh3[]

cell_neighbours=Vector{Int}(undef,1)
cell_nsign=Vector{Int}(undef,1)

push!(mesh,Mesh3(cells,cell_nodes,cell_faces,cell_neighbours,cell_nsign,faces,face_nodes,boundaries,nodes))
mesh
store_nodes=zeros(length(mesh[1].nodes)*3)
for i=1:length(mesh[1].nodes)
    store_nodes[(3*i-2):(3*i)]=mesh[1].nodes[i].coords
end
store_nodes
join(store_nodes," ")

LinRange(1,length(mesh[1].nodes),length(mesh[1].nodes))

store_faces=zeros(Int32,length(mesh[1].faces))
for i=1:length(mesh[1].faces)
    store_faces[i]=length(mesh[1].faces[1].nodes_range)*i
end
store_faces

mesh[1].face_nodes


function generate(points,faces,volumes,boundaryElements)
    nodes=generate_nodes(points)
    faces=generate_faces(faces)
    cells=generate_cells(volumes)
    boundaries=generate_boundaries(boundaryElements)
    return nodes,faces,cells,boundaries
end

generate(points,faces,volumes,boundaryElements)