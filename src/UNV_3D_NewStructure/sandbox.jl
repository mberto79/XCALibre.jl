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
    nodes=Node[]
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

face_nodes=generate_face_nodes(faces)


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

generate_cell_nodes(volumes)

function generate_cell_faces(faces)
    cell_faces=[]
    for n=1:length(faces)
        #for i=1:4
            push!(cell_faces,faces[n].faceindex)
        #end
    end
    return cell_faces
end

generate_cell_faces(faces)

#Nodes Range
function generate_cell_nodes_range(volumes)
    cell_nodes_range=UnitRange(0,0)
    for i=1:length(volumes)
        cell_nodes_range=UnitRange(volumes[i].volumes[1],volumes[i].volumes[end])
    end
    return cell_nodes_range
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
    for i=1:length(face_nodes)
        face_nodes_range=UnitRange((3*i-2),3*i)
    end
    return face_nodes_range
end

face_nodes_range=generate_face_nodes_range(faces)

#Faces Range
function generate_faces_range(volumes,faces)
    cell_faces_range=UnitRange(0,0)
    @inbounds for i=1:length(volumes)
        cell_faces_range=UnitRange(faces[(4*i)-3].faceindex,faces[4*i].faceindex)
    end
    return cell_faces_range
end

cell_faces_range=generate_faces_range(volumes,faces)

#Face Area

function generate_face_area(nodes,faces)
    face_area=0
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
    end
    #face_area
    return face_area
end

generate_face_area(nodes,faces)
#Face centre

function centre_of_face(nodes,faces)
    x_centre=0
    y_centre=0
    z_centre=0
    centre=0
    for i=1:length(faces)
        n1=faces[i].faces[1]
        n2=faces[i].faces[2]
        n3=faces[i].faces[3]
        x_centre = (nodes[n1].coords[1] + nodes[n2].coords[1] + nodes[n3].coords[1]) / 3
        y_centre = (nodes[n1].coords[2]+nodes[n2].coords[2]+nodes[n3].coords[2]) / 3
        z_centre = (nodes[n1].coords[3]+nodes[n2].coords[3]+nodes[n3].coords[3]) / 3
        centre=SVector(x_centre,y_centre,z_centre)
    end
    return centre
end

centre_of_face(nodes,faces)

#Cell centre

function centre_of_cell(nodes,volumes)
    x_centre=0
    y_centre=0
    z_centre=0
    centre=0
    @inbounds for i=1:length(volumes)
        n1=volumes[i].volumes[1]
        n2=volumes[i].volumes[2]
        n3=volumes[i].volumes[3]
        n4 =volumes[i].volumes[4]
        x_centre = (nodes[n1].coords[1]+nodes[n2].coords[1]+nodes[n3].coords[1]+nodes[n4].coords[1]) / 4
        y_centre = (nodes[n1].coords[2]+nodes[n2].coords[2]+nodes[n3].coords[2]+nodes[n4].coords[2]) / 4
        z_centre = (nodes[n1].coords[3]+nodes[n2].coords[3]+nodes[n3].coords[3]+nodes[n4].coords[3]) / 4
        centre=SVector{3,Float64}(x_centre,y_centre,z_centre)
    end
    return centre #(x_centre, y_centre, z_centre)
end
centre=centre_of_cell(nodes,volumes)

#Cell Volume
function volume_of_cell(nodes,volumes)
    AB=0
    AC=0
    AD=0
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
    end
    return abs(dot(AB, cross(AC, AD))) / 6
end

volume=volume_of_cell(nodes,volumes)

#Face Normals
function face_normal(nodes,faces)
    cross_product=0
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
    end
    return normalize(cross_product)
end
face_normal(nodes,faces)

#Generate function

function generate(points,faces,volumes)
    nodes=generate_nodes(points)
    face_nodes=generate_face_nodes(faces)
    cell_nodes=generate_cell_nodes(volumes)
    cell_faces=generate_cell_faces(faces)

    return nodes,face_nodes,cell_nodes,cell_faces
end


cells=Cell[]
mesh=Mesh3[]
push!(cells,Cell(centre,volume,cell_nodes_range,cell_faces_range))