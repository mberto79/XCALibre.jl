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

mesh=Mesh3[]
#Generate nodes
nodes=Node[]
@inbounds for i ∈ 1:length(points)
    point=points[i].xyz
    push!(nodes,Node(point))
end
nodes

#Generate Faces
face_nodes=[] #Vector(undef,length(faces)*3)

for n=1:length(faces)
    for i=1:3
        push!(face_nodes,faces[n].faces[i])
    end
end
face_nodes

#Generate cells
cell_nodes=[]
cell_faces=[]

for n=1:length(volumes)
    for i=1:4
        push!(cell_nodes,volumes[n].volumes[i])
    end
end
cell_nodes

for n=1:length(faces)
    #for i=1:4
        push!(cell_faces,faces[n].faceindex)
    #end
end
cell_faces

#Nodes Range
cell_nodes_range=UnitRange(0,0)
for i=1:length(volumes)
    cell_nodes_range=UnitRange(volumes[i].volumes[1],volumes[i].volumes[end])
end
cell_nodes_range

face_nodes_range=UnitRange(0,0)
for i=1:length(faces)
    face_nodes_range=UnitRange(faces[i].faces[1],faces[i].faces[end])
end
face_nodes_range


#Faces Range
cell_faces_range=UnitRange(faces[1].faceindex,faces[4].faceindex)

#Face Area
A=(nodes[1].coords)
B=(nodes[2].coords)
C=(nodes[3].coords)

AB=[B[1]-A[1],B[2]-A[2],B[3]-A[3]]
AC=[C[1]-A[1],C[2]-A[2],C[3]-A[3]]

face_area=0.5*norm(cross(AB,AC))

#Face centre

function centre_of_face(nodes)
    x_centre = (nodes[1].coords[1] + nodes[2].coords[1] + nodes[3].coords[1]) / 3
    y_centre = (nodes[1].coords[2]+nodes[2].coords[2]+nodes[3].coords[2]) / 3
    z_centre = (nodes[1].coords[3]+nodes[2].coords[3]+nodes[3].coords[3]) / 3
    return (x_centre, y_centre, z_centre)
end

centre_of_face(nodes)

#Cell centre
function centre_of_cell(nodes)
    x_centre = (nodes[1].coords[1]+nodes[2].coords[1]+nodes[3].coords[1]+nodes[4].coords[1]) / 4
    y_centre = (nodes[1].coords[2]+nodes[2].coords[2]+nodes[3].coords[2]+nodes[4].coords[2]) / 4
    z_centre = (nodes[1].coords[3]+nodes[2].coords[3]+nodes[3].coords[3]+nodes[4].coords[3]) / 4
    return (x_centre, y_centre, z_centre)
end

centre_of_cell(nodes)

#Cell Volume
function volume_of_cell(nodes)
    A=(nodes[1].coords)
    B=(nodes[2].coords)
    C=(nodes[3].coords)
    D=(nodes[4].coords)
    AB=[B[1]-A[1],B[2]-A[2],B[3]-A[3]]
    AC=[C[1]-A[1],C[2]-A[2],C[3]-A[3]]
    AD=[D[1]-A[1],D[2]-A[2],D[3]-A[3]]
    return abs(dot(AB, cross(AC, AD))) / 6
end

volume_of_cell(nodes)

#Face Normals
function face_normal(nodes)
    A=(nodes[1].coords)
    B=(nodes[2].coords)
    C=(nodes[3].coords)
    AB=[B[1]-A[1],B[2]-A[2],B[3]-A[3]]
    AC=[C[1]-A[1],C[2]-A[2],C[3]-A[3]]
    cross_product = cross(AB, AC)
    return normalize(cross_product)
end

face_normal(nodes)