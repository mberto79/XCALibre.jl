unv_mesh="src/UNV_3D/Mesh_tetrasmall.unv"

using StaticArrays
using LinearAlgebra
using Setfield
using Adapt
include("UNV_3D_NS_Types.jl")

struct Face3D{I,F}
    nodes_map::SVector{3,I}
    ownerCells::SVector{2,I}
    centre::SVector{3, F}
    normal::SVector{3, F}
    e::SVector{3, F}
    area::F
    delta::F
    weight::F
end
Adapt.@adapt_structure Face3D
Face3D(I,F) = begin
    zi = zero(I); zf = zero(F)
    vec_2I = SVector{2,I}(zi,zi)
    vec_3I=SVector{3,I}(zi,zi,zi)
    vec_3F = SVector{3,F}(zf,zf,zf)
    Face3D(vec_3I, vec_2I, vec_3F, vec_3F, vec_3F, zf, zf, zf)
end

struct Mesh3{I,F} #<: AbstractMesh
    cells::Vector{Cell{I,F}}
    cell_nodes::Vector{I}
    cell_faces::Vector{I}
    cell_neighbours::Vector{I}
    cell_nsign::Vector{I}
    faces::Vector{Face3D{I,F}}
    face_nodes::Vector{I}
    boundaries::Vector{Boundary{I}}
    nodes::Vector{Node{I,F}}
end
Adapt.@adapt_structure Mesh3

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
        bcindx=indx
    end
end

#Check
pointindx
elementindx
bcindx

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
        #allelementCount=parse(Int,sline[end])
        #allelementindex=parse(Int,sline[1])
        continue
    end

    if length(sline)==2 && indx>elementindx
        edge=[parse(Int,sline[i]) for i=1:length(sline)]
        push!(edges,Edge(edgeindex,edgeCount,edge))
        #allelement=[parse(Int,sline[i]) for i=1:length(sline)]
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
        push!(faces,Face(faceindex,faceCount,face))
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
        push!(volumes,Volume(volumeindex,volumeCount,volume))
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
        boundaryElements[currentBoundary].name=sline[1]
        continue
    end

    if length(sline)==8 && indx>boundaryindx && parse(Int64,sline[2])!=0
        boundary=[parse(Int,sline[i]) for i=1:length(sline)]
        push!(boundarys,(boundaryindex,boundary))
        #push!(boundaryElements[currentBC].elements,parse(Int,sline[2]))
        #push!(boundaryElements[currentBC].elements,parse(Int,sline[6]))
        #push!(elements,parse(Int64,sline[2]))
        
        #push!(boundaryElements,BoundaryLoader(name,groupNumber,parse(Int,sline[2])))
        #push!(boundaryElements,BoundaryLoader(name,groupNumber,parse(Int,sline[6])))
        
        #push!(boundaryElements,BoundaryLoader(name,groupNumber,elements))
        #push!(elements,parse(Int64,sline[6]))
        #push!(boundaryElements,BoundaryLoader(name,groupNumber,elements))
        #element1=parse(Int64,sline[2])
        #element2=parse(Int64,sline[6])
        #push!(elements,element1)
        #push!(elements,element2)
        
        #elements=[parse(Int64,sline[4*i-2]) for i=1:2]
        #push!(boundaryElements,BoundaryLoader(name,groupNumber,elements))
        
        push!(boundaryElements[currentBoundary].elements,parse(Int64,sline[2]))
        push!(boundaryElements[currentBoundary].elements,parse(Int64,sline[6]))
        continue
    end


    #if length(sline)==1 && indx>bcindx
        #currentBC=currentBC+1
        #boundaryElements[currentBC].groupNumber=currentBC
        #boundaryElements[currentBC].name=sline[1]
        #continue
    #end

    #if length(sline)==8 && indx>bcindx && parse(Int64,sline[2])!=0
        #push!(boundaryElements,BoundaryLoader(name,groupNumber,parse(Int,sline[2])))
        #push!(boundaryElements,BoundaryLoader(name,groupNumber,parse(Int,sline[6])))
        # push!(boundaryElements[currentBC].elements,parse(Int,sline[2]))
        #push!(boundaryElements[currentBC].elements,parse(Int,sline[6]))
        #continue
    #end

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
length(boundarys)*2

boundaryElements
boundaryNumber
currentBoundary
elements

#Generate Nodes
nodes=Node[]
@inbounds for i ∈ 1:length(points)
    point1= points[i].xyz
    push!(nodes,Node(point1))
end
nodes

#convert
boundaries1 = Vector{Boundary{integer}}(undef, (length(boundarys)*2))
cells1 = Vector{Cell{integer,float}}(undef, length(volumes))
faces1 = Vector{Face3D{integer,float}}(undef, length(faces))
nodes1 = Vector{Node{integer,float}}(undef, length(points))


