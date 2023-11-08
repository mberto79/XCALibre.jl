meshFile="src/UNV_3D/Mesh_tetrasmall.unv"
#Structs
using StaticArrays
using LinearAlgebra
using Setfield
using Adapt
include("UNV_3D_Types.jl")

    pointindx=0
    elementindx=0
    groupindx=0
    bcindx=0

    #Defining Arrays with Structs
    points=Point[]
    vertices=Vertex[]
    faces=Face[]
    groups=Volume[]
    boundaryElements=BoundaryCondition[]
    elements=Element[]

    #Defining Arrays for data collection
    #Points
    point=[]
    pointindex=[]

    #Vertices
    vertex=[]
    vertexCount=[]
    vertexindex=[]

    #Faces
    face=[]
    faceindex=[]
    faceCount=[]

    #Volumes
    group=[]
    groupindex=[]
    groupCount=[]

    #bc
    bc=[]
    bcs=[]
    bcindex=[]
    currentBC=0
    bcNumber=0
    
    

    #Splits UNV file into sections
    for (indx,line) in enumerate(eachline(meshFile))
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
    for (indx,line) in enumerate(eachline(meshFile))
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
            vertexCount=parse(Int,sline[end])
            vertexindex=parse(Int,sline[1])
            #allelementCount=parse(Int,sline[end])
            #allelementindex=parse(Int,sline[1])
            continue
        end

        if length(sline)==2 && indx>elementindx
            vertex=[parse(Int,sline[i]) for i=1:length(sline)]
            push!(vertices,Vertex(vertexindex,vertexCount,vertex))
            #allelement=[parse(Int,sline[i]) for i=1:length(sline)]
            push!(elements,Element(vertexindex,vertexCount,vertex))
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
            groupCount=parse(Int,sline[end])
            groupindex=parse(Int,sline[1])
            groupindx=indx
            continue
        end

        if length(sline)==4 && indx>elementindx
            group=[parse(Int,sline[i]) for i=1:length(sline)]
            push!(groups,Volume(groupindex,groupCount,group))
            push!(elements,Element(groupindex,groupCount,group))
            continue
        end

        #BC
        if length(sline)==1 && indx>bcindx && typeof(tryparse(Int64,sline[1]))==Nothing
            bcindex=sline[1]
            currentBC=currentBC+1
            newBoundary=BoundaryCondition(0)
            push!(boundaryElements, newBoundary)
            name=sline[1]
            name=convert(String,name)
            bcNumber=bcNumber+1
            boundaryElements[currentBC].bcNumber=currentBC
            boundaryElements[currentBC].name=sline[1]
            continue
        end

        if length(sline)==8 && indx>bcindx && parse(Int64,sline[2])!=0
            bc=[parse(Int,sline[i]) for i=1:length(sline)]
            push!(bcs,(bcindex,bc))
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
            
            push!(boundaryElements[currentBC].elements,parse(Int64,sline[2]))
            push!(boundaryElements[currentBC].elements,parse(Int64,sline[6]))
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

vertexCount
vertexindex
vertex
vertices

faceCount
faceindex
face
faces

groupindx
groupCount
groupindex
group
groups

bcindex
bcindx
bc
bcs
length(bcs)*2

boundaryElements
bcNumber
currentBC
elements



#Total Boundary Faces
sum=zero(Int64)
@inbounds for boundary ∈ boundaryElements
    sum += length(boundary.elements)
end
sum

#First 2D element

element_index=zero(Int64)
@inbounds for counter ∈ eachindex(elements)
    nvertices = elements[counter].elementCount 
        if nvertices > 2
            element_index = counter
            return element_index
        end
    end

element_index=zero(Int64)
element_index=length(vertices)

#Generate Nodes
struct Node{TI, TF}
    coords::SVector{3, TF}
    neighbourCells::Vector{TI}
end
Adapt.@adapt_structure Node
Node(TF) = begin
    zf = zero(TF)
    vec_3F = SVector{3,TF}(zf,zf,zf)
    Node(vec_3F, Int64[])
end
Node(x::F, y::F, z::F) where F<:AbstractFloat = Node(SVector{3, F}(x,y,z), Int64[])
Node(zero::F) where F<:AbstractFloat = Node(zero, zero, zero)
Node(vector::F) where F<:AbstractVector = Node(vector, Int64[])



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

nodes

#Generate Faces
struct Face3D{I,F}
    nodesID::SVector{3,I}
    ownerCells::SVector{2,I}
    centre::SVector{3, F}
    normal::SVector{3, F}
    e::SVector{3, F}
    area::F
    delta::F
    weight::F
end
Face3D(I,F) = begin
    zi = zero(I); zf = zero(F)
    vec_2I = SVector{2,I}(zi,zi)
    vec_3I=SVector{3,I}(zi,zi,zi)
    vec_3F = SVector{3,F}(zf,zf,zf)
    Face3D(vec_3I, vec_2I, vec_3F, vec_3F, vec_3F, zf, zf, zf)
end

faces1 = Face3D[]

#@inbounds for boundary ∈ boundaryElements
    @inbounds for elementi=element_index+1:length(faces)+element_index
            face1 = Face3D(Integer,Float64)
            vertex1 = elements[elementi].elements[1]
            vertex2 = elements[elementi].elements[2]
            vertex3 = elements[elementi].elements[3]
            if vertex1 < vertex2 && vertex2 < vertex3
                face1 = @set face1.nodesID = SVector{3,Integer}(vertex1, vertex2, vertex3)
                push!(faces1, face1)
                continue
            elseif vertex1 > vertex2 && vertex3 > vertex1
                face1 = @set face1.nodesID = SVector{3,Integer}(vertex2, vertex1,vertex3)
                push!(faces1, face1)
                continue
            elseif vertex3 > vertex1 && vertex2 > vertex3
                face1 = @set face1.nodesID = SVector{3,Integer}(vertex1, vertex3,vertex2)
                push!(faces1, face1)
                continue
            elseif vertex3 > vertex2 && vertex1 > vertex3
                face1 = @set face1.nodesID = SVector{3,Integer}(vertex2, vertex3,vertex1)
                push!(faces1, face1)
                continue
            elseif vertex1 > vertex3 && vertex2 > vertex1
                face1 = @set face1.nodesID = SVector{3,Integer}(vertex3, vertex1,vertex2)
                push!(faces1, face1)
                continue
            elseif vertex2 > vertex3 && vertex1 > vertex2
                face1 = @set face1.nodesID = SVector{3,Integer}(vertex3, vertex2,vertex1)
                push!(faces1, face1)
                continue
            else
                throw("Boundary elements are inconsistent: possible mesh corruption")
            end 
    end
#end

faces1
face1

struct Cell{I,F}
    nodesID::Vector{I}
    facesID::Vector{I}
    neighbours::Vector{I}
    nsign::Vector{I}
    centre::SVector{3, F}
    volume::F
end
Cell(I,F) = begin
    zf = zero(F)
    vec3F = SVector{3,F}(zf,zf,zf)
    Cell(I[], I[], I[], I[], vec3F, zf)
end

cells=Cell[]
@inbounds for i ∈ length(vertices)+length(faces)+1:length(elements)
    cell = Cell(Int64,Float64)
    nodesID = elements[i].elements
    # if length(nodesID) > 2
    @inbounds for nodeID ∈ nodesID
            push!(cell.nodesID, nodeID)
        end
        push!(cells, cell)
    # end
end

cells