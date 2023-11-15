meshFile="src/UNV_3D/Mesh_tetrasmall.unv"

#Structs
using StaticArrays
using LinearAlgebra
using Setfield
using Adapt

struct Point{TF<:AbstractFloat}
    xyz::SVector{3, TF}
end
Point(z::TF) where TF<:AbstractFloat = Point(SVector{3, TF}(zero(TF), zero(TF), zero(TF)))

mutable struct Vertices{TI<:Integer} 
    vertexindex1::TI
    vertexCount1::TI
    vertices1::Vector{TI}
end
Vertices(z::TI) where TI<:Integer = Vertices(0 , 0, TI[])

mutable struct Faces{TI<:Integer} 
    faceindex::TI
    faceCount::TI
    faces::Vector{TI}
end
Faces(z::TI) where TI<:Integer = Faces(0 , 0, TI[])

mutable struct Volumes{TI<:Integer}
    groupindex::TI
    groupCount::TI
    groups::Vector{TI}
end
Volumes(z::TI) where TI<:Integer = Volumes(0 , 0, TI[])

mutable struct BoundaryLoader{TI<:Integer}
    name::String
    groupNumber::TI
    elements::Vector{TI}
end
BoundaryLoader(z::TI) where TI<:Integer = BoundaryLoader("default", 0, TI[])

mutable struct Element{TI<:Integer}
    index::TI
    vertexCount::TI
    vertices::Vector{TI}
end
Element(z::TI) where TI<:Integer = Element(0,0,TI[])

    #Defining Variables
    pointindx=0
    elementindx=0
    groupindx=0
    bcindx=0

    #Defining Arrays with Structs
    points=Point[]
    vertices=Vertices[]
    faces=Faces[]
    groups=Volumes[]
    boundaryElements=BoundaryLoader[]
    elements=Element[]

    #Defining Arrays for data collection
    #Points
    point=[]
    pointindex=[]

    #Vertices
    vertex=[]
    vertexCount1=[]
    vertexindex1=[]

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
    groupNumber=0
    
    

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
            vertexCount1=parse(Int,sline[end])
            vertexindex1=parse(Int,sline[1])
            #allelementCount=parse(Int,sline[end])
            #allelementindex=parse(Int,sline[1])
            continue
        end

        if length(sline)==2 && indx>elementindx
            vertex=[parse(Int,sline[i]) for i=1:length(sline)]
            push!(vertices,Vertices(vertexindex1,vertexCount1,vertex))
            #allelement=[parse(Int,sline[i]) for i=1:length(sline)]
            push!(elements,Element(vertexindex1,vertexCount1,vertex))
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
            push!(faces,Faces(faceindex,faceCount,face))
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
            push!(groups,Volumes(groupindex,groupCount,group))
            push!(elements,Element(groupindex,groupCount,group))
            continue
        end

        #BC
        if length(sline)==1 && indx>bcindx && typeof(tryparse(Int64,sline[1]))==Nothing
            bcindex=sline[1]
            currentBC=currentBC+1
            newBoundary=BoundaryLoader(0)
            push!(boundaryElements, newBoundary)
            name=sline[1]
            name=convert(String,name)
            groupNumber=groupNumber+1
            boundaryElements[currentBC].groupNumber=currentBC
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

vertexCount1
vertexindex1
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
groupNumber
currentBC
elements
