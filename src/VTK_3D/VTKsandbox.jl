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

mutable struct Vertex{TI<:Integer} 
    vertexindex::TI
    vertexCount::TI
    vertices::Vector{TI}
end
Vertex(z::TI) where TI<:Integer = Vertex(0 , 0, TI[])

mutable struct Face{TI<:Integer} 
    faceindex::TI
    faceCount::TI
    faces::Vector{TI}
end
Face(z::TI) where TI<:Integer = Face(0 , 0, TI[])

mutable struct Volume{TI<:Integer}
    groupindex::TI
    groupCount::TI
    groups::Vector{TI}
end
Volume(z::TI) where TI<:Integer = Volume(0 , 0, TI[])

mutable struct BoundaryCondition{TI<:Integer}
    name::String
    bcNumber::TI
    elements::Vector{TI}
end
BoundaryCondition(z::TI) where TI<:Integer = BoundaryCondition("default", 0, TI[])

mutable struct Element{TI<:Integer}
    index::TI
    elementCount::TI
    elements::Vector{TI}
end
Element(z::TI) where TI<:Integer = Element(0,0,TI[])

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
   @inbounds for i ∈ 1:length(groups) 
           cellID += 1
           @inbounds for nodeID ∈ groups[i].groups
               push!(nodes[nodeID].neighbourCells, cellID)
           end
   end

nodes

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

args = (
        ("U", model.U), 
        ("p", model.p)
    )

args = (
        ("U", model.U), 
        ("p", model.p),
        ("k", model.turbulence.k),
        ("omega", model.turbulence.omega),
        ("nut", model.turbulence.nut)
    )
    


name="test"
filename=name*".vtk"
    open(filename,"w") do io
        write(io,"# vtk DataFile Version 3.0\n")
        write(io,"Julia 3D CFD Simulation Data\n")
        write(io,"ASCII\n")
        write(io,"DATASET UNSTRUCTURED_GRID\n")
        nPoints=length(nodes)
        nCells=length(cells)
        write(io,"POINTS $(nPoints) float\n")
        for node ∈ nodes
            (;coords)=node
            println(io, coords[1]," ",coords[2]," ", coords[3])
        end

        sum=0

        for cell ∈ cells
            sum += length(cell.nodesID)
        end

        cellListSize=sum+nCells
        write(io,"CELLS $(nCells) $(cellListSize)\n")

        for cell ∈ cells
            nNodes=length(cell.nodesID)
            nodes=""
            for nodeID ∈ cell.nodesID
                node="$(nodeID-1)"
                nodes=nodes*" "*node
            end
            println(io,nNodes," ",nodes)
        end

        write(io,"CELL_TYPES $(nCells)\n")

        for cell ∈ cells
            nCellIDs = length(cell.nodesID)
            if nCellIDs == 3
                type = "5"
            elseif nCellIDs == 4
                type = "9"
            elseif nCellIDs > 4
                type = "7"
            end
            println(io, type)
        end

        write(io, "CELL_DATA $(nCells)\n")

        for arg ∈ args
            label = arg[1]
            field = arg[2]
            field_type = typeof(field)
            if field_type <: ScalarField
                write(io, "SCALARS $(label) double 1\n")
                write(io, "LOOKUP_TABLE CellColors\n")
                for value ∈ field.values
                    println(io, value)
                end
            elseif field_type <: VectorField
                write(io, "VECTORS $(label) double\n")
                for i ∈ eachindex(field.x)
                    println(io, field.x[i]," ",field.y[i] ," ",field.z[i] )
                end
            else
                throw("""
                Input data should be a ScalarField or VectorField e.g. ("U", U)
                """)
            end
        end
    end




