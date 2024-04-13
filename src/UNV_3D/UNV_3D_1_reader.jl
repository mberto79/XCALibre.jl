#UNV Reader New Structure

export load_3D

function load_3D(unv_mesh; scale, integer, float)
    #Defining Variables
    pointindx=0
    elementindx=0
    cellindx=0
    boundaryindx=0
    faceindx=0
    edgeindx=0
    
    #Defining Arrays with Structs
    points=UNV_3D.Point{float,SVector{3,float}}[]
    #edges=UNV_3D.Edge{integer,Vector{integer}}[]
    faces=UNV_3D.Face{integer,Vector{integer}}[]
    cells=UNV_3D.Cell_UNV{integer,Vector{integer}}[]
    boundaryElements=UNV_3D.BoundaryElement{String,integer,Vector{integer}}[]
    #elements=UNV_3D.Element{integer,Vector{integer}}[]
    
    #Defining Arrays for data collection
    #Points
    point=float[]
    pointindex=integer
    
    #Kept in case of future uses.
    #Edges
    # edge=integer[]
    # edgeCount=integer[]
    # edgeindex=integer[]
    edgeindex=0
    
    #Faces
    face=integer[] # Nodes ID for face
    faceindex=integer
    faceCount=integer
    
    #Cells
    cell=integer[] # Nodes ID for cell
    cellindex=integer
    cellCount=integer
    
    #bc
    boundary=integer[] # Element ID for boundary
    boundarys=Tuple{SubString{String}, Vector{Int64}}[]
    boundaryindex=integer
    currentBoundary=zero(integer)
    boundaryNumber=zero(integer)
    
    #Splits UNV file into sections
    for (indx,line) in enumerate(eachline(unv_mesh))
        sline=split(line)
    
        #Points = 2411
        if sline[1]=="2411" && length(sline)==1
            pointindx=indx
        end
        #Elements = 2412 (Lines, Faces, Cells)
        if sline[1]=="2412" && length(sline)==1
            elementindx=indx
        end
        #BC=2467
        if sline[1]== "2467" && length(sline)==1
            boundaryindx=indx
        end
    end
    
    #Extracting Data from UNV file
    # To avoid UNV file jumping indexs if exporting Salome mesh from Windows.
    edge_counter=0
    face_counter=0
    cell_counter=0 

    #face_index_UNV=Int64[]

    for (indx,line) in enumerate(eachline(unv_mesh))
        
        sline=split(line)
        #Points
        if indx>pointindx && indx<elementindx && length(sline)==4 && parse(Float64,sline[4])==11
            pointindex=parse(Int64,sline[1])
            continue
        end
    
        if length(sline)==3 && indx>pointindx && indx<elementindx
            point=[parse(Float64,sline[i]) for i=1:length(sline)]
            push!(points,Point(scale * SVector{3,Float64}(point)))
            continue
        end
    
        #Lines/Edges
        if length(sline)==6 && parse(Int64,sline[end])==2
            #edgeCount=parse(Int,sline[end])
            edge_counter=edge_counter+1
            edgeindex=edge_counter
            edgeindx=indx
            continue
        end
    
        #No need to extract line data as it is not used in builder.
        # if length(sline)==2 && indx>elementindx
        #     edge=[parse(Int,sline[i]) for i=1:length(sline)]
        #     push!(edges,Edge(edgeindex,edgeCount,edge))
        #     #push!(elements,Element(edgeindex,edgeCount,edge))
        #     continue
        # end
    
        #Faces
        #Triangle
        if length(sline)==6 && parse(Int64,sline[2])==41 && parse(Int64,sline[end])==3
            faceCount=parse(Int,sline[end])
            face_counter=face_counter+1
            faceindex=face_counter
            faceindx=indx
            if parse(Int64,sline[1])-edgeindex != face_counter
                throw("Face Index in UNV file are not in order! At UNV index = $(parse(Int64,sline[1]))")
            end
            continue
        end
    
        if length(sline)==3 && indx>elementindx && indx==faceindx+1 #&& parse(Int,sline[end]) ≠ 1
            face=[parse(Int,sline[i]) for i=1:length(sline)]
            push!(faces,Face(faceindex,faceCount,face))
            continue
        end

        #Quad
        if length(sline)==6 && parse(Int,sline[2])==44 && parse(Int,sline[end])==4
            faceCount=parse(Int,sline[end])
            face_counter=face_counter+1
            faceindex=face_counter
            faceindx=indx
            if parse(Int64,sline[1])-edgeindex != face_counter
                throw("Face Index in UNV file are not in order! At UNV index = $(parse(Int64,sline[1]))")
            end
            continue
        end

        if length(sline)==4 && indx>elementindx && indx==faceindx+1
            face=[parse(Int,sline[i]) for i=1:length(sline)]
            push!(faces,Face(faceindex,faceCount,face))
            continue
        end

        #Cells
        #Tetrahedral
        if length(sline)==6 && parse(Int,sline[2])==111
            cellCount=parse(Int,sline[end])
            cell_counter=cell_counter+1
            cellindex=cell_counter
            cellindx=indx
            if parse(Int64,sline[1])-faceindex-edgeindex != cell_counter
                throw("Cell Index in UNV file are not in order! At UNV index = $(parse(Int64,sline[1]))")
            end
            continue
        end
    
        if length(sline)==4 && indx<boundaryindx && indx>elementindx
            cell=[parse(Int64,sline[i]) for i=1:length(sline)]
            push!(cells,Cell_UNV(cellindex,cellCount,cell))
            continue
        end

        #Hexahedral
        if length(sline)==6 && parse(Int,sline[2])==115
            cellCount=parse(Int,sline[end])
            cell_counter=cell_counter+1
            cellindex=cell_counter
            cellindx=indx
            if parse(Int64,sline[1])-faceindex-edgeindex != cell_counter
                throw("Cell Index in UNV file are not in order! At UNV index = $(parse(Int64,sline[1]))")
            end
            continue
        end

        if length(sline)==8 && indx<boundaryindx && indx>elementindx
            cell=[parse(Int,sline[i]) for i=1:length(sline)]
            push!(cells,Cell_UNV(cellindex,cellCount,cell))
            continue
        end

        #Wedge
        if length(sline)==6 && parse(Int,sline[2])==112
            cellCount=parse(Int,sline[end])
            cell_counter=cell_counter+1
            cellindex=cell_counter
            cellindx=indx
            if parse(Int64,sline[1])-faceindex-edgeindex != cell_counter
                throw("Cell Index in UNV file are not in order! At UNV index = $(parse(Int64,sline[1]))")
            end
            continue
        end

        if length(sline)==6 && indx<boundaryindx && indx>elementindx
            cell=[parse(Int,sline[i]) for i=1:length(sline)]
            push!(cells,Cell_UNV(cellindex,cellCount,cell))
            continue
        end
    
        #Boundary
        if length(sline)==1 && indx>boundaryindx && typeof(tryparse(Int64,sline[1]))==Nothing
            boundaryindex=sline[1]
            currentBoundary=currentBoundary+1
            newBoundary=BoundaryElement(0)
            push!(boundaryElements, newBoundary)
            boundaryNumber=boundaryNumber+1
            boundaryElements[currentBoundary].index=currentBoundary
            boundaryElements[currentBoundary].name=boundaryindex
            continue
        end

        #Window Users need to have this enabled
        # dict=Dict() # To avoid UNV from skipping index, dictionary is used to assign UNV index to new face index.
        # for (n,f) in enumerate(face_index_UNV)
        #     dict[f] = n
        # end
    
        if length(sline)==8 && indx>boundaryindx && parse(Int64,sline[2])!=0
            boundary=[parse(Int64,sline[i]) for i=1:length(sline)]
            push!(boundarys,(boundaryindex,boundary))
            #push!(boundaryElements[currentBoundary].elements,dict[parse(Int64,sline[2])]) # Kept for Dict
            push!(boundaryElements[currentBoundary].facesID,parse(Int64,sline[2])-edgeindex)
            #push!(boundaryElements[currentBoundary].elements,dict[parse(Int64,sline[6])]) # Kept for Dict
            push!(boundaryElements[currentBoundary].facesID,parse(Int64,sline[6])-edgeindex)
            continue
        end

        if length(sline)==4 && indx>boundaryindx && parse(Int64,sline[2])!=0
            boundary=[parse(Int64,sline[i]) for i=1:length(sline)]
            push!(boundarys,(boundaryindex,boundary))
            #push!(boundaryElements[currentBoundary].elements,dict[parse(Int64,sline[2])]) # Kept for Dict
            push!(boundaryElements[currentBoundary].facesID,parse(Int64,sline[2])-edgeindex)
            continue
        end
    
    end
    return points,faces,cells,boundaryElements

end