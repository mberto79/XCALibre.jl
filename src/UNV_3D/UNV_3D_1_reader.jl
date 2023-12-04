#UNV Reader New Structure

export load_3D

function load_3D(unv_mesh)
    #Defining Variables
    pointindx=0
    elementindx=0
    volumeindx=0
    boundaryindx=0
    faceindx=0
    
    #Defining Arrays with Structs
    points=UNV_3D.Point[]
    edges=UNV_3D.Edge[]
    faces=UNV_3D.Face[]
    volumes=UNV_3D.Volume[]
    boundaryElements=UNV_3D.BoundaryElement[]
    elements=UNV_3D.Element[]
    
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
        #Tetrahedral
        if length(sline)==6 && parse(Int64,sline[2])==41 && parse(Int64,sline[end])==3
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

        #Hexahedral
        if length(sline)==6 && parse(Int,sline[2])==44 && parse(Int,sline[end])==4
            faceCount=parse(Int,sline[end])
            faceindex=parse(Int,sline[1])
            faceindx=indx
            continue
        end

        if length(sline)==4 && indx>elementindx && indx==faceindx+1
            face=[parse(Int,sline[i]) for i=1:length(sline)]
            push!(faces,Face(faceindex-edgeindex,faceCount,face))
            push!(elements,Element(faceindex,faceCount,face))
            continue
        end
    
        #Volumes
        #Tetrahedral
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

        #Hexahedral
        if length(sline)==6 && parse(Int,sline[2])==115
            volumeCount=parse(Int,sline[end])
            volumeindex=parse(Int,sline[1])
            volumeindx=indx
            continue
        end

        if length(sline)==8 && indx<boundaryindx
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
    return points,edges,faces,volumes,boundaryElements

end