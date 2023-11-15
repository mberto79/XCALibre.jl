#UNV Reader New Structure

#For Tetrahedral Cell Types

#Define Variables
nodeindx=0
elementindx=0
boundaryindx=0

#Define Arrays
nodes=Node[]
#faces=Mesh3[]
#cells=Cells[]
elements=Mesh3[]
boundarys=Boundary[]

#Define Arrays for Data Collection
#Nodes
node=[]
nodeindex=[]

#Faces
face=[]
faceindex=[]
faceCount=[]

#Cells 
cell=[]
cellindex=[]
cellcount=[]

#Boundary
boundary=[]
boundaryindex=[]
currentBC=0

#Split UNV file
for (indx,line) in enumerate(eachline(unv_mesh))
    sline=split(line)
    #Nodes = 2411
    if sline[1]=="2411" && length(sline)==1
        nodeindx=indx
    end
    #Edges,Faces,Cells=2412
    if sline[1]=="2412" && length(sline)==1
        elementindx=indx
    end
    #Boundary
    if sline[1]=="2467" && length(sline)==1
        boundaryindx=indx
    end
end

#Check
nodeindx
elementindx
boundaryindx

#Extract Data
for (indx,line) in enumerate(eachline(unv_mesh))
    sline=split(line)
    #Nodes
    if length(sline)==3 && indx>pointindx && indx<elementindx
        node=[parse(Float64,sline[i]) for i=1:length(sline)]
        push!(nodes,Node(SVector{3,Float64}(node)))
        continue
    end

    #Faces
    if length(sline)==6 && parse(Int64,sline[end])==3
        faceCount=parse(Int,sline[end])
        continue
    end

    if length(sline)==3 && indx>elementindx && parse(Int,sline[end]) ≠ 1
        face=[parse(Int,sline[i]) for i=1:length(sline)]
        push!(elements.face_nodes,face)
        continue
    end

    #Cells
    if length(sline)==6 && parse(Int,sline[2])==111
        groupCount=parse(Int,sline[end])
        groupindx=indx
        continue
    end

    if length(sline)==4 && indx>elementindx
        cell=[parse(Int,sline[i]) for i=1:length(sline)]
        push!(elements.cell_nodes, cell)
        continue
    end

    #Boundary
    if length(sline)==1 && indx>bcindx && typeof(tryparse(Int64,sline[1]))==Nothing
        boundaryindx=indx
        name=sline[1]
        name=convert(Symbol,name)
        currentBC=currentBC+1
        boundarys[currentBC].name=name
        continue
    end

    if length(sline)==8 && indx>bcindx && parse(Int64,sline[2])!=0
        push!(boundarys[currentBC].facesID,parse(Int64,sline[2]))
        push!(boundarys[currentBC].facesID,parse(Int64,sline[6]))
    end
end