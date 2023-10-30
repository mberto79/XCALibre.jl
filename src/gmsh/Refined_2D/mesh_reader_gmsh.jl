

struct elementBlock
    entityDim::Int32
    entityTag::Int32
    elementType::Int32
    numElementsInBlock::Int32
    elementTag::Vector{Int32}
    nodeTag::Matrix{Int32}
end

struct curveEntity
    curveTag::Int32
    numPhysicalTags::Int32
    physicalTags::Vector{Int32}
end

function mesh_reader_gmsh(filename)

    Nv = 0 # Number of nodes
    K = 0
    BCType = Int32.(zeros(K, 3))
    EToV = Int32.(zeros(K, 3))
    x = ""
    nodes = zeros(Nv, 3)

    open(filename) do fid
        # Read Intro
        for i =1:4
            x = readline(fid)
        end

        # Physical names
        numPhys = parse.(Int32, readline(fid))
        boundaries = Vector{String}(undef, numPhys)
        boundaryid = Vector{Int32}(undef, numPhys)
        for i =1:numPhys
            tmp = split(readline(fid)," ",keepempty=false)
            id = parse(Int32, tmp[2])

            BC = tmp[3][2:end-1]
            boundaries[id] = tmp[3][2:end-1]
            boundaryid[id] = 0
            if BC == "Inlet"
                boundaryid[id] = 1
            end
            if BC == "Outlet"
                boundaryid[id] = 2
            end
            if BC == "Wall"
                boundaryid[id] = 3
            end
        end

        x = readline(fid)
        x = readline(fid)

        # Entity descriptions
        numEnt = parse.(Int32, split(readline(fid),' ',keepempty=false))
        for i =1:numEnt[1]
            x = readline(fid)
        end

        println(numEnt)

        curveEntities = Vector{curveEntity}(undef, numEnt[2])
        for i =1:numEnt[2]
            tmp = parse.(Float64, split(readline(fid)," ",keepempty=false))
            curveTag = tmp[1]
            minX = tmp[2]
            minY = tmp[3]
            minZ = tmp[4]
            maxX = tmp[5]
            maxY = tmp[6]
            maxZ = tmp[7]
            numPhysicalTags = Int(tmp[8])
            physicalTag = Int.(zeros(numPhysicalTags))
            if numPhysicalTags != 0
                for j = 1:numPhysicalTags
                    physicalTag[j] = Int(tmp[8+j])
                end
            else
                physicalTag = [0]
            end
            curveEntities[i] = curveEntity(curveTag, numPhysicalTags, physicalTag)
        
            println(curveEntities[i])
        end

        for i =1:numEnt[3]
            tmp = parse.(Float64, split(readline(fid)," ",keepempty=false))
            curveTag = tmp[1]
            minX = tmp[2]
            minY = tmp[3]
            minZ = tmp[4]
            maxX = tmp[5]
            maxY = tmp[6]
            maxZ = tmp[7]
            numPhysicalTags = Int(tmp[8])
            if numPhysicalTags != 0
                physicalTag = Int.(zeros(numPhysicalTags))
                for j = 1:numPhysicalTags
                    physicalTag[j] = Int(tmp[8+j])
                end
            end
            numBoundingCurves = Int(tmp[8+numPhysicalTags])
            if numBoundingCurves != 0
                curveTag = Int.(zeros(numBoundingCurves))
                for j = 1:numBoundingCurves
                    curveTag[j] = Int(tmp[8+numPhysicalTags+j])
                end
            end
        end

        for i =1:numEnt[4] 
            x = readline(fid)
        end

        x = readline(fid)
        x = readline(fid)

        # Nodes
        nodeData = parse.(Int32, split(readline(fid),' ',keepempty=false))
        numEntityBlocks = nodeData[1]
        numNodes = nodeData[2]
        minNodeTag = nodeData[3]
        maxNodeTag = nodeData[4]

        nodes = zeros(maxNodeTag,3)

        for j = 1:nodeData[1]
            nodeBlockData = parse.(Int32, split(readline(fid),' ',keepempty=false))
            tag = Int32.(zeros(nodeBlockData[4]))
            for k = 1:nodeBlockData[4]
                tag[k] = parse(Int32, readline(fid))[1]
            end
            for k = 1:nodeBlockData[4]
                tmp = parse.(Float64, split(readline(fid)," ",keepempty=false))
                nodes[tag[k], :] .= tmp[1:3]
            end
        end

        x = readline(fid)
        x = readline(fid)

        # Elements
        elementData = parse.(Int32, split(readline(fid),' ',keepempty=false))
        numEntityBlocks = elementData[1]
        numElements = elementData[2]
        minElementTag = elementData[3]
        maxElementTag = elementData[4]

        elementBlocks = Vector{elementBlock}(undef, numEntityBlocks)

        for j = 1:elementData[1]
            elementsBlockData = parse.(Int32, split(readline(fid),' ',keepempty=false))
            entityDim = elementsBlockData[1]
            entityTag = elementsBlockData[2]
            elementType = elementsBlockData[3]
            numElementsInBlock = elementsBlockData[4]
            
            elementTag = zeros(numElementsInBlock)

            numElements = numElementsFunc(elementType)
            nodeTags = zeros(numElementsInBlock, numElements)

            for k = 1:elementsBlockData[4]
                tmp = parse.(Int32, split(readline(fid)," ",keepempty=false))

                elementTag[k] = tmp[1]
                nodeTags[k, :] = tmp[2:end]
            end
            elementBlocks[j] = elementBlock(entityDim, entityTag, elementType, numElementsInBlock, elementTag, nodeTags)
        end

        num2DElements = 0
        for elementBlock in elementBlocks
            if elementBlock.entityDim == 2
                num2DElements += elementBlock.numElementsInBlock
            end
        end

        K = num2DElements
        EToV = Int32.(zeros(K, 3))

        oldid = 1
        for elementBlock in elementBlocks
            if elementBlock.entityDim == 2
                newid = oldid + elementBlock.numElementsInBlock - 1
                EToV[oldid:newid,:] .= elementBlock.nodeTag[:, :] # For triangles only
                oldid = newid + 1
            end
        end

        BCType = Int32.(zeros(K, 3))

        i = 1
        for elementBlock in elementBlocks
            if elementBlock.entityDim == 1
                while curveEntities[i].numPhysicalTags == 0
                    i +=1 
                end
                
                println(curveEntities[i])
                if curveEntities[i].numPhysicalTags > 0
                    println(i, " ",boundaryid[curveEntities[i].physicalTags[1]])
                    for j = 1:elementBlock.numElementsInBlock
                        id1 = Tuple.(findall(x-> x == elementBlock.nodeTag[j,1], EToV))
                        id2 = Tuple.(findall(x-> x == elementBlock.nodeTag[j,end], EToV[first.(id1), :]))
    
                        id3 = first.(id1[first.(id2)])[1]
                        id4 = last.(id1[first.(id2)])[1]
                        idsup = findall(x-> x == elementBlock.nodeTag[j,end], EToV[id3, :])[1]
                        # println(id4, " ",idsup)
                        elementid = max(id4, idsup) - abs(id4 - idsup) 

                        # println(elementBlock.nodeTag[j,:], " ", EToV[id3,:], " ", EToV[id4,:])
                        BCType[id3, elementid] = boundaryid[curveEntities[i].physicalTags[1]]
                        # BCType[id3, id4] = boundaryid[curveEntities[i].physicalTags[1]]
                    end
                end
                i +=1 
            end
        end
 
        close(fid) 
    end
    
    return mesh_struct(Nv, nodes, K, EToV, BCType)
end

function numElementsFunc(i)

    tableOfElements = [2, 3, 4, 4, 8, 6, 5, 3, 6, 9, 10, 27, 18, 14, 1, 8, 20, 15, 13]

    return numElements = tableOfElements[i]
end