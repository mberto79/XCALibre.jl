mutable struct ElementBlock
    entityDim::Int64
    entityTag::Int64
    elementType::Int64
    numElementsInBlock::Int64
    connectivity::Vector{Vector{Int64}}
end

struct Mesh
    nodes::Array{Float64,2}
    sconnect::Array{ElementBlock,1}
    vconnect::Array{ElementBlock,1}
end


function readgmesh(fname::String, dim::Int)
    thisMesh = Mesh
    open(fname,"r") do f
        id = 0
        bin_flag = 0
        ent_id = 1
        block_id = 0
        node_id = 0
        block2_id = 0
        tag = 0
        element_id = 0
        snum = 0
        surfToPhysiclTag = Dict{Integer,Integer}()
        global vnum = 0
        for line in eachline(f)
            if line == "\$MeshFormat"
                id = 1
                println("Reading Mesh Format...")
            elseif line == "\$PhysicalNames"
                id = 2
            elseif line == "\$Entities"
                id = 3
                println("Reading Entities...")
            elseif line == "\$PartitionedEntities"
                id = 4
            elseif line == "\$Nodes"
                id = 5
                println("Reading Nodes...")
            elseif line == "\$Elements"
                id = 6
                println("Reading Elements...")
            elseif line == "\$Periodic"
                id = 7
            elseif line == "\$GhostElements"
                id = 8
            elseif line == "\$Parametrizations"
                id = 9
            elseif line == "\$NodeData"
                id = 10
            elseif line == "\$ElementData"
                id = 11
            elseif line == "\$ElementNodeData"
                id = 12
            elseif line == "\$InterpolationScheme"
                id = 13
            elseif length(line)>=4 && line[1:4] == "\$End"
                id = 0
            else
                if id == 1 # Mesh Format
                    temp = split(line)
                    version = parse(Float16,temp[1])
                    bin_flag = parse(Int16,temp[2]) # 0 if ASCII, 1 if binary
                    data_size = parse(Int16,temp[3])
                elseif id == 3 # Entities
                    if ent_id == 1
                        temp = split(line)
                        global no_points = parse(Int32,temp[1])
                        global no_curves = parse(Int32,temp[2])
                        global no_surfaces = parse(Int32,temp[3])
                        global no_volumes = parse(Int32,temp[4])
                        global count_points = 0
                        global count_curves = 0
                        global count_surfaces = 0
                        global count_volumes = 0
                        ent_id = 2
                    elseif ent_id == 2 # Point tags
                        count_points += 1
                        temp = split(line)
                        pointTag = parse(Int32,temp[1])
                        count_points == no_points ? ent_id = 3 : ent_id = 2
                    elseif ent_id == 3 # Curve tags
                        count_curves += 1
                        temp = split(line)
                        curveTag = parse(Int32,temp[1])
                        count_curves == no_curves ? ent_id = 4 : ent_id = 3
                    elseif ent_id == 4 # Surface tags
                        count_surfaces += 1
                        temp = split(line)
                        surfTag = parse(Int32,temp[1])
                        numPhysicalTags = parse(Int32,temp[8])  # If this is zero it seems to catch non existent internal faces
                        physicalTag = parse(Int32,temp[9])
                        if numPhysicalTags == 1
                            surfToPhysiclTag[surfTag] = physicalTag
                        elseif numPhysicalTags == 0
                            surfToPhysiclTag[surfTag] = 0
                        end
                        count_surfaces == no_surfaces ? ent_id = 5 : ent_id = 4
                    elseif ent_id == 5 # Volume tags
                        count_volumes += 1
                        temp = split(line)
                        volTag = parse(Int32,temp[1])
                    end
                elseif id == 5 # Nodes
                    if block_id == 0
                        temp = split(line)
                        numEntityBlocks = parse(Int32,temp[1])

                        numNodes = parse(Int32,temp[2])
                        minNodeTag = parse(Int32,temp[3])
                        maxNodeTag = parse(Int32,temp[4])

                        global block_dim = Array{Int32,1}(undef,numEntityBlocks)
                        global block_tag = Array{Int32,1}(undef,numEntityBlocks)
                        global block_numNodes = Array{Int32,1}(undef,numEntityBlocks)

                        global nodeTag = Array{Int32,1}(undef,numNodes)
                        global nodes = zeros(numNodes,3)

                        block_id = 1
                    elseif block_id > 0 && node_id == 0
                        temp = split(line)
                        block_dim[block_id] = parse(Int32,temp[1])
                        block_tag[block_id] = parse(Int32,temp[2])
                        block_numNodes[block_id] = parse(Int32,temp[4])
                        if block_numNodes[block_id] == 0
                            block_id += 1
                        else
                            node_id += 1
                            tag = 1
                        end
                    elseif node_id > 0
                        if tag == 1 && block_numNodes[block_id] != 0
                            temp = split(line)
                            global nodeTag1 = parse(Int32,temp[1])
                            nodeTag[nodeTag1] = nodeTag1
                            node_id == block_numNodes[block_id] ? tag = 2 : node_id += 1
                            tag == 2 ? node_id = 1 : node_id = node_id
                        elseif tag == 2
                            nodebit = nodeTag1 - block_numNodes[block_id] + node_id
                            temp = split(line)
                            nodes[nodebit,1] = parse(Float64,temp[1])
                            nodes[nodebit,2] = parse(Float64,temp[2])
                            nodes[nodebit,3] = parse(Float64,temp[3])
                            node_id == block_numNodes[block_id] ? node_id = 0 : node_id += 1
                            node_id == 0 ? block_id += 1 : node_id = node_id
                        end
                    end
                elseif id == 6 # Elements
                    if block2_id == 0
                        temp = split(line)
                        numEntityBlocks = parse(Int32,temp[1])

                        global block = Array{ElementBlock}(undef,numEntityBlocks)
                        global allTags = Array{Int}(undef,numEntityBlocks)
                        numElements = parse(Int32,temp[2])
                        minElementTag = parse(Int32,temp[3])
                        maxElementTag = parse(Int32,temp[4])

                        block2_id = 1
                    elseif (block2_id > 0 && element_id == 0)
                        temp = split(line)
                        global entityDim = parse(Int32,temp[1])
                        global entityTag = parse(Int32,temp[2])
                        global elementType = parse(Int32,temp[3])
                        global numElementsInBlock = parse(Int32,temp[4])
                        global element = Array{Int32,1}(undef,numElementsInBlock)

                        entityDim == dim - 1 ? snum += 1 : snum += 0
                        entityDim == dim ? vnum += 1 : vnum += 0

                        global type_map = [2,3,4,4,8,6,5,3,6,9,10,27,18,14,1,8,20,15,13,9,10,12,15,15,21,4,5,6,20,35,56,64,125]

                        global connectivity = Vector[]
                        element_id = 1
                    elseif element_id > 0
                        temp = split(line)
                        element[element_id] = parse(Int32,temp[1])

                        push!(connectivity, map(x->parse(Int32, x), temp[2:type_map[elementType]+1]))

                        if element_id == numElementsInBlock
                            global block[block2_id] = ElementBlock(entityDim, entityTag, elementType, numElementsInBlock, connectivity)
                            global allTags[block2_id] = entityTag
                            element_id = 0
                            block2_id += 1
                        else
                            element_id += 1
                        end
                    end
                end
            end
        end

        s = 0
        v = 0

        # uniqueBoundaryNum = length(unique(values(surfToPhysiclTag)))-1
        uniqueBoundaryNum = length(unique(values(surfToPhysiclTag)))
        # print(surfToPhysiclTag)

        internals = [k for (k, v) in surfToPhysiclTag if v==0]
        #Dealing with the pesky internal surfaces from gmsh
        if length(internals) > 0
            uniqueBoundaryNum -= 1
        end

        sblock = Array{ElementBlock}(undef,uniqueBoundaryNum)
        vblock = Array{ElementBlock}(undef,1)

        for i = 1:length(block)
            if block[i].entityDim == dim - 1
                s += 1
                if surfToPhysiclTag[allTags[i]] != 0
                    if !isassigned(sblock,Int(surfToPhysiclTag[allTags[i]]))
                        sblock[surfToPhysiclTag[allTags[i]]] = block[i]
                        sblock[surfToPhysiclTag[allTags[i]]].entityTag = surfToPhysiclTag[allTags[i]]
                    else
                        connectS = sblock[surfToPhysiclTag[allTags[i]]].connectivity
                        connectB = block[i].connectivity
                        sblock[surfToPhysiclTag[allTags[i]]].connectivity = [connectS;connectB]
                        sblock[surfToPhysiclTag[allTags[i]]].numElementsInBlock = size([connectS;connectB],1)
                    end
                end
            elseif block[i].entityDim == dim
                if v == 0
                    vblock[1] = block[i]
                    v = 1
                else
                    connectV = vblock[1].connectivity
                    connectB = block[i].connectivity
                    vblock[1].connectivity = [connectV;connectB]
                    vblock[1].numElementsInBlock = size([connectV;connectB],1)
                end
            end
        end

        thisMesh = Mesh(nodes, sblock, vblock)

    end
    return thisMesh
end
