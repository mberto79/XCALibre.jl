export load

function load(meshFile, TI, TF)
    blockStart = false
    processDataset2411 = false
    processDataset2412 = false
    processDataset2467 = false
    
    points = UNV.Point{TF}[] # Array to hold points
    elements = UNV.Element{TI}[] # Array to hold elements
    boundaries = UNV.Boundary{TI}[] # Array to hold boundaries
    
    index = 0
    vertexCount = 0
    vertices = TI[]
    newBoundary = UNV.Boundary(0)
    currentBC = 0
    
    @inbounds for (indx, line) in enumerate(eachline(meshFile))
        sline = split(line)
    
        if !blockStart && sline[1] == "-1"
            blockStart = true
            # println("Block start found, ", indx)
            continue
        end
        if blockStart && sline[1] == "-1"
            blockStart = false
            processDataset2411 = false
            processDataset2412 = false
            processDataset2467 = false
            # println("Block END found, ", indx)
            continue
        end
        if blockStart && sline[1] == "2411" && !processDataset2411 && !processDataset2412 &&
            !processDataset2467
            println("Reading nodes (dataset 2411), from line ", indx)
            processDataset2411 = true
            continue
        end
        if blockStart && sline[1] == "2412" && !processDataset2411 && !processDataset2412 &&
            !processDataset2467
            println("Reading elements (dataset 2412), from line ", indx)
            processDataset2412 = true
            continue
        end
        if blockStart && sline[1] == "2467" && !processDataset2411 && !processDataset2412 &&
            !processDataset2467
            println("Reading boundary information (dataset 2467), from line ", indx)
            processDataset2467 = true
            continue
        end
        # Read points
        if processDataset2411
            if length(split(sline[1], ".")) > 1
                x = parse(TF, sline[1])
                y = parse(TF, sline[2])
                z = parse(TF, sline[3])
                push!(points, UNV.Point(SVector{3, TF}(x, y, z)))
                continue
            end
        end
        # Read elements
        if processDataset2412
            if parse(TI, sline[1]) == 0
                continue
            end
            if length(sline) == 6 && parse(Int32, sline[end]) == 2
            vertexCount = parse(TI, sline[end])
            index = parse(TI, sline[1])
            continue
            end
            if length(sline) == 2
                vertices = [parse(TI, sline[i]) for i=1:length(sline)]
                push!(elements, UNV.Element(index, vertexCount, vertices))
                continue
            end
            if length(sline) == 6 && parse(Int32, sline[end]) == 3
                vertexCount = parse(TI, sline[end])
                index = parse(TI, sline[1])
                continue
            end
            if length(sline) == 3
                vertices = [parse(TI, sline[i]) for i=1:length(sline)]
                push!(elements, UNV.Element(index, vertexCount, vertices))
                continue
            end
            if length(sline) == 6 && parse(TI, sline[end]) == 4
                vertexCount = parse(TI, sline[end])
                index = parse(TI, sline[1])
                continue
            end
            if length(sline) == 4
                vertices = [parse(TI, sline[i]) for i=1:length(sline)]
                push!(elements, UNV.Element(index, vertexCount, vertices))
                continue
            end
        end
        # Read boundary cells
        if processDataset2467
            if typeof(tryparse(TI, sline[1]))!= Nothing && tryparse(Int32, sline[2]) == 0
                newBoundary = UNV.Boundary(0)
                push!(boundaries, newBoundary)
                currentBC = tryparse(TI, sline[1])
                boundaries[currentBC].groupNumber = currentBC 
                continue
            end
            if typeof(tryparse(TI, sline[1]))== Nothing
                println("Boundary name: ", sline[1])
                boundaries[currentBC].name = sline[1]
                continue
            end
            push!(boundaries[currentBC].elements, parse(TI, sline[2]))
            if length(sline) >= 6
                push!(boundaries[currentBC].elements, parse(TI, sline[6]))
            end   
        end
    end
    return points, elements, boundaries
    end