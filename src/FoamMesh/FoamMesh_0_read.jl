function read_foamMesh(file_path; integer, float)

    points = read_points(joinpath(file_path,"points"), integer, float)
    face_nodes = read_faces(joinpath(file_path,"faces"), integer, float)
    face_neighbour_cell = read_neighbour(joinpath(file_path,"neighbour"), integer, float)
    face_owner_cell = read_owner(joinpath(file_path,"owner"), integer, float)

    bnames, bnFaces, bstartFace = begin
        read_boundary(joinpath(file_path,"boundary"), integer, float)
    end

    return points, face_nodes, face_neighbour_cell, face_owner_cell, bnames, bnFaces, bstartFace
end

function read_boundary(file_path, TI, TF)
    delimiters = [' ', ';', '{', '}']

    # find the total number of boundaries and line to start reading data from
    nBoundaries = 0
    readfrom = 0
    for (n, line) ∈ enumerate(eachline(file_path))
        if isnothing(tryparse(TI, line))
            continue
        else 
            nBoundaries = parse(TI, line)
            readfrom = n + 1
            println("number of boundaries is ", nBoundaries)
            break
        end
    end

    names = Symbol[:dummy for _ ∈ 1:nBoundaries] 
    nFaces = zeros(TI, nBoundaries)
    startFace = zeros(TI, nBoundaries)

    bcounter = 0
    for (n, line) ∈ enumerate(eachline(file_path)) 
        if line == ")"
            break 
        elseif n > readfrom
            sline = split(line, delimiters, keepempty=false)

            if length(sline) == 1
                bcounter += 1
                names[bcounter] = Symbol(sline[1])
                continue
            end

            if length(sline) == 2 && sline[1] == "nFaces"
                nFaces[bcounter] = parse(TI, sline[2])
                continue
            end

            if length(sline) == 2 && sline[1] == "startFace"
                startFace[bcounter] = parse(TI, sline[2]) + one(TI) # make 1-indexed
                continue
            end

        end
    end
    return names, nFaces, startFace
end

function read_faces(file_path, TI, TF)
    delimiters = ['(',' ', ')']

    # find the total number of faces and line to start reading data from
    nfaces = 0
    readfrom = 0
    for (n, line) ∈ enumerate(eachline(file_path))
        if isnothing(tryparse(TI, line))
            continue
        else 
            nfaces = parse(TI, line)
            readfrom = n + 1
            println("number of faces is ", nfaces)
            break
        end
    end

    face_nodes = Vector{TI}[]
    fcounter = 0
    for (n, line) ∈ enumerate(eachline(file_path)) 
        if line == ")"
            break 
        elseif n > readfrom
            fcounter += 1
            sline = split(line, delimiters, keepempty=false)
            # nFaceNodes = parse(TI, sline[1])
            nodesIDs = parse.(TI, sline[2:end])
            # totalFaceNodes += nFaceNodes
            # face_nodes_range[fcounter] = UnitRange{TI}(
            # totalFaceNodes - nFaceNodes + 1, totalFaceNodes)
            push!(face_nodes, nodesIDs)
        end
    end

    # extract total number of nodes and define ranges for each face to access them
    # face_nodes_range = UnitRange{TI}[1:2 for _ ∈ 1:nfaces]
    # totalFaceNodes = 0
    # fcounter = 0
    # for (n, line) ∈ enumerate(eachline(file_path)) 
    #     if line == ")"
    #         break 
    #     elseif n > readfrom
    #         fcounter += 1
    #         sline = split(line, delimiters, keepempty=false)
    #         nFaceNodes = parse(TI, sline[1])
    #         totalFaceNodes += nFaceNodes
    #         face_nodes_range[fcounter] = UnitRange{TI}(
    #             totalFaceNodes - nFaceNodes + 1, totalFaceNodes)
    #     end
    # end

    # loop through data again and store nodes IDs
    # face_nodes = zeros(TI, totalFaceNodes)
    # nodei = 0
    # for (n, line) ∈ enumerate(eachline(file_path)) 
    #     if line == ")"
    #         break 
    #     elseif n > readfrom
    #         sline = split(line, delimiters, keepempty=false)
    #         nodesIDs = parse.(TI, sline[2:end])
    #         for nodeID ∈ nodesIDs
    #             nodei += 1
    #             face_nodes[nodei] = nodeID + one(TI) # make 1-indexed
    #         end
    #     end
    # end
    # return face_nodes, face_nodes_range
    return face_nodes
end

function read_neighbour(file_path, TI, TF)
    nfaces = 0
    readfrom = 0
    for (n, line) ∈ enumerate(eachline(file_path))
        if isnothing(tryparse(TI, line))
            continue
        else 
            nfaces = parse(TI, line)
            readfrom = n + 1
            println("number of faces is ", nfaces)
            break
        end
    end

    face_neighbour_cell = zeros(TI, nfaces)

    fcounter = 0
    for (n, line) ∈ enumerate(eachline(file_path)) 
        if line == ")"
            break 
        elseif n > readfrom
            fcounter += 1
            p = split(line, keepempty=false)
            face_neighbour_cell[fcounter] = parse(TI, p[1]) + one(TI) # make 1-indexed
        end
    end
    return face_neighbour_cell
end

function read_owner(file_path, TI, TF)
    face_owner_cell = read_neighbour(file_path, TI, TF)
end

function read_points(file_path, TI, TF)
    delimiters = ['(',' ', ')']
    npoints = 0
    readfrom = 0
    for (n, line) ∈ enumerate(eachline(file_path))
        if isnothing(tryparse(TI, line))
            continue
        else 
            npoints = parse(TI, line)
            readfrom = n + 1
            println("number of points is ", npoints)
            break
        end
    end

    zvec = zeros(TF,3)
    points = [SVector{3}(zvec) for _ ∈ 1:npoints]
    pcounter = 0
    for (n, line) ∈ enumerate(eachline(file_path)) 
        if line == ")"
            break 
        elseif n > readfrom
            pcounter += 1
            p = split(line, delimiters, keepempty=false)
            points[pcounter] = @inbounds SVector{3}(parse.(TF, p))
        end
    end
    return points
end