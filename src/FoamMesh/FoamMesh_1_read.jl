

function read_FOAM3D(file_path, scale, integer, float)

    points_file = joinpath(file_path,"points")
    faces_file = joinpath(file_path,"faces")
    neighbour_file = joinpath(file_path,"neighbour")
    owner_file = joinpath(file_path,"owner")
    boundary_file = joinpath(file_path,"boundary")

    foamdata = FoamMeshData(integer, float)

    foamdata.points = read_points(points_file, scale, integer, float)
    foamdata.boundaries = read_boundary(boundary_file, integer, float)

    face_nodes = read_faces(faces_file, integer, float)
    face_neighbours = read_neighbour(neighbour_file, integer, float)
    face_owners = read_owner(owner_file, integer, float)

    assign_faces!(foamdata, face_nodes, face_neighbours, face_owners, integer)

    return foamdata
end

function assign_faces!(foamdata, face_nodes, face_neighbours, face_owners, TI)
    foamdata.n_faces = n_faces = length(face_owners)
    foamdata.n_ifaces = n_ifaces = length(face_neighbours)
    foamdata.n_bfaces = n_bfaces = foamdata.n_faces - foamdata.n_ifaces
    foamdata.n_cells = max(maximum(face_owners), maximum(face_neighbours))

    foamdata.faces = [Face(TI, length(nodesID)) for nodesID ∈ face_nodes]

    # p1 = one(eltype(face_owners))

    for (nIDs, oID, nID, face) ∈ zip(face_nodes, face_owners, face_neighbours, foamdata.faces)
        face.nodesID = nIDs
        face.owner = oID
        face.neighbour = nID
    end

    for fID ∈ (n_ifaces + 1):n_faces
        face = foamdata.faces[fID]
        face.nodesID = face_nodes[fID]
        face.owner = face_owners[fID]
        face.neighbour = face_owners[fID]
    end
    
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

    boundaries = [Boundary(TI) for _ ∈ 1:nBoundaries]

    bcounter = 0
    for (n, line) ∈ enumerate(eachline(file_path)) 
        if line == ")"
            break 
        elseif n > readfrom
            sline = split(line, delimiters, keepempty=false)

            if length(sline) == 1
                bcounter += 1
                boundaries[bcounter].name = Symbol(sline[1])
                continue
            end

            if length(sline) == 2 && sline[1] == "nFaces"
                boundaries[bcounter].nFaces = parse(TI, sline[2])
                continue
            end

            if length(sline) == 2 && sline[1] == "startFace"
                boundaries[bcounter].startFace = parse(TI, sline[2]) + one(TI) # make 1-indexed
                continue
            end

        end
    end
    return boundaries
end

function read_faces(file_path, TI, TF)
    # Version 3

    # Find line number with the entry giving total number of faces
    startLine = 0
    for (n, line) ∈ enumerate(eachline(file_path)) 
        line_content = tryparse(Int64, line)
        if line_content !== nothing
            startLine = n
            println("Number of faces to read: $line_content (from line: $startLine)")
            break
        end
    end

    # Read file contents skipping header information (using startLine from above)
    io = IOBuffer()
    for (n, line) ∈ enumerate(eachline(file_path)) 
        if n >= startLine
            println(io, line)
        end
    end

    file_data = String(take!(io)) # Convert IOBuffer to String
    delimiters = ['(',' ', ')', '\n']
    data_split = split(file_data, delimiters, keepempty=false)
    data = tryparse.(Int64, data_split)
    dataClean = filter(!isnothing, data)
    nfaces = dataClean[1]
    println("Number of faces to read: $nfaces (after cleaning file)")
    face_nodes = [TI[] for _ ∈ 1:nfaces]

    sizeIndex = 2 # counter to provide index where number of nodes data is stored
    for facei ∈ eachindex(face_nodes)
        nnodes = dataClean[sizeIndex]
        faceNodes = zeros(TI, nnodes)
        for i ∈ 1:nnodes
            faceNodes[i] = dataClean[sizeIndex + i] .+ one(TI)
        end
        sizeIndex += nnodes + 1
        face_nodes[facei] = faceNodes
    end

    
    # Version 2
    # delimiters = ['(',' ', ')', '\n']

    # file_data = read(file_path, String)
    # data_split = split(file_data, delimiters, keepempty=false)
    # data = tryparse.(Int64, data_split)
    # dataClean = filter(!isnothing, data)
    # nfaces = dataClean[2]
    # println("number of faces is ", nfaces)
    # face_nodes = [TI[] for _ ∈ 1:nfaces]

    # sizeIndex = 3
    # for facei ∈ eachindex(face_nodes)
    #     nnodes = dataClean[sizeIndex]
    #     faceNodes = zeros(TI, nnodes)
    #     for i ∈ 1:nnodes
    #         faceNodes[i] = dataClean[sizeIndex + i] .+ one(TI)
    #     end
    #     sizeIndex += nnodes + 1
    #     face_nodes[facei] = faceNodes
    # end

    # OLD VERSION

    # delimiters = ['(',' ', ')']

    # # find the total number of faces and line to start reading data from
    # nfaces = 0
    # readfrom = 0
    # for (n, line) ∈ enumerate(eachline(file_path))
    #     if isnothing(tryparse(TI, line))
    #         continue
    #     else 
    #         nfaces = parse(TI, line)
    #         readfrom = n + 1
    #         println("number of faces is ", nfaces)
    #         break
    #     end
    # end

    # # face_nodes = Vector{TI}[]
    # face_nodes = [TI[] for _ ∈ 1:nfaces]
    # fcounter = 0
    # for (n, line) ∈ enumerate(eachline(file_path)) 
    #     if line == ")"
    #         break 
    #     elseif n > readfrom
    #         fcounter += 1
    #         sline = split(line, delimiters, keepempty=false)
    #         nodesIDs = parse.(TI, sline[2:end]) .+ one(TI) # make 1-indexed
    #         face_nodes[fcounter] = nodesIDs
    #     end
    # end

    
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
            println("number of neighbours/owners is ", nfaces)
            break
        end
    end

    face_neighbour_cell = zeros(TI, nfaces)

    fcounter = 0
    for (n, line) ∈ enumerate(eachline(file_path)) 
        if line == ")"
            break 
        elseif n > readfrom
            # fcounter += 1
            line_data = split(line, keepempty=false)
            # face_neighbour_cell[fcounter] = parse(TI, p[1]) + one(TI) # make 1-indexed
            for data ∈ line_data 
                fcounter += 1
                cellID = parse(TI, data) + one(TI) # make 1-indexed
                face_neighbour_cell[fcounter] = cellID 
            end
        end
    end
    return face_neighbour_cell
end

function read_owner(file_path, TI, TF)
    face_owner_cell = read_neighbour(file_path, TI, TF)
end

function read_points(file_path, scale, TI, TF)
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
            points[pcounter] = @inbounds SVector{3}(scale*parse.(TF, p))
        end
    end
    return points
end