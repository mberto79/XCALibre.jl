
mutable struct FoamMeshData{B,P,F,I}
    boundaries::B
    points::P
    faces::F
    n_cells::I
    n_faces::I
    n_ifaces::I
    n_bfaces::I
end
FoamMeshData(TI, TF) = FoamMeshData(
        Boundary{TI,Symbol}[],
        SVector{3, TF}[],
        Face{TI}[],
        zero(TI),
        zero(TI),
        zero(TI),
        zero(TI)
    )

mutable struct Boundary{I<:Integer, S<:Symbol}
    name::S
    startFace::I
    nFaces::I
end
Boundary(TI) = Boundary(:default, zero(TI), zero(TI))

mutable struct Face{I}
    nodesID::Vector{I}
    owner::I
    neighbour::I
end
Face(nnodes::I) where I<:Integer = begin
    nodesIDs = zeros(I, nnodes)
    z = zero(I)
    Face(nodesIDs, z, z)
end

function read_foamMesh(file_path, integer, float)

    points_file = joinpath(file_path,"points")
    faces_file = joinpath(file_path,"faces")
    neighbour_file = joinpath(file_path,"neighbour")
    owner_file = joinpath(file_path,"owner")
    boundary_file = joinpath(file_path,"boundary")

    foamdata = FoamMeshData(integer, float)

    foamdata.points = read_points(points_file, integer, float)
    foamdata.boundaries = read_boundary(boundary_file, integer, float)

    face_nodes = read_faces(faces_file, integer, float)
    face_neighbours = read_neighbour(neighbour_file, integer, float)
    face_owners = read_owner(owner_file, integer, float)

    assign_faces!(foamdata, face_nodes, face_neighbours, face_owners)

    return foamdata
end

function assign_faces!(foamdata, face_nodes, face_neighbours, face_owners)
    foamdata.n_faces = n_faces = length(face_owners)
    foamdata.n_ifaces = n_ifaces = length(face_neighbours)
    foamdata.n_bfaces = n_bfaces = foamdata.n_faces - foamdata.n_ifaces
    foamdata.n_cells = max(maximum(face_owners), maximum(face_neighbours))

    foamdata.faces = [Face(length(nodesID)) for nodesID ∈ face_nodes]

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
            nodesIDs = parse.(TI, sline[2:end]) .+ one(TI) # make 1-indexed
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