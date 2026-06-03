

function read_FOAM3D(file_path, scale, integer, float)

    points_file = joinpath(file_path,"points")
    faces_file = joinpath(file_path,"faces")
    neighbour_file = joinpath(file_path,"neighbour")
    owner_file = joinpath(file_path,"owner")
    boundary_file = joinpath(file_path,"boundary")

    foamdata = FoamMeshData(integer, float)

    foamdata.points = read_points(points_file, scale, integer, float)
    foamdata.boundaries = read_boundary(boundary_file, integer, float)

    face_nodes, face_nodes_range = read_faces(faces_file, integer, float)
    face_neighbours = read_neighbour(neighbour_file, integer, float)
    face_owners = read_owner(owner_file, integer, float)

    assign_faces!(foamdata, face_nodes, face_nodes_range, face_neighbours, face_owners, integer)

    return foamdata
end

function assign_faces!(foamdata, face_nodes, face_nodes_range, face_neighbours, face_owners, TI)
    foamdata.n_faces = n_faces = length(face_owners)
    foamdata.n_ifaces = n_ifaces = length(face_neighbours)
    foamdata.n_bfaces = n_faces - n_ifaces
    foamdata.n_cells = max(maximum(face_owners), maximum(face_neighbours))

    foamdata.face_nodes = face_nodes
    foamdata.face_nodes_range = face_nodes_range
    foamdata.face_owner = face_owners

    face_neighbour = Vector{TI}(undef, n_faces)
    for fi ∈ 1:n_ifaces
        face_neighbour[fi] = face_neighbours[fi]
    end
    for fi ∈ (n_ifaces + 1):n_faces
        face_neighbour[fi] = face_owners[fi] # boundary: neighbour == owner
    end
    foamdata.face_neighbour = face_neighbour
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
# advance pos past non-digit bytes, then parse one non-negative integer
@inline function _next_uint(bytes::Vector{UInt8}, pos::Int, len::Int)
    while pos <= len && (bytes[pos] < 0x30 || bytes[pos] > 0x39)
        pos += 1
    end
    v = 0
    while pos <= len && bytes[pos] >= 0x30 && bytes[pos] <= 0x39
        v = 10v + (bytes[pos] - 0x30)
        pos += 1
    end
    return v, pos
end

# skip n newlines in bytes, return pos just after the nth newline
@inline function _skip_lines(bytes::AbstractVector{UInt8}, n::Int, len::Int)
    pos = 1
    skipped = 0
    while pos <= len && skipped < n
        if bytes[pos] == UInt8('\n')
            skipped += 1
        end
        pos += 1
    end
    return pos
end

function read_faces(file_path, TI, TF)
    # find count line (skips FoamFile header safely)
    startLine = 0
    nfaces = 0
    for (n, line) ∈ enumerate(eachline(file_path))
        line_content = tryparse(Int64, line)
        if line_content !== nothing
            startLine = n
            nfaces = TI(line_content)
            println("Number of faces to read: $line_content (from line: $startLine)")
            break
        end
    end
    println("Number of faces to read: $nfaces (after cleaning file)")

    bytes = read(file_path)
    len = length(bytes)
    pos = _skip_lines(bytes, startLine, len) # land just after count line

    face_nodes = Vector{TI}(undef, 0)
    sizehint!(face_nodes, 4 * Int(nfaces))
    face_nodes_range = Vector{UnitRange{TI}}(undef, nfaces)
    startIdx = one(TI)
    for facei ∈ 1:nfaces
        nnodes, pos = _next_uint(bytes, pos, len) # per-face node count
        for i ∈ 1:nnodes
            nid, pos = _next_uint(bytes, pos, len)
            push!(face_nodes, TI(nid) + one(TI)) # +1 shift
        end
        endIdx = startIdx + TI(nnodes) - one(TI)
        face_nodes_range[facei] = UnitRange{TI}(startIdx, endIdx)
        startIdx = endIdx + one(TI)
    end

    return face_nodes, face_nodes_range
end

function read_neighbour(file_path, TI, TF)
    nfaces = 0
    startLine = 0
    for (n, line) ∈ enumerate(eachline(file_path))
        if isnothing(tryparse(TI, line))
            continue
        else
            nfaces = parse(TI, line)
            startLine = n
            println("number of neighbours/owners is ", nfaces)
            break
        end
    end

    face_neighbour_cell = Vector{TI}(undef, nfaces)
    bytes = read(file_path)
    len = length(bytes)
    pos = _skip_lines(bytes, startLine, len)

    for i ∈ 1:nfaces
        v, pos = _next_uint(bytes, pos, len)
        face_neighbour_cell[i] = TI(v) + one(TI) # +1 shift
    end
    return face_neighbour_cell
end

function read_owner(file_path, TI, TF)
    face_owner_cell = read_neighbour(file_path, TI, TF)
end

function read_points(file_path, scale, TI, TF)
    npoints = 0
    startLine = 0
    for (n, line) ∈ enumerate(eachline(file_path))
        if isnothing(tryparse(TI, line))
            continue
        else
            npoints = parse(TI, line)
            startLine = n
            println("number of points is ", npoints)
            break
        end
    end

    # read as String for SubString/parse compatibility (bit-identical floats)
    file_str = read(file_path, String)
    bytes = codeunits(file_str) # byte view for scanning
    len = length(bytes)
    pos = _skip_lines(bytes, startLine, len)

    zvec = zeros(TF, 3)
    points = [SVector{3}(zvec) for _ ∈ 1:npoints]
    for pi ∈ 1:npoints
        # skip to point '(' then collect 3 separator-delimited float tokens
        while pos <= len && bytes[pos] != UInt8('('); pos += 1; end
        pos += 1 # skip '('
        comp = MVector{3,TF}(undef)
        for ci ∈ 1:3
            # skip separators: '(' ')' space tab newline cr all delimit tokens
            while pos <= len && (bytes[pos] == UInt8('(') || bytes[pos] == UInt8(')') || bytes[pos] == UInt8(' ') || bytes[pos] == UInt8('\t') || bytes[pos] == UInt8('\n') || bytes[pos] == UInt8('\r')); pos += 1; end
            lo = pos
            # advance to next separator
            while pos <= len && bytes[pos] != UInt8('(') && bytes[pos] != UInt8(')') && bytes[pos] != UInt8(' ') && bytes[pos] != UInt8('\t') && bytes[pos] != UInt8('\n') && bytes[pos] != UInt8('\r'); pos += 1; end
            comp[ci] = scale * parse(TF, SubString(file_str, lo, pos - 1))
        end
        points[pi] = SVector{3}(comp)
    end
    return points
end
