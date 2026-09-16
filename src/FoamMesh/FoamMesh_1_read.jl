

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

function _foam_tokens(file_path)
    contents = read(file_path, String)
    contents = replace(contents, r"(?s)/\*.*?\*/" => " ")
    contents = replace(contents, r"//[^\r\n]*" => " ")
    token_pattern = r"\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'|[{}();]|[^\s{}();]+"
    return [match.match for match in eachmatch(token_pattern, contents)]
end

@inline function _foam_name(token)
    if length(token) >= 2 && ((first(token) == '"' && last(token) == '"') ||
                              (first(token) == '\'' && last(token) == '\''))
        return token[2:(end - 1)]
    end
    return token
end

function _parse_boundary_integer(token, TI, field, patch_name)
    value = tryparse(TI, token)
    isnothing(value) && throw(ArgumentError(
        "invalid $field value '$token' for OpenFOAM boundary '$patch_name'",
    ))
    value < zero(TI) && throw(ArgumentError(
        "$field must be non-negative for OpenFOAM boundary '$patch_name'",
    ))
    return value
end

function read_boundary(file_path, TI, TF)
    tokens = _foam_tokens(file_path)

    # The patch count is the first integer token immediately followed by the
    # outer list opener. This deliberately skips the FoamFile header dictionary.
    count_index = findfirst(eachindex(tokens)) do index
        index < length(tokens) && tryparse(TI, tokens[index]) !== nothing &&
            tokens[index + 1] == "("
    end
    isnothing(count_index) && throw(ArgumentError(
        "could not find the declared OpenFOAM boundary list in '$file_path'",
    ))

    n_boundaries = parse(TI, tokens[count_index])
    n_boundaries < zero(TI) && throw(ArgumentError(
        "the OpenFOAM boundary count must be non-negative in '$file_path'",
    ))
    boundaries = Vector{Boundary{TI,Symbol}}()
    sizehint!(boundaries, Int(n_boundaries))
    names = Set{Symbol}()
    index = count_index + 2

    for patch_index in 1:Int(n_boundaries)
        while index <= length(tokens) && tokens[index] == ";"
            index += 1
        end
        index > length(tokens) && throw(ArgumentError(
            "OpenFOAM boundary list ended after $(patch_index - 1) of $n_boundaries patches",
        ))
        tokens[index] == ")" && throw(ArgumentError(
            "OpenFOAM boundary list contains $(patch_index - 1) patches, expected $n_boundaries",
        ))

        patch_name = Symbol(_foam_name(tokens[index]))
        patch_name in names && throw(ArgumentError(
            "duplicate OpenFOAM boundary name '$patch_name'",
        ))
        index += 1
        index <= length(tokens) && tokens[index] == "{" || throw(ArgumentError(
            "expected a dictionary for OpenFOAM boundary '$patch_name'",
        ))
        index += 1

        brace_depth = 1
        n_faces = nothing
        start_face = nothing
        while index <= length(tokens) && brace_depth > 0
            token = tokens[index]
            if token == "{"
                brace_depth += 1
            elseif token == "}"
                brace_depth -= 1
            elseif brace_depth == 1 && (token == "nFaces" || token == "startFace")
                index == length(tokens) && throw(ArgumentError(
                    "missing value for $token in OpenFOAM boundary '$patch_name'",
                ))
                value = _parse_boundary_integer(tokens[index + 1], TI, token, patch_name)
                if token == "nFaces"
                    isnothing(n_faces) || throw(ArgumentError(
                        "duplicate nFaces entry for OpenFOAM boundary '$patch_name'",
                    ))
                    n_faces = value
                else
                    isnothing(start_face) || throw(ArgumentError(
                        "duplicate startFace entry for OpenFOAM boundary '$patch_name'",
                    ))
                    start_face = value
                end
                index += 1
            end
            index += 1
        end
        brace_depth == 0 || throw(ArgumentError(
            "unterminated dictionary for OpenFOAM boundary '$patch_name'",
        ))
        isnothing(n_faces) && throw(ArgumentError(
            "missing nFaces entry for OpenFOAM boundary '$patch_name'",
        ))
        isnothing(start_face) && throw(ArgumentError(
            "missing startFace entry for OpenFOAM boundary '$patch_name'",
        ))
        start_face == typemax(TI) && throw(ArgumentError(
            "startFace overflows one-based indexing for OpenFOAM boundary '$patch_name'",
        ))
        n_faces > zero(TI) && start_face > typemax(TI) - n_faces && throw(ArgumentError(
            "face range overflows for OpenFOAM boundary '$patch_name'",
        ))

        push!(names, patch_name)
        push!(boundaries, Boundary{TI,Symbol}(patch_name, start_face + one(TI), n_faces))
    end

    while index <= length(tokens) && tokens[index] == ";"
        index += 1
    end
    index <= length(tokens) && tokens[index] == ")" || throw(ArgumentError(
        "OpenFOAM boundary list declares $n_boundaries patches but contains additional or malformed entries",
    ))
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
