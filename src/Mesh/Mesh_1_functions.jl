export get_boundaries
export _get_float, _get_int, _get_backend, _convert_array!
export bounding_box
export boundary_info, boundary_map
export total_boundary_faces, boundary_index
export norm_static
export is_boundary
export convert_mesh_float
export validate_single_precision_mesh
# export x, y, z # access cell centres
# export xf, yf, zf # access face centres

_get_int(mesh) = eltype(mesh.get_int)
_get_float(mesh) = eltype(mesh.get_float)
_get_backend(mesh) = get_backend(mesh.cells)

# Boundary faces store their owner cell twice: every mesh reader sets ownerCells this way
# (UNV2, UNV3, FoamMesh), and the MPI path must do the same for processor faces.
is_boundary(ownerCells::SVector{2,<:Integer}) = ownerCells[1] == ownerCells[2]
is_boundary(face::Union{Face2D,Face3D}) = is_boundary(face.ownerCells)

# Laplacian face coefficient. norm(((Sf.Sf)/(Sf.e))*e)/delta reduces to area/(|normal.e|*delta)
# because ns cancels and normal and e are unit vectors. Boundary faces keep area/delta, which
# is what every @define_boundary Laplacian block uses.
_gDiff(ownerCells, normal, e, area, delta) = begin
    den = is_boundary(ownerCells) ? delta : abs(normal ⋅ e)*delta
    den > zero(den) ? area/den : zero(den)
end

_gDiff(face::Union{Face2D,Face3D}) =
    _gDiff(face.ownerCells, face.normal, face.e, face.area, face.delta)

face_gDiff_coefficients(faces) = _gDiff.(faces)

# the 3D readers build the mesh before filling the face geometry, so the array built with it
# is refreshed once the geometry is final
update_face_gDiff!(mesh) = (mesh.face_gDiff .= _gDiff.(mesh.faces); mesh)

# Internal face properties from distance vectors: C1F1/C2F1 = cell1/cell2 centre to face centre,
# C1C2 = cell1 centre to cell2 centre
weight_delta_e(C1F1, C2F1, C1C2, normal) = begin
    # weight = norm(C2F1)/(norm(C1F1) + norm(C2F1)) # face-distance based
    projection = C1C2⋅normal
    if isfinite(projection) && abs(projection) > eps(projection)
        wi = (C1F1⋅normal)/projection
        weight = one(wi) - wi # normal aligned interpolation weight
    else
        d1 = norm(C1F1)
        d2 = norm(C2F1)
        dsum = d1 + d2
        weight = dsum > zero(dsum) ? d2/dsum : oftype(dsum, 0.5)
    end
    delta = norm(C1C2)
    if delta > zero(delta)
        e = C1C2/delta
    else
        delta = norm(C1F1) + norm(C2F1) # fallback when cell centres coincide (degenerate cell)
        e = normal
    end
    delta = max(delta, eps(one(delta))) # keep delta > 0 for degenerate faces (area is 0 there)
    weight = clamp(weight, zero(weight), one(weight)) # keep interpolation weight physical on skewed cells
    return weight, delta, e
end

# function to calculate boundary face properties
weight_delta_e(C1F1, normal) = begin
    weight = one(eltype(C1F1))
    delta = norm(C1F1)
    e = delta > zero(delta) ? C1F1/delta : normal
    delta = max(delta, eps(one(delta))) # keep delta > 0 for degenerate faces (area is 0 there)
    return weight, delta, e
end

# Face orientation from topology. Testing a normal against an estimated cell centre fails on
# skewed or concave cells, where the estimate can lie on the wrong side of a face; these
# functions use only how the faces of one closed cell connect, then fix the overall sign with
# the divergence theorem (positive area or volume), so the result holds for any cell shape.
# Each returns (signs, ok): signs[i] is +1 when face i, in its stored node order, points out of
# the cell and -1 when it points in; ok is false when the faces do not close up (then signs is
# meaningless and the caller falls back to another test).

# 2D (x-y plane): edges[i] = (a, b) node IDs. An edge taken from a to b points out of the cell
# along (b - a) × k when the edges chain head to tail anticlockwise.
function _outward_edge_signs(coords, edges)
    n = length(edges)
    signs = zeros(Int8, n)
    n >= 3 || return signs, false
    signs[1] = 1
    start, head = edges[1]
    @inbounds for _ in 2:n
        found = false
        for j in 2:n
            signs[j] == 0 || continue
            a, b = edges[j]
            if a == head
                signs[j] = 1; head = b; found = true; break
            elseif b == head
                signs[j] = -1; head = a; found = true; break
            end
        end
        found || return signs, false
    end
    head == start || return signs, false
    area2 = zero(eltype(coords[edges[1][1]]))
    @inbounds for j in 1:n
        a, b = signs[j] > 0 ? edges[j] : (edges[j][2], edges[j][1])
        pa, pb = coords[a], coords[b]
        area2 += pa[1]*pb[2] - pb[1]*pa[2]
    end
    area2 == zero(area2) && return signs, false
    area2 < zero(area2) && (signs .= -signs)
    return signs, true
end

# 2D mesh builders (UNV2, BlockMesher2D): for every face (edge), the sign s such that
# s*((p2 - p1) × k), with p1, p2 its nodes in stored order, points out of its owner cell, i.e.
# from owner to neighbour. 0 marks a face whose owner and neighbour cells both fail to close
# up; the caller orients it another way. cells[c].facesID lists internal faces only, so each
# cell's boundary faces are added from the boundaries.
function _owner_outward_signs_2d(cells, faces, boundaries, nodes)
    cell_edges = [collect(cell.facesID) for cell in cells]
    for boundary in boundaries
        for (c, f) in zip(boundary.cellsID, boundary.facesID)
            push!(cell_edges[c], f)
        end
    end
    coords = [node.coords for node in nodes]
    owner_sign = zeros(Int8, length(faces))
    neighbour_sign = zeros(Int8, length(faces))
    for (c, fIDs) in enumerate(cell_edges)
        edges = [(faces[f].nodesID[1], faces[f].nodesID[2]) for f in fIDs]
        signs, ok = _outward_edge_signs(coords, edges)
        ok || continue
        for (i, f) in enumerate(fIDs)
            if faces[f].ownerCells[1] == c
                owner_sign[f] = signs[i]
            else
                neighbour_sign[f] = signs[i]
            end
        end
    end
    return [owner_sign[f] != 0 ? owner_sign[f] : -neighbour_sign[f] for f in eachindex(faces)]
end

# 3D: face_nodes[i] = node IDs of face i in stored order, area_vectors[i] its area vector (right-
# hand rule on that order) and centres[i] its centre. In a closed cell every edge belongs to two
# faces that traverse it in opposite directions when both point outwards.
function _outward_face_signs(face_nodes, area_vectors, centres)
    n = length(face_nodes)
    signs = zeros(Int8, n)
    n >= 4 || return signs, false
    signs[1] = 1
    stack = [1]
    @inbounds while !isempty(stack)
        f = pop!(stack)
        nf = face_nodes[f]; kf = length(nf)
        for i in 1:kf
            p = nf[i]; q = nf[i == kf ? 1 : i + 1]
            for g in 1:n
                g == f && continue
                ng = face_nodes[g]; kg = length(ng)
                for j in 1:kg
                    u = ng[j]; v = ng[j == kg ? 1 : j + 1]
                    s = (u == p && v == q) ? -signs[f] : (u == q && v == p) ? signs[f] : Int8(0)
                    s == 0 && continue
                    if signs[g] == 0
                        signs[g] = s; push!(stack, g)
                    elseif signs[g] != s
                        return signs, false # edge used inconsistently: not a closed manifold
                    end
                end
            end
        end
    end
    any(iszero, signs) && return signs, false
    p0 = centres[1]
    volume3 = zero(eltype(p0))
    @inbounds for i in 1:n
        volume3 += signs[i]*(area_vectors[i] ⋅ (centres[i] - p0))
    end
    volume3 == zero(volume3) && return signs, false
    volume3 < zero(volume3) && (signs .= -signs)
    return signs, true
end

function face_geometry(nodes, nIDs, apex::SVector{3, TF}) where {TF<:AbstractFloat}
    area_vector = SVector{3, TF}(0, 0, 0)
    n_nodes = length(nIDs)
    @inbounds for i in 1:n_nodes
        inext = i == n_nodes ? 1 : i + 1
        point = nodes[nIDs[i]].coords
        next_point = nodes[nIDs[inext]].coords
        area_vector += ((point - apex) × (next_point - apex))/TF(2)
    end

    area = norm(area_vector)
    normal = area > zero(TF) ? area_vector/area : SVector{3, TF}(0, 0, 0)
    centre_sum = SVector{3, TF}(0, 0, 0)
    projected_area = zero(TF)
    @inbounds for i in 1:n_nodes
        inext = i == n_nodes ? 1 : i + 1
        point = nodes[nIDs[i]].coords
        next_point = nodes[nIDs[inext]].coords
        triangle_vector = ((point - apex) × (next_point - apex))/TF(2)
        weight = triangle_vector ⋅ normal
        projected_area += weight
        centre_sum += weight*(apex + point + next_point)/TF(3)
    end
    centre = projected_area > floatmin(TF) ? centre_sum/projected_area : apex
    return normal, area, centre
end

# Reorders face nodes so every normal points out of the owner cell (from owner to neighbour).
# Each face takes its orientation from the topology of its owner cell, or of its neighbour when
# the owner's faces do not close up; only when neither closes does the old test against the
# estimated cell centres decide, with a warning.
function _orient_faces_3d!(mesh::Mesh3, centre_estimates)
    (; cells, faces, face_nodes, boundary_cellsID) = mesh
    TF = _get_float(mesh)
    n_cells = length(cells)
    n_bfaces = length(boundary_cellsID)

    # all faces of every cell, owner and neighbour roles
    counts = zeros(Int, n_cells + 1)
    for (fID, face) in enumerate(faces)
        counts[face.ownerCells[1] + 1] += 1
        fID > n_bfaces && (counts[face.ownerCells[2] + 1] += 1)
    end
    offsets = cumsum(counts) # offsets[c]+1 : offsets[c+1] are the faces of cell c
    all_cell_faces = Vector{Int}(undef, offsets[end])
    cursor = offsets[1:n_cells]
    for (fID, face) in enumerate(faces)
        c = face.ownerCells[1]; cursor[c] += 1; all_cell_faces[cursor[c]] = fID
        if fID > n_bfaces
            c = face.ownerCells[2]; cursor[c] += 1; all_cell_faces[cursor[c]] = fID
        end
    end

    owner_sign = zeros(Int8, length(faces))
    neighbour_sign = zeros(Int8, length(faces))
    for c in 1:n_cells
        fIDs = @view all_cell_faces[(offsets[c] + 1):offsets[c + 1]]
        fnodes = [@view(face_nodes[faces[f].nodes_range]) for f in fIDs]
        areas = [faces[f].area*faces[f].normal for f in fIDs]
        centres = [faces[f].centre for f in fIDs]
        signs, ok = _outward_face_signs(fnodes, areas, centres)
        ok || continue
        for (i, f) in enumerate(fIDs)
            if faces[f].ownerCells[1] == c
                owner_sign[f] = signs[i]
            else
                neighbour_sign[f] = signs[i]
            end
        end
    end

    n_fallback = 0
    for (fID, face) in enumerate(faces)
        flip = if owner_sign[fID] != 0
            owner_sign[fID] < 0
        elseif neighbour_sign[fID] != 0
            neighbour_sign[fID] > 0
        else
            n_fallback += 1
            owner = face.ownerCells[1]
            direction = fID <= n_bfaces ?
                face.centre - centre_estimates[owner] :
                centre_estimates[face.ownerCells[2]] - centre_estimates[owner]
            direction ⋅ face.normal < zero(TF)
        end
        flip || continue
        reverse!(@view face_nodes[face.nodes_range])
        faces[fID] = Face3D(
            face.nodes_range, face.ownerCells, face.centre, -face.normal, face.e,
            face.area, face.delta, face.weight,
        )
    end
    n_fallback > 0 && @warn "compute_3d_geometry!: $n_fallback face(s) belong only to cells whose faces do not close up; they were oriented against estimated cell centres, which can fail on skewed cells."
    return mesh
end

# orient_faces=false keeps the stored face orientation, for formats that define it (OpenFOAM:
# right-hand rule points from owner to neighbour, and out of the domain on boundaries)
function compute_3d_geometry!(mesh::Mesh3; orient_faces::Bool=true)
    (; cells, faces, face_nodes, nodes, boundary_cellsID) = mesh
    TF = _get_float(mesh)
    n_cells = length(cells)
    n_bfaces = length(boundary_cellsID)

    for (fID, face) in enumerate(faces)
        nIDs = @view face_nodes[face.nodes_range]
        apex = sum(nodes[nID].coords for nID in nIDs)/TF(length(nIDs))
        normal, area, centre = face_geometry(nodes, nIDs, apex)
        faces[fID] = Face3D(
            face.nodes_range, face.ownerCells, centre, normal, face.e,
            area, face.delta, face.weight,
        )
    end

    centre_estimates = fill(SVector{3, TF}(0, 0, 0), n_cells)
    n_cell_faces = zeros(_get_int(mesh), n_cells)
    for face in faces
        owner = face.ownerCells[1]
        centre_estimates[owner] += face.centre
        n_cell_faces[owner] += one(eltype(n_cell_faces))
    end
    for fID in (n_bfaces + 1):length(faces)
        face = faces[fID]
        neighbour = face.ownerCells[2]
        centre_estimates[neighbour] += face.centre
        n_cell_faces[neighbour] += one(eltype(n_cell_faces))
    end
    for cID in eachindex(cells)
        centre_estimates[cID] /= TF(n_cell_faces[cID])
    end

    orient_faces && _orient_faces_3d!(mesh, centre_estimates)

    centre_sums = fill(SVector{3, TF}(0, 0, 0), n_cells)
    triple_volumes = zeros(TF, n_cells)
    max_areas = zeros(TF, n_cells)
    for face in faces
        owner = face.ownerCells[1]
        area_vector = face.area*face.normal
        triple_volume = area_vector ⋅ (face.centre - centre_estimates[owner])
        pyramid_centre = TF(3/4)*face.centre + TF(1/4)*centre_estimates[owner]
        centre_sums[owner] += triple_volume*pyramid_centre
        triple_volumes[owner] += triple_volume
        max_areas[owner] = max(max_areas[owner], face.area)
    end
    for fID in (n_bfaces + 1):length(faces)
        face = faces[fID]
        neighbour = face.ownerCells[2]
        area_vector = face.area*face.normal
        triple_volume = area_vector ⋅ (centre_estimates[neighbour] - face.centre)
        pyramid_centre = TF(3/4)*face.centre + TF(1/4)*centre_estimates[neighbour]
        centre_sums[neighbour] += triple_volume*pyramid_centre
        triple_volumes[neighbour] += triple_volume
        max_areas[neighbour] = max(max_areas[neighbour], face.area)
    end

    fixed = 0
    for (cID, cell) in enumerate(cells)
        triple_volume = triple_volumes[cID]
        centre = abs(triple_volume) > floatmin(TF) ?
            centre_sums[cID]/triple_volume : centre_estimates[cID]
        volume = triple_volume/TF(3)
        if !(isfinite(volume) && volume > zero(TF))
            estimate = max_areas[cID]^TF(1.5)*TF(1e-3)
            volume = max(isfinite(volume) ? abs(volume) : zero(TF), estimate)
            fixed += 1
        end
        cells[cID] = Cell(centre, volume, cell.nodes_range, cell.faces_range)
    end
    fixed > 0 && @warn "compute_3d_geometry!: $fixed cell(s) had non-positive volume (degenerate/sliver cells); replaced with positive estimates."

    for (fID, face) in enumerate(faces)
        owner_centre = cells[face.ownerCells[1]].centre
        if fID <= n_bfaces
            weight, delta, direction = weight_delta_e(
                face.centre - owner_centre, face.normal,
            )
        else
            neighbour_centre = cells[face.ownerCells[2]].centre
            weight, delta, direction = weight_delta_e(
                face.centre - owner_centre,
                face.centre - neighbour_centre,
                neighbour_centre - owner_centre,
                face.normal,
            )
        end
        faces[fID] = Face3D(
            face.nodes_range, face.ownerCells, face.centre, face.normal, direction,
            face.area, delta, weight,
        )
    end
    update_face_gDiff!(mesh)
    return mesh
end

function _convert_array!(arr, backend::CPU)
    return arr
end

# Function to prevent redundant CPU copy
function get_boundaries(boundaries::Array)
    return boundaries
end

# Function to copy from GPU to CPU
function get_boundaries(boundaries::AbstractGPUArray)
    # Copy boundaries to CPU
    boundaries_cpu = Array{eltype(boundaries)}(undef, length(boundaries))
    copyto!(boundaries_cpu, boundaries)
    return boundaries_cpu
end

function bounding_box(mesh::AbstractMesh)
    (; faces, face_nodes, nodes) = mesh
    nbfaces = total_boundary_faces(mesh)

    backend = get_backend(faces)
    F = _get_float(mesh)

    pmin = KernelAbstractions.zeros(backend, F, 3)
    pmax = KernelAbstractions.zeros(backend, F, 3)

    ndrange = nbfaces
    workgroup = typeof(backend) <: CPU ? cld(nbfaces, Threads.nthreads()) : 32
    kernel! = _bounding_box(backend, workgroup, ndrange)
    kernel!(pmin, pmax, faces, face_nodes, nodes)
    return pmin, pmax
end

@kernel function _bounding_box(pmin, pmax, faces, face_nodes, nodes)
    fID = @index(Global)
    @inbounds face = faces[fID]
    (; nodes_range) = face

    @inbounds for nID ∈ @view face_nodes[nodes_range]
        node = nodes[nID]
        coords = node.coords
        pmin[1] = min(pmin[1], coords[1])
        pmin[2] = min(pmin[2], coords[2])
        pmin[3] = min(pmin[3], coords[3])
        pmax[1] = max(pmax[1], coords[1])
        pmax[2] = max(pmax[2], coords[2])
        pmax[3] = max(pmax[3], coords[3])
    end
end


# function total_boundary_faces(mesh::Mesh2{I,F}) where {I,F}
function total_boundary_faces(mesh::AbstractMesh)
    nbfaces = zero(_get_int(mesh))
    @inbounds for boundary ∈ get_boundaries(mesh.boundaries)
        nbfaces += length(boundary.IDs_range)
    end
    nbfaces
end

# Extract bundary index based on set name 
struct boundary_info{I<:Integer, S<:Symbol}
    ID::I
    Name::S
end
Adapt.@adapt_structure boundary_info

# Create LUT to map boudnary names to indices
function boundary_map(mesh)
    I = _get_int(mesh); S = Symbol
    boundary_map = boundary_info{I,S}[]

    mesh_temp = adapt(CPU(), mesh) # WARNING: Temp solution 

    for (i, boundary) in enumerate(mesh_temp.boundaries)
        push!(boundary_map, boundary_info{I,S}(i, boundary.name))
    end

    return boundary_map
end

function boundary_index(
    boundaries::Vector{boundary_info{TI, S}}, name::S
    ) where{TI<:Integer,S<:Symbol}
    for index in eachindex(boundaries)
        if boundaries[index].Name == name
            return boundaries[index].ID
        end
    end
end

# Accept boundaries on any backend; Symbols force a host-side search, so copy to CPU first
function boundary_index(boundaries::AbstractArray{<:Boundary}, name::Symbol)
    bs = get_boundaries(boundaries)
    for index ∈ eachindex(bs)
        if bs[index].name == name
            return index
        end
    end
end

# Convert mesh to float type TF, but only after a cheap representability check.
# Falls back to the original mesh (with a warning) if narrowing would be unsafe.
function convert_mesh_float(mesh::AbstractMesh, ::Type{TF}) where {TF<:AbstractFloat}
    _get_float(mesh) === TF && return mesh
    if TF === Float32 && !float32_representable(mesh)
        @warn "Mesh geometry is not reliably representable in Float32; keeping $(_get_float(mesh)) mesh."
        return mesh
    end
    return _rebuild_mesh_float(mesh, TF)
end

function _rebuild_mesh_float(mesh::Mesh3, ::Type{TF}) where {TF<:AbstractFloat}
    nodes = [Node(SVector{3,TF}(n.coords), n.cells_range) for n in mesh.nodes]
    cells = [Cell(SVector{3,TF}(c.centre), TF(c.volume), c.nodes_range, c.faces_range) for c in mesh.cells]
    faces = [Face3D(f.nodes_range, f.ownerCells, SVector{3,TF}(f.centre), SVector{3,TF}(f.normal),
                    SVector{3,TF}(f.e), TF(f.area), TF(f.delta), TF(f.weight)) for f in mesh.faces]
    Mesh3(cells, mesh.cell_nodes, mesh.cell_faces, mesh.cell_neighbours, mesh.cell_nsign,
          faces, mesh.face_nodes, mesh.boundaries, nodes, mesh.node_cells,
          SVector{3,TF}(mesh.get_float), mesh.get_int, mesh.boundary_cellsID)
end

function _rebuild_mesh_float(mesh::Mesh2, ::Type{TF}) where {TF<:AbstractFloat}
    nodes = [Node(SVector{3,TF}(n.coords), n.cells_range) for n in mesh.nodes]
    cells = [Cell(SVector{3,TF}(c.centre), TF(c.volume), c.nodes_range, c.faces_range) for c in mesh.cells]
    faces = [Face2D(f.nodes_range, f.ownerCells, SVector{3,TF}(f.centre), SVector{3,TF}(f.normal),
                    SVector{3,TF}(f.e), TF(f.area), TF(f.delta), TF(f.weight)) for f in mesh.faces]
    Mesh2(cells, mesh.cell_nodes, mesh.cell_faces, mesh.cell_neighbours, mesh.cell_nsign,
          faces, mesh.face_nodes, mesh.boundaries, nodes, mesh.node_cells,
          SVector{3,TF}(mesh.get_float), mesh.get_int, mesh.boundary_cellsID)
end

# Cheap check: volumes/areas/deltas/weights finite & positive and length scales above Float32 spacing.
function float32_representable(mesh::AbstractMesh)
    _count_invalid_positive(c.volume for c in mesh.cells) == 0 || return false
    _count_invalid_positive(f.area for f in mesh.faces) == 0 || return false
    _count_invalid_positive(f.delta for f in mesh.faces) == 0 || return false
    count(!isfinite, mesh.face_gDiff) == 0 || return false
    count(f -> !isfinite(f.weight), mesh.faces) == 0 || return false

    max_coord = 0.0
    for node in mesh.nodes, coord in node.coords
        max_coord = max(max_coord, abs(Float64(coord)))
    end
    min_delta = Inf
    for face in mesh.faces
        delta = Float64(face.delta)
        isfinite(delta) && delta > 0 && (min_delta = min(min_delta, delta))
    end
    spacing = Float64(eps(Float32)) * max(max_coord, 1.0)
    return min_delta > 16 * spacing
end

# Used by the loaders/tests to reject a Float32 mesh that is not safely representable.
function validate_single_precision_mesh(mesh::AbstractMesh; source="mesh conversion")
    _get_float(mesh) === Float32 || return mesh
    float32_representable(mesh) && return mesh
    throw(ArgumentError("Single-precision mesh validation failed during $source: " *
        "the Float32 mesh geometry is not reliable (non-positive volumes/areas/deltas, " *
        "non-finite weights, or length scales below Float32 spacing). Use float_type=Float64."))
end

function _count_invalid_positive(values)
    count(v -> !isfinite(v) || v <= zero(v), values)
end

# Static normalise function
function norm_static(arr, p = 2)
    sum = 0
    for i in eachindex(arr)
        val = (abs(arr[i]))^p
        sum += val
    end
    return sum^(1/p)
end
