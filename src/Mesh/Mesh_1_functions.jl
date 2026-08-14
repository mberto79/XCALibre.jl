export get_boundaries
export _get_float, _get_int, _get_backend, _convert_array!
export bounding_box
export boundary_info, boundary_map
export total_boundary_faces, boundary_index
export norm_static
export convert_mesh_float
export validate_single_precision_mesh
export compute_3d_geometry!, face_geometry
# export x, y, z # access cell centres
# export xf, yf, zf # access face centres

_get_int(mesh) = eltype(mesh.get_int)
_get_float(mesh) = eltype(mesh.get_float)
_get_backend(mesh) = get_backend(mesh.cells)

# function to calculate internal face properties
# C1F1 = distance vector from cell1 centre to face centre
# C2F1 = distance vector from cell2 centre to face centre
# C1C2 = distance vector from cell1 to cell2
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

function face_geometry(nodes, node_ids, apex::SVector{3, TF}) where {TF<:AbstractFloat}
    area_vector = SVector{3, TF}(0, 0, 0)
    n_nodes = length(node_ids)
    @inbounds for index in 1:n_nodes
        next_index = index == n_nodes ? 1 : index + 1
        point = nodes[node_ids[index]].coords
        next_point = nodes[node_ids[next_index]].coords
        area_vector += ((point - apex) × (next_point - apex))/TF(2)
    end

    area = norm(area_vector)
    normal = area > zero(TF) ? area_vector/area : SVector{3, TF}(0, 0, 0)
    centre_sum = SVector{3, TF}(0, 0, 0)
    projected_area = zero(TF)
    @inbounds for index in 1:n_nodes
        next_index = index == n_nodes ? 1 : index + 1
        point = nodes[node_ids[index]].coords
        next_point = nodes[node_ids[next_index]].coords
        triangle_vector = ((point - apex) × (next_point - apex))/TF(2)
        weight = triangle_vector ⋅ normal
        projected_area += weight
        centre_sum += weight*(apex + point + next_point)/TF(3)
    end
    centre = projected_area > floatmin(TF) ? centre_sum/projected_area : apex
    return normal, area, centre
end

function compute_3d_geometry!(mesh::Mesh3)
    (; cells, faces, face_nodes, nodes, boundary_cellsID) = mesh
    TF = _get_float(mesh)
    n_cells = length(cells)
    n_boundary_faces = length(boundary_cellsID)

    for (face_id, face) in enumerate(faces)
        node_ids = @view face_nodes[face.nodes_range]
        apex = sum(nodes[node_id].coords for node_id in node_ids)/TF(length(node_ids))
        normal, area, centre = face_geometry(nodes, node_ids, apex)
        faces[face_id] = Face3D(
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
    for face_id in (n_boundary_faces + 1):length(faces)
        face = faces[face_id]
        neighbour = face.ownerCells[2]
        centre_estimates[neighbour] += face.centre
        n_cell_faces[neighbour] += one(eltype(n_cell_faces))
    end
    for cell_id in eachindex(cells)
        centre_estimates[cell_id] /= TF(n_cell_faces[cell_id])
    end

    for (face_id, face) in enumerate(faces)
        owner = face.ownerCells[1]
        direction = face_id <= n_boundary_faces ?
            face.centre - centre_estimates[owner] :
            centre_estimates[face.ownerCells[2]] - centre_estimates[owner]
        direction ⋅ face.normal >= zero(TF) && continue
        reverse!(@view face_nodes[face.nodes_range])
        faces[face_id] = Face3D(
            face.nodes_range, face.ownerCells, face.centre, -face.normal, face.e,
            face.area, face.delta, face.weight,
        )
    end

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
    for face_id in (n_boundary_faces + 1):length(faces)
        face = faces[face_id]
        neighbour = face.ownerCells[2]
        area_vector = face.area*face.normal
        triple_volume = area_vector ⋅ (centre_estimates[neighbour] - face.centre)
        pyramid_centre = TF(3/4)*face.centre + TF(1/4)*centre_estimates[neighbour]
        centre_sums[neighbour] += triple_volume*pyramid_centre
        triple_volumes[neighbour] += triple_volume
        max_areas[neighbour] = max(max_areas[neighbour], face.area)
    end

    fixed = 0
    for (cell_id, cell) in enumerate(cells)
        triple_volume = triple_volumes[cell_id]
        centre = abs(triple_volume) > floatmin(TF) ?
            centre_sums[cell_id]/triple_volume : centre_estimates[cell_id]
        volume = triple_volume/TF(3)
        if !(isfinite(volume) && volume > zero(TF))
            estimate = max_areas[cell_id]^TF(1.5)*TF(1e-3)
            volume = max(isfinite(volume) ? abs(volume) : zero(TF), estimate)
            fixed += 1
        end
        cells[cell_id] = Cell(centre, volume, cell.nodes_range, cell.faces_range)
    end
    fixed > 0 && @warn "compute_3d_geometry!: $fixed cell(s) had non-positive volume (degenerate/sliver cells); replaced with positive estimates."

    for (face_id, face) in enumerate(faces)
        owner_centre = cells[face.ownerCells[1]].centre
        if face_id <= n_boundary_faces
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
        faces[face_id] = Face3D(
            face.nodes_range, face.ownerCells, face.centre, face.normal, direction,
            face.area, delta, weight,
        )
    end
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

# function x(mesh::Mesh2{I,F}) where {I,F}
#     cells = mesh.cells
#     out = zeros(F, length(cells))
#     @inbounds for i ∈ eachindex(cells)
#         out[i] = cells[i].centre[1]
#     end
#     return out
# end

# function y(mesh::Mesh2{I,F}) where {I,F}
#     cells = mesh.cells
#     out = zeros(F, length(cells))
#     @inbounds for i ∈ eachindex(cells)
#         out[i] = cells[i].centre[2]
#     end
#     return out
# end

# function z(mesh::Mesh2{I,F}) where {I,F}
#     cells = mesh.cells
#     out = zeros(F, length(cells))
#     @inbounds for i ∈ eachindex(cells)
#         out[i] = cells[i].centre[3]
#     end
#     return out
# end

# function xf(mesh::Mesh2{I,F}) where {I,F}
#     faces = mesh.faces
#     out = zeros(F, length(faces))
#     @inbounds for i ∈ eachindex(faces)
#         out[i] = faces[i].centre[1]
#     end
#     return out
# end

# function yf(mesh::Mesh2{I,F}) where {I,F}
#     faces = mesh.faces
#     out = zeros(F, length(faces))
#     @inbounds for i ∈ eachindex(faces)
#         out[i] = faces[i].centre[2]
#     end
#     return out
# end

# function zf(mesh::Mesh2{I,F}) where {I,F}
#     faces = mesh.faces
#     out = zeros(F, length(faces))
#     @inbounds for i ∈ eachindex(faces)
#         out[i] = faces[i].centre[3]
#     end
#     return out
# end

# Static normalise function
function norm_static(arr, p = 2)
    sum = 0
    for i in eachindex(arr)
        val = (abs(arr[i]))^p
        sum += val
    end
    return sum^(1/p)
end
