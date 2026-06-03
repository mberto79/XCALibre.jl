export get_boundaries
export _get_float, _get_int, _get_backend, _convert_array!
export bounding_box
export boundary_info, boundary_map
export total_boundary_faces, boundary_index
export norm_static
export convert_mesh_float
export validate_single_precision_mesh
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

function boundary_index(boundaries::Vector{Boundary{S, UR}}, name::S) where {S<:Symbol,UR}
    # bci = zero(TI)
    for index ∈ eachindex(boundaries)
        # bci += 1
        if boundaries[index].name == name
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
