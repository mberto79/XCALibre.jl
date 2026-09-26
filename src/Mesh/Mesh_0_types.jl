export Node, Boundary, Cell
export Face2D, Face3D
export Mesh2, Mesh3
export AbstractMesh
export AbstractWriter, VTK, OpenFOAM

abstract type AbstractMesh end

"""
    struct Node{SV3<:SVector{3,<:AbstractFloat}, UR<:UnitRange{<:Integer}}
        coords::SV3     # node coordinates
        cells_range::UR # range to access neighbour cells in Mesh3.node_cells
    end
"""
struct Node{SV3<:SVector{3,<:AbstractFloat}, UR<:UnitRange{<:Integer}}
    coords::SV3     # node coordinates
    cells_range::UR # range to access neighbour cells in Mesh3.node_cells
end
Adapt.@adapt_structure Node


"""
    struct Boundary{S<:Symbol, UR<:UnitRange{<:Integer}}
        name::S         # Boundary patch name
        IDs_range::UR   # range to access boundary info (faces and boundary_cellsID)
    end
"""
struct Boundary{S<:Symbol, UR<:UnitRange{<:Integer}}
    name::S         # Boundary patch name
    IDs_range::UR   # range to access boundary info (faces and boundary_cellsID)
end
Adapt.@adapt_structure Boundary


"""
    struct Cell{F<:AbstractFloat, SV3<:SVector{3,F},UR<:UnitRange{<:Integer}}
        centre::SV3     # coordinate of cell centroid
        volume::F       # cell volume
        nodes_range::UR # range to access cell nodes in Mesh3.cell_nodes
        faces_range::UR # range to access cell faces info (faces, neighbours cells, etc.)
    end
"""
struct Cell{F<:AbstractFloat, SV3<:SVector{3,F},UR<:UnitRange{<:Integer}}
    centre::SV3     # coordinate of cell centroid
    volume::F       # cell volume
    nodes_range::UR # range to access cell nodes in Mesh3.cell_nodes
    faces_range::UR # range to access cell faces info (faces, neighbours cells, etc.)
end
Adapt.@adapt_structure Cell

# dispatch on ::Type{} so TI and TF may differ and the result type is inferable
Cell(::Type{TI}, ::Type{TF}) where {TI<:Integer, TF<:AbstractFloat} = begin
    Cell(
        SVector{3,TF}(0.0,0.0,0.0),
        zero(TF),
        UnitRange{TI}(0,0),
        UnitRange{TI}(0,0)
        )
end


# 2D and 3D Face types

struct Face2D{
    F<:AbstractFloat, 
    SV2<:SVector{2,<:Integer},
    SV3<:SVector{3,F}, 
    UR<:UnitRange{<:Integer}
    }

    nodes_range::UR
    ownerCells::SV2
    centre::SV3
    normal::SV3
    e::SV3
    area::F
    delta::F
    weight::F
end
Adapt.@adapt_structure Face2D


"""
    struct Face3D{
        F<:AbstractFloat, 
        SV2<:SVector{2,<:Integer},
        SV3<:SVector{3,F}, 
        UR<:UnitRange{<:Integer}
        }
        
        nodes_range::UR # range to access face nodes in Mesh3.face_nodes
        ownerCells::SV2 # IDs of face owner cells (always 2)
        centre::SV3     # coordinates of face centre
        normal::SV3     # face normal unit vector
        e::SV3          # unit vector in the direction between owner cells
        area::F         # face area
        delta::F        # distance between owner cells centres
        weight::F       # linear interpolation weight
    end
"""
struct Face3D{
    F<:AbstractFloat, 
    SV2<:SVector{2,<:Integer},
    SV3<:SVector{3,F}, 
    UR<:UnitRange{<:Integer}
    }
    
    nodes_range::UR # range to access face nodes in Mesh3.face_nodes
    ownerCells::SV2 # IDs of face owner cells (always 2)
    centre::SV3     # coordinates of face centre
    normal::SV3     # face normal unit vector
    e::SV3          # unit vector in the direction between owner cells
    area::F         # face area
    delta::F        # distance between owner cells centres
    weight::F       # linear interpolation weight
end
Adapt.@adapt_structure Face3D

# dispatch on ::Type{} so TI and TF may differ and the result type is inferable
Face3D(::Type{TI}, ::Type{TF}) where {TI<:Integer, TF<:AbstractFloat} = begin
    Face3D(
        UnitRange{TI}(0,0),
        SVector{2,TI}(0,0),
        SVector{3,TF}(0.0,0.0,0.0),
        SVector{3,TF}(0.0,0.0,0.0),
        SVector{3,TF}(0.0,0.0,0.0),
        zero(TF),
        zero(TF),
        zero(TF)
    )
end

# 2D and 3D Mesh types

# NEW SECTION: element arrays stored per field so kernels move only the columns they read
# Same-typed columns share one type parameter to keep mesh-carrying types small for inference.

struct FaceArrays{T<:Union{Face2D,Face3D}, VR, VO, VV, VF} <: AbstractVector{T}
    nodes_range::VR
    ownerCells::VO
    centre::VV
    normal::VV
    e::VV
    area::VF
    delta::VF
    weight::VF
end

struct CellArrays{T<:Cell, VV, VF, VR} <: AbstractVector{T}
    centre::VV
    volume::VF
    nodes_range::VR
    faces_range::VR
end

struct NodeArrays{T<:Node, VV, VR} <: AbstractVector{T}
    coords::VV
    cells_range::VR
end

const ElementArrays = Union{FaceArrays, CellArrays, NodeArrays}

# element type recovered from the columns, so adapt may change their storage freely
FaceArrays{D}(nr::VR, oc::VO, c::VV, n::VV, e::VV, a::VF, d::VF, w::VF) where {D, VR, VO, VV, VF} =
    FaceArrays{D{eltype(a), eltype(oc), eltype(c), eltype(nr)}, VR, VO, VV, VF}(nr, oc, c, n, e, a, d, w)
CellArrays(c::VV, v::VF, nr::VR, fr::VR) where {VV, VF, VR} =
    CellArrays{Cell{eltype(v), eltype(c), eltype(nr)}, VV, VF, VR}(c, v, nr, fr)
NodeArrays(c::VV, r::VR) where {VV, VR} = NodeArrays{Node{eltype(c), eltype(r)}, VV, VR}(c, r)

# a storage change keeps the element type, so it is carried over rather than recomputed
_rewrap(::FaceArrays{T}, c) where T = FaceArrays{T, typeof(c[1]), typeof(c[2]), typeof(c[3]), typeof(c[6])}(c...)
_rewrap(::CellArrays{T}, c) where T = CellArrays{T, typeof(c[1]), typeof(c[2]), typeof(c[3])}(c...)
_rewrap(::NodeArrays{T}, c) where T = NodeArrays{T, typeof(c[1]), typeof(c[2])}(c...)

_columns(x::FaceArrays) = (x.nodes_range, x.ownerCells, x.centre, x.normal, x.e, x.area, x.delta, x.weight)
_columns(x::CellArrays) = (x.centre, x.volume, x.nodes_range, x.faces_range)
_columns(x::NodeArrays) = (x.coords, x.cells_range)
_columns(v::AbstractVector{T}) where T = ntuple(i -> map(e -> getfield(e, i), v), Val(fieldcount(T)))

_soa(x::ElementArrays) = x
_soa(x::AbstractVector{<:Face2D}) = FaceArrays{Face2D}(_columns(x)...)
_soa(x::AbstractVector{<:Face3D}) = FaceArrays{Face3D}(_columns(x)...)
_soa(x::AbstractVector{<:Cell}) = CellArrays(_columns(x)...)
_soa(x::AbstractVector{<:Node}) = NodeArrays(_columns(x)...)

Base.size(x::ElementArrays) = size(getfield(x, 1))
Base.IndexStyle(::Type{<:ElementArrays}) = IndexLinear()

# one bounds check per element, as for a plain vector; columns share its length
@inline function Base.getindex(x::FaceArrays{T}, i::Int) where T
    @boundscheck checkbounds(x.area, i)
    @inbounds T(x.nodes_range[i], x.ownerCells[i], x.centre[i], x.normal[i], x.e[i], x.area[i], x.delta[i], x.weight[i])
end
@inline function Base.getindex(x::CellArrays{T}, i::Int) where T
    @boundscheck checkbounds(x.volume, i)
    @inbounds T(x.centre[i], x.volume[i], x.nodes_range[i], x.faces_range[i])
end
@inline function Base.getindex(x::NodeArrays{T}, i::Int) where T
    @boundscheck checkbounds(x.coords, i)
    @inbounds T(x.coords[i], x.cells_range[i])
end

Base.@propagate_inbounds function Base.setindex!(x::FaceArrays{T}, v, i::Int) where T
    f = convert(T, v)
    x.nodes_range[i] = f.nodes_range; x.ownerCells[i] = f.ownerCells
    x.centre[i] = f.centre; x.normal[i] = f.normal; x.e[i] = f.e
    x.area[i] = f.area; x.delta[i] = f.delta; x.weight[i] = f.weight
    x
end
Base.@propagate_inbounds function Base.setindex!(x::CellArrays{T}, v, i::Int) where T
    c = convert(T, v)
    x.centre[i] = c.centre; x.volume[i] = c.volume
    x.nodes_range[i] = c.nodes_range; x.faces_range[i] = c.faces_range
    x
end
Base.@propagate_inbounds function Base.setindex!(x::NodeArrays{T}, v, i::Int) where T
    n = convert(T, v)
    x.coords[i] = n.coords; x.cells_range[i] = n.cells_range
    x
end

Base.similar(x::ElementArrays, ::Type{T}, dims::Tuple{Int}) where T = T === eltype(x) ?
    _rewrap(x, map(c -> similar(c, dims), _columns(x))) : similar(getfield(x, 1), T, dims)
Base.copy(x::ElementArrays) = _rewrap(x, map(copy, _columns(x)))
KernelAbstractions.get_backend(x::ElementArrays) = get_backend(getfield(x, 1))

# no closure over `to`: a captured type is stored as its kind and adapt stops inferring
Adapt.adapt_structure(to, x::FaceArrays) = _rewrap(x, (adapt(to, x.nodes_range), adapt(to, x.ownerCells),
    adapt(to, x.centre), adapt(to, x.normal), adapt(to, x.e), adapt(to, x.area), adapt(to, x.delta), adapt(to, x.weight)))
Adapt.adapt_structure(to, x::CellArrays) = _rewrap(x, (adapt(to, x.centre), adapt(to, x.volume),
    adapt(to, x.nodes_range), adapt(to, x.faces_range)))
Adapt.adapt_structure(to, x::NodeArrays) = _rewrap(x, (adapt(to, x.coords), adapt(to, x.cells_range)))

# normal signs are only ±1, so one byte each
_nsign(x::AbstractArray{Int8}) = x
_nsign(x::AbstractArray) = Int8.(x)

# float/int type tags are one-element arrays stored like `like`, so they share its type parameter
_type_tag(like::A, x::A) where A = x
_type_tag(like, x) = fill!(similar(like, eltype(x), 1), zero(eltype(x)))

struct Mesh2{VV, VTF, VR, VO, VI, VS, VB} <: AbstractMesh
    cell_centre::VV      # cell centroid coordinates
    cell_volume::VTF     # cell volumes
    cell_nodes_range::VR # range of each cell's nodes in cell_nodes
    cell_faces_range::VR # range of each cell's faces in cell_faces, cell_neighbours, cell_nsign
    cell_nodes::VI       # node IDs of each cell
    cell_faces::VI       # internal face IDs of each cell
    cell_neighbours::VI  # neighbour cell IDs across each cell face
    cell_nsign::VS       # face normal sign per cell face (1 or -1)
    face_nodes_range::VR # range of each face's nodes in face_nodes
    face_ownerCells::VO  # owner cell ID pair of each face (equal on boundary faces)
    face_centre::VV      # face centre coordinates
    face_normal::VV      # face unit normals
    face_e::VV           # unit vectors between owner cell centres
    face_area::VTF       # face areas
    face_delta::VTF      # distance between owner cell centres
    face_weight::VTF     # linear interpolation weights
    face_nodes::VI       # node IDs of each face
    face_gDiff::VTF      # Laplacian face coefficient (derived, see `_gDiff`)
    boundaries::VB       # boundary patches
    node_coords::VV      # node coordinates
    node_cells_range::VR # range of each node's cells in node_cells
    node_cells::VI       # cell IDs around each node
    get_float::VTF       # one-element array tagging the mesh float type
    get_int::VI          # one-element array tagging the mesh integer type
    boundary_cellsID::VI # owner cell ID of each boundary face
    function Mesh2(cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
        face_gDiff, boundaries, nodes, node_cells, get_float, get_int, boundary_cellsID)
        c, f, n, s = _soa(cells), _soa(faces), _soa(nodes), _nsign(cell_nsign)
        fields = (c.centre, c.volume, c.nodes_range, c.faces_range, cell_nodes, cell_faces,
            cell_neighbours, s, f.nodes_range, f.ownerCells, f.centre, f.normal, f.e, f.area, f.delta,
            f.weight, face_nodes, face_gDiff, boundaries, n.coords, n.cells_range, node_cells,
            _type_tag(f.area, get_float), _type_tag(cell_nodes, get_int), boundary_cellsID)
        new{typeof(f.centre), typeof(f.area), typeof(f.nodes_range), typeof(f.ownerCells),
            typeof(cell_nodes), typeof(s), typeof(boundaries)}(fields...)
    end
end

@inline Base.getproperty(m::Mesh2, s::Symbol) =
    s === :faces ? FaceArrays{Face2D}(getfield(m, :face_nodes_range), getfield(m, :face_ownerCells),
        getfield(m, :face_centre), getfield(m, :face_normal), getfield(m, :face_e),
        getfield(m, :face_area), getfield(m, :face_delta), getfield(m, :face_weight)) :
    s === :cells ? CellArrays(getfield(m, :cell_centre), getfield(m, :cell_volume),
        getfield(m, :cell_nodes_range), getfield(m, :cell_faces_range)) :
    s === :nodes ? NodeArrays(getfield(m, :node_coords), getfield(m, :node_cells_range)) :
    getfield(m, s)

Adapt.adapt_structure(to, m::Mesh2) = Mesh2(adapt(to, m.cells), adapt(to, m.cell_nodes),
    adapt(to, m.cell_faces), adapt(to, m.cell_neighbours), adapt(to, m.cell_nsign), adapt(to, m.faces),
    adapt(to, m.face_nodes), adapt(to, m.face_gDiff), adapt(to, m.boundaries), adapt(to, m.nodes),
    adapt(to, m.node_cells), adapt(to, m.get_float), adapt(to, m.get_int), adapt(to, m.boundary_cellsID))

Mesh2(cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
      boundaries, nodes, node_cells, get_float, get_int, boundary_cellsID) = Mesh2(
    cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
    face_gDiff_coefficients(faces), boundaries, nodes, node_cells, get_float, get_int,
    boundary_cellsID)

"""
    struct Mesh3{VV, VTF, VR, VO, VI, VS, VB} <: AbstractMesh

3D unstructured mesh. `Mesh2` has identical fields. Each cell, face and node property is stored as
its own array; `mesh.cells`, `mesh.faces` and `mesh.nodes` return views that index as `Cell`,
`Face3D` and `Node` elements. Boundary faces come first in the face arrays.

```julia
    cell_centre::VV      # cell centroid coordinates
    cell_volume::VTF     # cell volumes
    cell_nodes_range::VR # range of each cell's nodes in cell_nodes
    cell_faces_range::VR # range of each cell's faces in cell_faces, cell_neighbours, cell_nsign
    cell_nodes::VI       # node IDs of each cell
    cell_faces::VI       # internal face IDs of each cell
    cell_neighbours::VI  # neighbour cell IDs across each cell face
    cell_nsign::VS       # face normal sign per cell face (1 or -1)
    face_nodes_range::VR # range of each face's nodes in face_nodes
    face_ownerCells::VO  # owner cell ID pair of each face (equal on boundary faces)
    face_centre::VV      # face centre coordinates
    face_normal::VV      # face unit normals
    face_e::VV           # unit vectors between owner cell centres
    face_area::VTF       # face areas
    face_delta::VTF      # distance between owner cell centres
    face_weight::VTF     # linear interpolation weights
    face_nodes::VI       # node IDs of each face
    face_gDiff::VTF      # Laplacian face coefficient (derived, see `_gDiff`)
    boundaries::VB       # boundary patches
    node_coords::VV      # node coordinates
    node_cells_range::VR # range of each node's cells in node_cells
    node_cells::VI       # cell IDs around each node
    get_float::VTF       # one-element array tagging the mesh float type
    get_int::VI          # one-element array tagging the mesh integer type
    boundary_cellsID::VI # owner cell ID of each boundary face
```
"""
struct Mesh3{VV, VTF, VR, VO, VI, VS, VB} <: AbstractMesh
    cell_centre::VV      # cell centroid coordinates
    cell_volume::VTF     # cell volumes
    cell_nodes_range::VR # range of each cell's nodes in cell_nodes
    cell_faces_range::VR # range of each cell's faces in cell_faces, cell_neighbours, cell_nsign
    cell_nodes::VI       # node IDs of each cell
    cell_faces::VI       # internal face IDs of each cell
    cell_neighbours::VI  # neighbour cell IDs across each cell face
    cell_nsign::VS       # face normal sign per cell face (1 or -1)
    face_nodes_range::VR # range of each face's nodes in face_nodes
    face_ownerCells::VO  # owner cell ID pair of each face (equal on boundary faces)
    face_centre::VV      # face centre coordinates
    face_normal::VV      # face unit normals
    face_e::VV           # unit vectors between owner cell centres
    face_area::VTF       # face areas
    face_delta::VTF      # distance between owner cell centres
    face_weight::VTF     # linear interpolation weights
    face_nodes::VI       # node IDs of each face
    face_gDiff::VTF      # Laplacian face coefficient (derived, see `_gDiff`)
    boundaries::VB       # boundary patches
    node_coords::VV      # node coordinates
    node_cells_range::VR # range of each node's cells in node_cells
    node_cells::VI       # cell IDs around each node
    get_float::VTF       # one-element array tagging the mesh float type
    get_int::VI          # one-element array tagging the mesh integer type
    boundary_cellsID::VI # owner cell ID of each boundary face
    function Mesh3(cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
        face_gDiff, boundaries, nodes, node_cells, get_float, get_int, boundary_cellsID)
        c, f, n, s = _soa(cells), _soa(faces), _soa(nodes), _nsign(cell_nsign)
        fields = (c.centre, c.volume, c.nodes_range, c.faces_range, cell_nodes, cell_faces,
            cell_neighbours, s, f.nodes_range, f.ownerCells, f.centre, f.normal, f.e, f.area, f.delta,
            f.weight, face_nodes, face_gDiff, boundaries, n.coords, n.cells_range, node_cells,
            _type_tag(f.area, get_float), _type_tag(cell_nodes, get_int), boundary_cellsID)
        new{typeof(f.centre), typeof(f.area), typeof(f.nodes_range), typeof(f.ownerCells),
            typeof(cell_nodes), typeof(s), typeof(boundaries)}(fields...)
    end
end

@inline Base.getproperty(m::Mesh3, s::Symbol) =
    s === :faces ? FaceArrays{Face3D}(getfield(m, :face_nodes_range), getfield(m, :face_ownerCells),
        getfield(m, :face_centre), getfield(m, :face_normal), getfield(m, :face_e),
        getfield(m, :face_area), getfield(m, :face_delta), getfield(m, :face_weight)) :
    s === :cells ? CellArrays(getfield(m, :cell_centre), getfield(m, :cell_volume),
        getfield(m, :cell_nodes_range), getfield(m, :cell_faces_range)) :
    s === :nodes ? NodeArrays(getfield(m, :node_coords), getfield(m, :node_cells_range)) :
    getfield(m, s)

Base.propertynames(m::Union{Mesh2,Mesh3}, private::Bool=false) = (:cells, :faces, :nodes, fieldnames(typeof(m))...)

Adapt.adapt_structure(to, m::Mesh3) = Mesh3(adapt(to, m.cells), adapt(to, m.cell_nodes),
    adapt(to, m.cell_faces), adapt(to, m.cell_neighbours), adapt(to, m.cell_nsign), adapt(to, m.faces),
    adapt(to, m.face_nodes), adapt(to, m.face_gDiff), adapt(to, m.boundaries), adapt(to, m.nodes),
    adapt(to, m.node_cells), adapt(to, m.get_float), adapt(to, m.get_int), adapt(to, m.boundary_cellsID))

Mesh3(cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
      boundaries, nodes, node_cells, get_float, get_int, boundary_cellsID) = Mesh3(
    cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
    face_gDiff_coefficients(faces), boundaries, nodes, node_cells, get_float, get_int,
    boundary_cellsID)

# first touch: arrays reached through a per-cell, face or node range are cut by that range, so
# each thread places the entries of the elements it owns
Adapt.adapt_structure(to::FirstTouch, m::Mesh2) = _first_touch_mesh(Mesh2, to, m)
Adapt.adapt_structure(to::FirstTouch, m::Mesh3) = _first_touch_mesh(Mesh3, to, m)
_first_touch_mesh(M, to, m) = M(adapt(to, m.cells), first_touch_copy(m.cell_nodes, m.cell_nodes_range),
    first_touch_copy(m.cell_faces, m.cell_faces_range), first_touch_copy(m.cell_neighbours, m.cell_faces_range),
    first_touch_copy(m.cell_nsign, m.cell_faces_range), adapt(to, m.faces),
    first_touch_copy(m.face_nodes, m.face_nodes_range), adapt(to, m.face_gDiff), adapt(to, m.boundaries),
    adapt(to, m.nodes), first_touch_copy(m.node_cells, m.node_cells_range), adapt(to, m.get_float),
    adapt(to, m.get_int), adapt(to, m.boundary_cellsID))

Base.show(io::IO, mesh::AbstractMesh) = begin
    if typeof(mesh) <: Mesh2
        meshType = "2D"
    elseif typeof(mesh) <: Mesh3
        meshType = "3D"
    end

    output = 
"""
$meshType Mesh with:
-> $(length(mesh.cells)) cells
-> $(length(mesh.faces)) faces
-> $(length(mesh.nodes)) nodes"""
    print(io, output)
end

# Mesh write output
abstract type AbstractWriter end
struct VTK <: AbstractWriter end
struct OpenFOAM <: AbstractWriter end