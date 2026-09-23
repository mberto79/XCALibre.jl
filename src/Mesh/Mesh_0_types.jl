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

_rewrap(::FaceArrays{<:Face2D}, cols) = FaceArrays{Face2D}(cols...)
_rewrap(::FaceArrays{<:Face3D}, cols) = FaceArrays{Face3D}(cols...)
_rewrap(::CellArrays, cols) = CellArrays(cols...)
_rewrap(::NodeArrays, cols) = NodeArrays(cols...)

_columns(x::ElementArrays) = ntuple(i -> getfield(x, i), Val(fieldcount(typeof(x))))
_columns(v::AbstractVector{T}) where T = ntuple(i -> map(e -> getfield(e, i), v), Val(fieldcount(T)))

_soa(x::ElementArrays) = x
_soa(x::AbstractVector{<:Face2D}) = FaceArrays{Face2D}(_columns(x)...)
_soa(x::AbstractVector{<:Face3D}) = FaceArrays{Face3D}(_columns(x)...)
_soa(x::AbstractVector{<:Cell}) = CellArrays(_columns(x)...)
_soa(x::AbstractVector{<:Node}) = NodeArrays(_columns(x)...)

Base.size(x::ElementArrays) = size(getfield(x, 1))
Base.IndexStyle(::Type{<:ElementArrays}) = IndexLinear()

Base.@propagate_inbounds Base.getindex(x::FaceArrays{T}, i::Int) where T = T(x.nodes_range[i],
    x.ownerCells[i], x.centre[i], x.normal[i], x.e[i], x.area[i], x.delta[i], x.weight[i])
Base.@propagate_inbounds Base.getindex(x::CellArrays{T}, i::Int) where T =
    T(x.centre[i], x.volume[i], x.nodes_range[i], x.faces_range[i])
Base.@propagate_inbounds Base.getindex(x::NodeArrays{T}, i::Int) where T = T(x.coords[i], x.cells_range[i])

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

Adapt.adapt_structure(to, x::ElementArrays) = _rewrap(x, map(c -> adapt(to, c), _columns(x)))

# normal signs are only ±1, so one byte each
_nsign(x::AbstractArray{Int8}) = x
_nsign(x::AbstractArray) = Int8.(x)

struct Mesh2{VC, VI, VS, VF<:AbstractArray{<:Face2D}, VTF, VB, VN, SV3, UR} <: AbstractMesh
    cells::VC
    cell_nodes::VI
    cell_faces::VI
    cell_neighbours::VI
    cell_nsign::VS
    faces::VF
    face_nodes::VI
    face_gDiff::VTF
    boundaries::VB
    nodes::VN
    node_cells::VI # can be empty for now
    get_float::SV3
    get_int::UR
    boundary_cellsID::VI
    function Mesh2(cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
        face_gDiff, boundaries, nodes, node_cells, get_float, get_int, boundary_cellsID)
        c, f, n, s = _soa(cells), _soa(faces), _soa(nodes), _nsign(cell_nsign)
        new{typeof(c), typeof(cell_nodes), typeof(s), typeof(f), typeof(face_gDiff), typeof(boundaries),
            typeof(n), typeof(get_float), typeof(get_int)}(c, cell_nodes, cell_faces, cell_neighbours,
            s, f, face_nodes, face_gDiff, boundaries, n, node_cells, get_float, get_int,
            boundary_cellsID)
    end
end
Adapt.@adapt_structure Mesh2

# face_gDiff is derived here rather than passed in, so no call site can supply a stale array.
# Readers that fill the face geometry after construction must call update_face_gDiff!
Mesh2(cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
      boundaries, nodes, node_cells, get_float, get_int, boundary_cellsID) = Mesh2(
    cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
    face_gDiff_coefficients(faces), boundaries, nodes, node_cells, get_float, get_int,
    boundary_cellsID)


"""
    struct Mesh3{VC, VI, VS, VF<:AbstractArray{<:Face3D}, VTF, VB, VN, SV3, UR} <: AbstractMesh
        cells::VC           # vector of cells
        cell_nodes::VI      # vector of indices to access cell nodes
        cell_faces::VI      # vector of indices to access cell faces
        cell_neighbours::VI # vector of indices to access cell neighbours
        cell_nsign::VS      # face normal sign per cell face (Int8, 1 or -1)
        faces::VF           # vector of faces
        face_nodes::VI      # vector of indices to access face nodes
        face_gDiff::VTF     # Laplacian face coefficient (derived, see `_gDiff`)
        boundaries::VB      # vector of boundaries
        nodes::VN           # vector of nodes
        node_cells::VI      # vector of indices to access node cells
        get_float::SV3      # store mesh float type
        get_int::UR         # store mesh integer type
        boundary_cellsID::VI # vector of indices of boundary cell IDs
    end
"""
struct Mesh3{VC, VI, VS, VF<:AbstractArray{<:Face3D}, VTF, VB, VN, SV3, UR} <: AbstractMesh
    cells::VC
    cell_nodes::VI
    cell_faces::VI
    cell_neighbours::VI
    cell_nsign::VS
    faces::VF
    face_nodes::VI
    face_gDiff::VTF
    boundaries::VB
    nodes::VN
    node_cells::VI # can be empty for now
    get_float::SV3
    get_int::UR
    boundary_cellsID::VI
    function Mesh3(cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
        face_gDiff, boundaries, nodes, node_cells, get_float, get_int, boundary_cellsID)
        c, f, n, s = _soa(cells), _soa(faces), _soa(nodes), _nsign(cell_nsign)
        new{typeof(c), typeof(cell_nodes), typeof(s), typeof(f), typeof(face_gDiff), typeof(boundaries),
            typeof(n), typeof(get_float), typeof(get_int)}(c, cell_nodes, cell_faces, cell_neighbours,
            s, f, face_nodes, face_gDiff, boundaries, n, node_cells, get_float, get_int,
            boundary_cellsID)
    end
end
Adapt.@adapt_structure Mesh3

# face_gDiff is derived here rather than passed in, so no call site can supply a stale array.
# Readers that fill the face geometry after construction must call update_face_gDiff!
Mesh3(cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
      boundaries, nodes, node_cells, get_float, get_int, boundary_cellsID) = Mesh3(
    cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
    face_gDiff_coefficients(faces), boundaries, nodes, node_cells, get_float, get_int,
    boundary_cellsID)

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