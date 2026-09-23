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

# no closure over `to`: a captured type is stored as its kind and adapt stops inferring
Adapt.adapt_structure(to, x::FaceArrays) = _rewrap(x, (adapt(to, x.nodes_range), adapt(to, x.ownerCells),
    adapt(to, x.centre), adapt(to, x.normal), adapt(to, x.e), adapt(to, x.area), adapt(to, x.delta), adapt(to, x.weight)))
Adapt.adapt_structure(to, x::CellArrays) = _rewrap(x, (adapt(to, x.centre), adapt(to, x.volume),
    adapt(to, x.nodes_range), adapt(to, x.faces_range)))
Adapt.adapt_structure(to, x::NodeArrays) = _rewrap(x, (adapt(to, x.coords), adapt(to, x.cells_range)))

# normal signs are only ±1, so one byte each
_nsign(x::AbstractArray{Int8}) = x
_nsign(x::AbstractArray) = Int8.(x)

struct Mesh2{VV, VTF, VR, VO, VI, VS, VB, SV3, UR} <: AbstractMesh
    cell_centre::VV
    cell_volume::VTF
    cell_nodes_range::VR
    cell_faces_range::VR
    cell_nodes::VI
    cell_faces::VI
    cell_neighbours::VI
    cell_nsign::VS
    face_nodes_range::VR
    face_ownerCells::VO
    face_centre::VV
    face_normal::VV
    face_e::VV
    face_area::VTF
    face_delta::VTF
    face_weight::VTF
    face_nodes::VI
    face_gDiff::VTF
    boundaries::VB
    node_coords::VV
    node_cells_range::VR
    node_cells::VI
    get_float::SV3
    get_int::UR
    boundary_cellsID::VI
    function Mesh2(cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
        face_gDiff, boundaries, nodes, node_cells, get_float, get_int, boundary_cellsID)
        c, f, n, s = _soa(cells), _soa(faces), _soa(nodes), _nsign(cell_nsign)
        fields = (c.centre, c.volume, c.nodes_range, c.faces_range, cell_nodes, cell_faces,
            cell_neighbours, s, f.nodes_range, f.ownerCells, f.centre, f.normal, f.e, f.area, f.delta,
            f.weight, face_nodes, face_gDiff, boundaries, n.coords, n.cells_range, node_cells,
            get_float, get_int, boundary_cellsID)
        new{typeof(f.centre), typeof(f.area), typeof(f.nodes_range), typeof(f.ownerCells),
            typeof(cell_nodes), typeof(s), typeof(boundaries), typeof(get_float), typeof(get_int)}(fields...)
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

struct Mesh3{VV, VTF, VR, VO, VI, VS, VB, SV3, UR} <: AbstractMesh
    cell_centre::VV
    cell_volume::VTF
    cell_nodes_range::VR
    cell_faces_range::VR
    cell_nodes::VI
    cell_faces::VI
    cell_neighbours::VI
    cell_nsign::VS
    face_nodes_range::VR
    face_ownerCells::VO
    face_centre::VV
    face_normal::VV
    face_e::VV
    face_area::VTF
    face_delta::VTF
    face_weight::VTF
    face_nodes::VI
    face_gDiff::VTF
    boundaries::VB
    node_coords::VV
    node_cells_range::VR
    node_cells::VI
    get_float::SV3
    get_int::UR
    boundary_cellsID::VI
    function Mesh3(cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, faces, face_nodes,
        face_gDiff, boundaries, nodes, node_cells, get_float, get_int, boundary_cellsID)
        c, f, n, s = _soa(cells), _soa(faces), _soa(nodes), _nsign(cell_nsign)
        fields = (c.centre, c.volume, c.nodes_range, c.faces_range, cell_nodes, cell_faces,
            cell_neighbours, s, f.nodes_range, f.ownerCells, f.centre, f.normal, f.e, f.area, f.delta,
            f.weight, face_nodes, face_gDiff, boundaries, n.coords, n.cells_range, node_cells,
            get_float, get_int, boundary_cellsID)
        new{typeof(f.centre), typeof(f.area), typeof(f.nodes_range), typeof(f.ownerCells),
            typeof(cell_nodes), typeof(s), typeof(boundaries), typeof(get_float), typeof(get_int)}(fields...)
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

Adapt.adapt_structure(to, m::Mesh3) = Mesh3(adapt(to, m.cells), adapt(to, m.cell_nodes),
    adapt(to, m.cell_faces), adapt(to, m.cell_neighbours), adapt(to, m.cell_nsign), adapt(to, m.faces),
    adapt(to, m.face_nodes), adapt(to, m.face_gDiff), adapt(to, m.boundaries), adapt(to, m.nodes),
    adapt(to, m.node_cells), adapt(to, m.get_float), adapt(to, m.get_int), adapt(to, m.boundary_cellsID))

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