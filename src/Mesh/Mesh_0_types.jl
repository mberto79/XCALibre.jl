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

# element arrays are stored per field so kernels move only the columns they read
_soa(x::StructArray) = x
_soa(x::AbstractArray) = StructArray(x)

struct Mesh2{VC, VI, VF<:AbstractArray{<:Face2D}, VTF, VB, VN, SV3, UR} <: AbstractMesh
    cells::VC
    cell_nodes::VI
    cell_faces::VI
    cell_neighbours::VI
    cell_nsign::VI
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
        c, f, n = _soa(cells), _soa(faces), _soa(nodes)
        new{typeof(c), typeof(cell_nodes), typeof(f), typeof(face_gDiff), typeof(boundaries),
            typeof(n), typeof(get_float), typeof(get_int)}(c, cell_nodes, cell_faces, cell_neighbours,
            cell_nsign, f, face_nodes, face_gDiff, boundaries, n, node_cells, get_float, get_int,
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
    struct Mesh3{VC, VI, VF<:AbstractArray{<:Face3D}, VTF, VB, VN, SV3, UR} <: AbstractMesh
        cells::VC           # vector of cells
        cell_nodes::VI      # vector of indices to access cell nodes
        cell_faces::VI      # vector of indices to access cell faces
        cell_neighbours::VI # vector of indices to access cell neighbours
        cell_nsign::VI      # vector of indices to with face normal correction (1 or -1 )
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
struct Mesh3{VC, VI, VF<:AbstractArray{<:Face3D}, VTF, VB, VN, SV3, UR} <: AbstractMesh
    cells::VC
    cell_nodes::VI
    cell_faces::VI
    cell_neighbours::VI
    cell_nsign::VI
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
        c, f, n = _soa(cells), _soa(faces), _soa(nodes)
        new{typeof(c), typeof(cell_nodes), typeof(f), typeof(face_gDiff), typeof(boundaries),
            typeof(n), typeof(get_float), typeof(get_int)}(c, cell_nodes, cell_faces, cell_neighbours,
            cell_nsign, f, face_nodes, face_gDiff, boundaries, n, node_cells, get_float, get_int,
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