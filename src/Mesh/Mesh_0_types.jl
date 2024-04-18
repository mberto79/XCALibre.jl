export Node, Boundary, Cell
export Face2D, Face3D
export Mesh2, Mesh3
export AbstractMesh

abstract type AbstractMesh end

# MESH CONSTITUENTS

# Node structure
struct Node{SV3<:SVector{3,<:AbstractFloat}, UR<:UnitRange{<:Integer}}
    coords::SV3
    cells_range::UR # to access neighbour cells (can be dummy entry for now)
end
Adapt.@adapt_structure Node

# Boundary structure
struct Boundary{S<:Symbol, UR<:UnitRange{<:Integer}}
    name::S
    IDs_range::UR
end
Adapt.@adapt_structure Boundary

# Cell structure
struct Cell{F<:AbstractFloat, SV3<:SVector{3,F},UR<:UnitRange{<:Integer}}
    centre::SV3
    volume::F
    nodes_range::UR
    faces_range::UR
end
Adapt.@adapt_structure Cell

Cell(TI::T, TF::T) where T<:DataType = begin
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

struct Face3D{
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
Adapt.@adapt_structure Face3D

Face3D(TI::T, TF::T) where T<:DataType = begin
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

struct Mesh2{VC, VI, VF<:AbstractArray{<:Face2D}, VB, VN, SV3, UR} <: AbstractMesh
    cells::VC
    cell_nodes::VI
    cell_faces::VI
    cell_neighbours::VI
    cell_nsign::VI
    faces::VF
    face_nodes::VI
    boundaries::VB
    nodes::VN
    node_cells::VI
    get_float::SV3
    get_int::UR
    boundary_cellsID::VI
end
Adapt.@adapt_structure Mesh2

struct Mesh3{VC, VI, VF<:AbstractArray{<:Face3D}, VB, VN, SV3, UR} <: AbstractMesh
    cells::VC
    cell_nodes::VI
    cell_faces::VI
    cell_neighbours::VI
    cell_nsign::VI
    faces::VF
    face_nodes::VI
    boundaries::VB
    nodes::VN
    node_cells::VI
    get_float::SV3
    get_int::UR
    boundary_cellsID::VI
end
Adapt.@adapt_structure Mesh3