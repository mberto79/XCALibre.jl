export Node, Boundary, Cell
export Face2D, Face3D
export Mesh2, Mesh3
export AbstractMesh

abstract type AbstractMesh end

# struct UnitVectors{F<:AbstractFloat}
#     i::SVector{3, F}
#     j::SVector{3, F}
#     k::SVector{3, F}
# end
# Adapt.@adapt_structure UnitVectors

# UnitVectors(T) = UnitVectors{T}(
#         SVector{3,T}(1.0,0.0,0.0), 
#         SVector{3,T}(0.0,1.0,0.0), 
#         SVector{3,T}(0.0,0.0,1.0)
#         )

struct Node{SV3<:SVector{3,<:AbstractFloat}, UR<:UnitRange{<:Integer}}
    coords::SV3
    cells_range::UR # to access neighbour cells (can be dummy entry for now)
end
Adapt.@adapt_structure Node

struct Boundary{S<:Symbol, VI<:AbstractArray{<:Integer}}
    name::S
    facesID::VI
    cellsID::VI
end
Adapt.@adapt_structure Boundary

struct Cell{F<:AbstractFloat, SV3<:SVector{3,F},UR<:UnitRange{<:Integer}}
    centre::SV3
    volume::F
    nodes_range::UR
    faces_range::UR
end
Adapt.@adapt_structure Cell

# 2D and 3D Face types

struct Face2D{
    F<:AbstractFloat, 
    SV2<:SVector{3,<:Integer},
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
    SV2<:SVector{3,<:Integer},
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

# 2D and 3D Mesh types

struct Mesh2{VC, VI, VF<:AbstractArray{<:Face2D}, VB, VN} <: AbstractMesh
    cells::VC
    cell_nodes::VI
    cell_faces::VI
    cell_neighbours::VI
    cell_nsign::VI
    faces::VF
    face_nodes::VI
    boundaries::VB
    nodes::VN
    node_cells::VI # can be empty for now
end
Adapt.@adapt_structure Mesh2

struct Mesh3{VC, VI, VF<:AbstractArray{<:Face3D}, VB, VN} <: AbstractMesh
    cells::VC
    cell_nodes::VI
    cell_faces::VI
    cell_neighbours::VI
    cell_nsign::VI
    faces::VF
    face_nodes::VI
    boundaries::VB
    nodes::VN
    node_cells::VI # can be empty for now
end
Adapt.@adapt_structure Mesh3