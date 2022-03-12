export AbstractPoint
export Point, Edge, Element, Block, Patch 
export Wireframe, MeshBuilder2D

abstract type AbstractPoint end

struct Point{F<:AbstractFloat} <:AbstractPoint
    coords::SVector{3,F}
    boundary::Bool
end
Point(coords::SVector{3,F}) where F<:AbstractFloat = Point(coords, false)
Point(x::F, y::F, z::F) where F<:AbstractFloat = Point(SVector{3, F}(x,y,z), false)
Point(zero::F) where F<:AbstractFloat = Point(zero,zero,zero)

struct Edge{I<:Integer}
    # points::Vector{Point{F}}
    pointsID::Vector{I}
    ncells::I
    boundary::Bool
end
# Edge(p1::I, p2::I, ncells::I) where {I} = begin
#     pointsID = fill(zero(I), ncells+1)
#     # pointsID[1]      = p1
#     # pointsID[end]    = p2
#     # Edge(points, pointsID, ncells, false)
#     Edge(pointsID, ncells, false)
# end
# Edge(f) = (p1,p2,ncells) -> f(Edge(p1,p2,ncells))

struct Element{I<:Integer,F<:AbstractFloat}
    pointsID::SVector{4,I}
    centre::SVector{3,F}
end
Element(zi::I, zf::F) where {I,F}= begin
    Element(SVector{4, I}(zi,zi,zi,zi), SVector{3, F}(zf,zf,zf))
end

struct Patch{I<:Integer}
    name::Symbol
    # ID::I
    edgesID::Vector{I}
end

struct Block{I<:Integer}
    edgesID::SVector{4,I}
    nx::I
    ny::I
    nodesID::Matrix{I}
    inner_points::I
    linear::Bool
end

struct Wireframe{I<:Integer,F<:AbstractFloat}
    points::Vector{Point{F}}
    edges::Vector{Edge{I}}
    patches::Vector{Patch{I}}
    blocks::Vector{Block{I}}
end

struct MeshBuilder2D{I<:Integer,F<:AbstractFloat}
    points::Vector{Point{F}}
    elements::Vector{Element{I,F}}
end

# struct MultiBlock{I<:Integer,F<:AbstractFloat}
#     definition::MeshDefinition{I,F}
#     elements::Vector{Element{I,F}}
#     nodes::Vector{Node{F}}
# end
