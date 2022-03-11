export AbstractPoint
export Point, Edge, Element, Block, Patch 
# export MeshDefinition, MultiBlock

abstract type AbstractPoint end

struct Point{F<:AbstractFloat} <:AbstractPoint
    coords::SVector{3,F}
    boundary::Bool
end
Point(x::F, y::F, z::F) where F<:AbstractFloat = begin 
    Point(SVector{3, F}(x,y,z), false)
end
Point(zero::F) where F<:AbstractFloat = Point(zero,zero,zero)

struct Edge{I<:Integer,F}
    points::Vector{Point{F}}
    pointsID::Vector{I}
    npoints::I
    boundary::Bool
end
Edge(p1::I, p2::I, ncells::I) where {I} = begin
    pointsID = fill(zero(I), ncells+1)
    pointsID[1]      = p1
    pointsID[end]    = p2
    Edge(pointsID, ncells, false)
end
# Edge(f) = (p1,p2,ncells) -> f(Edge(p1,p2,ncells))

struct Element{I<:Integer,F<:AbstractFloat}
    pointsID::SVector{4,I}
    centre::SVector{3,F}
end
Element(i::I, f::F) where {I,F} = begin
    Element(SVector{4, I}(i,i,i,i), SVector{3, F}(f,f,f))
end

struct Patch{I<:Integer}
    name::Symbol
    ID::I
    edgesID::Vector{I}
end

struct Block{I<:Integer}
    edgesID::SVector{4,I}
    nx::I
    ny::I
    nodesID::Matrix{I} # use matrix here instead
end
Block(edgesID::Vector{I}, nx::I, ny::I) where I = begin
    Block(SVector{4,I}(edgesID), nx, ny, fill( zero(I), nx+1, ny+1) )
end

# struct MeshDefinition{I<:Integer,F<:AbstractFloat}
#     points::Vector{Point{F}}
#     edges::Vector{Edge{I}}
#     patches::Vector{Patch{I}}
#     blocks::Vector{Block{I}}
# end

# struct MultiBlock{I<:Integer,F<:AbstractFloat}
#     definition::MeshDefinition{I,F}
#     elements::Vector{Element{I,F}}
#     nodes::Vector{Node{F}}
# end
