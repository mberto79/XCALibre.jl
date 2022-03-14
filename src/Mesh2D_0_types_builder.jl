export AbstractPoint
export Point, Edge, Face2D, Patch, Element, Block
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
    pointsID::Vector{I}
    ncells::I
    boundary::Bool
end

struct Element{I<:Integer,F<:AbstractFloat}
    pointsID::SVector{4,I}
    centre::SVector{3,F}
end
Element(zi::I, zf::F) where {I,F}= begin
    Element(SVector{4, I}(zi,zi,zi,zi), SVector{3, F}(zf,zf,zf))
end

struct Patch{I<:Integer}
    name::Symbol
    edgesID::Vector{I}
end

struct Face2D{I<:Integer,F<:AbstractFloat}
    nodesID::SVector{2,I}
    centre::SVector{3,F}
end
Face2D(I,F) = begin
    zi = zero(I); zf = zero(F)
    Face2D(SVector{2,I}(zi,zi), SVector{3,F}(zf,zf,zf))
end
# Face2D(id1::I, id2::I) where I = Face2D(SVector{2,I}(id1, id2))

struct Block{I<:Integer}
    edgesID::SVector{4,I}
    nx::I
    ny::I
    pointsID::Matrix{I}
    elementsID::Matrix{I}
    facesID_NS::Matrix{I}
    facesID_EW::Matrix{I}
    inner_points::I
    linear::Bool
end
Block(zi::I) where I<:Integer = begin
    Block(
        SVector{4,I}(zi,zi,zi,zi),zi,zi,zeros(I, 2, 2),zeros(I, 2, 2),zeros(I, 2, 2),
        zeros(I, 2, 2),zi,true
    )
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
    faces::Vector{Face2D{I,F}}
end