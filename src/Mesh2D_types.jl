export AbstractPoint
export Point, Edge, Element, Block, Patch, MeshDefinition, Node, MultiBlock

abstract type AbstractPoint end

struct Point{F<:AbstractFloat} <:AbstractPoint
    coords::SVector{3,F}
    boundary::Bool
    processed::Bool
    ID::Int64 # Temp definition. Need to make parametric i.e. I (changing all signatures!)
end
Point(x::F, y::F, z::F) where F<:AbstractFloat = begin 
    Point(SVector{3, F}(x,y,z), false, false, 0)
end
Point(zero::F) where F<:AbstractFloat = Point(zero,zero,zero)

struct Node{F<:AbstractFloat} <:AbstractPoint
    coords::SVector{3,F}
end
Node(x::F, y::F, z::F) where F<:AbstractFloat = Node(SVector{3, F}(x,y,z))
Node(zero::F) where F<:AbstractFloat = Node(zero,zero,zero)

struct Edge{I<:Integer}
    p1::I
    p2::I
    n::I
    # points::Vector{Point{F}}
    boundary::Bool
    nodesID::Vector{I}
end
Edge(p1::I, p2::I, n::I) where {I} = begin
    # points  = fill(Point(zero(F)), n-1)
    nodesID = fill(zero(I), n+1)
    Edge(
        # p1, p2, n, points, false, nodesID
        p1, p2, n, false, nodesID
    )
end

struct Element{I<:Integer,F<:AbstractFloat}
    nodesID::SVector{4,I}
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
    edge_x1::I
    edge_x2::I
    edge_y1::I 
    edge_y2::I 
    nx::I
    ny::I
    nodesID::Matrix{I} # use matrix here instead
end
Block(
    ex1::I,ex2::I,ey1::I,ey2::I,nx::I,ny::I
    ) where I = begin
    # if nx != ex1.n || ny != ey1.n
    #     println("Edge and block elements must be equal")
    #     return
    # end
    Block(
        ex1, ex2, ey1, ey2, nx, ny, 
        fill(zero(I), (nx+1), (ny+1)) # use matrix here instead
        )
end

struct MeshDefinition{I<:Integer,F<:AbstractFloat}
    points::Vector{Point{F}}
    edges::Vector{Edge{I}}
    patches::Vector{Patch{I}}
    blocks::Vector{Block{I}}
end

struct MultiBlock{I<:Integer,F<:AbstractFloat}
    definition::MeshDefinition{I,F}
    elements::Vector{Element{I,F}}
    nodes::Vector{Node{F}}
end
