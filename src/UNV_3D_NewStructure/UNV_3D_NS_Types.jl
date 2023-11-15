struct Node{TI, TF}
    coords::SVector{3, TF}
    neighbourCells::Vector{TI}
end
Adapt.@adapt_structure Node
Node(TF) = begin
    zf = zero(TF)
    vec_3F = SVector{3,TF}(zf,zf,zf)
    Node(vec_3F, Int64[])
end
Node(x::F, y::F, z::F) where F<:AbstractFloat = Node(SVector{3, F}(x,y,z), Int64[])
Node(zero::F) where F<:AbstractFloat = Node(zero, zero, zero)
Node(vector::F) where F<:AbstractVector = Node(vector, Int64[])

struct Boundary{I}
    name::Symbol
    # nodesID::Vector{I} # can be deduced from face info
    facesID::Vector{I}
    cellsID::Vector{I}
    # normal::SVector{3, F} # correctly aligned by definition
end
Adapt.@adapt_structure Boundary

struct Cell{I,F}
    centre::SVector{3, F}
    volume::F
    nodes_map::UnitRange{I}
    # faces_map::SVector{2,I}
    faces_map::UnitRange{I}
end
Adapt.@adapt_structure Cell


struct Point{TF<:AbstractFloat}
    xyz::SVector{3, TF}
end
Point(z::TF) where TF<:AbstractFloat = Point(SVector{3, TF}(zero(TF), zero(TF), zero(TF)))

mutable struct Edge{TI<:Integer} 
    edgeindex::TI
    edgeCount::TI
    edges::Vector{TI}
end
Edge(z::TI) where TI<:Integer = Edge(0 , 0, TI[])

mutable struct Face{TI<:Integer} 
    faceindex::TI
    faceCount::TI
    faces::Vector{TI}
end
Face(z::TI) where TI<:Integer = Face(0 , 0, TI[])

mutable struct Volume{TI<:Integer}
    groupindex::TI
    groupCount::TI
    groups::Vector{TI}
end
Volume(z::TI) where TI<:Integer = Volume(0 , 0, TI[])

mutable struct BoundaryElement{TI<:Integer}
    name::String
    boundaryNumber::TI
    elements::Vector{TI}
end
BoundaryElement(z::TI) where TI<:Integer = BoundaryElement("default", 0, TI[])

mutable struct Element{TI<:Integer}
    index::TI
    elementCount::TI
    elements::Vector{TI}
end
Element(z::TI) where TI<:Integer = Element(0,0,TI[])
