#UNV3Dtypes
struct Point{TF<:AbstractFloat}
    xyz::SVector{3, TF}
end
Point(z::TF) where TF<:AbstractFloat = Point(SVector{3, TF}(zero(TF), zero(TF), zero(TF)))

mutable struct Vertex{TI<:Integer} 
    vertexindex::TI
    vertexCount::TI
    vertices::Vector{TI}
end
Vertex(z::TI) where TI<:Integer = Vertex(0 , 0, TI[])

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

mutable struct BoundaryCondition{TI<:Integer}
    name::String
    bcNumber::TI
    elements::Vector{TI}
end
BoundaryCondition(z::TI) where TI<:Integer = BoundaryCondition("default", 0, TI[])

mutable struct Element{TI<:Integer}
    index::TI
    elementCount::TI
    elements::Vector{TI}
end
Element(z::TI) where TI<:Integer = Element(0,0,TI[])