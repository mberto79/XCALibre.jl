#UNV3Dtypes
struct Point{TF<:AbstractFloat}
    xyz::SVector{3, TF}
end
Point(z::TF) where TF<:AbstractFloat = Point(SVector{3, TF}(zero(TF), zero(TF), zero(TF)))

mutable struct Vertices{TI<:Integer} 
    vertexindex1::TI
    vertexCount1::TI
    vertices1::Vector{TI}
end
Vertices(z::TI) where TI<:Integer = Vertices(0 , 0, TI[])

mutable struct Faces{TI<:Integer} 
    faceindex::TI
    faceCount::TI
    faces::Vector{TI}
end
Faces(z::TI) where TI<:Integer = Faces(0 , 0, TI[])

mutable struct Volumes{TI<:Integer}
    groupindex::TI
    groupCount::TI
    groups::Vector{TI}
end
Volumes(z::TI) where TI<:Integer = Volumes(0 , 0, TI[])

mutable struct BoundaryLoader{TI<:Integer}
    name::String
    groupNumber::TI
    elements::Vector{TI}
end
BoundaryLoader(z::TI) where TI<:Integer = BoundaryLoader("default", 0, TI[])

mutable struct Element{TI<:Integer}
    index::TI
    vertexCount::TI
    vertices::Vector{TI}
end
Element(z::TI) where TI<:Integer = Element(0,0,TI[])