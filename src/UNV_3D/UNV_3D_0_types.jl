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
    volumeindex::TI
    volumeCount::TI
    volumes::Vector{TI}
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