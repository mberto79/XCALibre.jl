struct Point{F<:AbstractFloat, SV3<:SVector{3,F}}
    xyz::SV3
end
Point(z::TF) where TF<:AbstractFloat = Point(SVector{3, TF}(zero(TF), zero(TF), zero(TF)))

mutable struct Edge{I<:Integer,VI<:AbstractArray{I}} 
    edgeindex::I
    edgeCount::I
    edges::VI
end
Edge(z::TI) where TI<:Integer = Edge(0 , 0, TI[])

mutable struct Face{I<:Integer, VI<:AbstractArray{I}} 
    faceindex::I
    faceCount::I
    faces::VI
end
Face(z::TI) where TI<:Integer = Face(0 , 0, TI[])

mutable struct Volume{I<:Integer,VI<:AbstractArray{I}}
    volumeindex::I
    volumeCount::I
    volumes::VI
end
Volume(z::TI) where TI<:Integer = Volume(zero(TI) , zero(TI), TI[])

mutable struct BoundaryElement{S<:String,I<:Integer,VI<:AbstractArray{I}}
    name::S
    boundaryNumber::I
    elements::VI
end
BoundaryElement(z::TI) where TI<:Integer = BoundaryElement("default", 0, TI[])

mutable struct Element{I<:Integer,VI<:AbstractArray{I}}
    index::I
    elementCount::I
    elements::VI
end
Element(z::TI) where TI<:Integer = Element(0,0,TI[])