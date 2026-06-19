struct Point{F<:AbstractFloat, SV3<:SVector{3,F}}
    xyz::SV3
end
Point(z::TF) where TF<:AbstractFloat = Point(SVector{3, TF}(zero(TF), zero(TF), zero(TF)))

#Edges not used in the mesh at this time.
# mutable struct Edge{I<:Integer,VI<:AbstractArray{I}} 
#     edgeindex::I
#     edgeCount::I
#     edges::VI
# end
# Edge(z::TI) where TI<:Integer = Edge(0 , 0, TI[])

mutable struct Face{I<:Integer, VI<:AbstractArray{I}} 
    index::I
    nodeCount::I # total what? nodeCount or nnodes would make easier to review
    nodesID::VI
end
Face(z::TI) where TI<:Integer = Face(0 , 0, TI[])

mutable struct Cell_UNV{I<:Integer,VI<:AbstractArray{I}}
    index::I
    nodeCount::I # total what? nodeCount or nnodes would make easier to review
    nodesID::VI
end
Cell_UNV(z::TI) where TI<:Integer = Cell_UNV(zero(TI) , zero(TI), TI[])

mutable struct BoundaryElement{S<:String,I<:Integer,VI<:AbstractArray{I}}
    name::S
    index::I
    facesID::VI # these are nodes IDs - should probably just call them that
end
BoundaryElement(z::TI) where TI<:Integer = BoundaryElement("default", TI(0), TI[])

#Not used.
# mutable struct Element{I<:Integer,VI<:AbstractArray{I}}
#     index::I
#     elementCount::I
#     elements::VI
# end
# Element(z::TI) where TI<:Integer = Element(0,0,TI[])