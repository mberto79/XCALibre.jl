struct Point{F<:AbstractFloat, SV3<:SVector{3,F}}
    xyz::SV3
end
Point(z::TF) where TF<:AbstractFloat = Point(SVector{3, TF}(zero(TF), zero(TF), zero(TF)))


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
