export Node, Boundary,Cell 

struct Node{VI<:AbstractArray{Int}, SV3<:SVector{3,Float64}}
    coords::SV3
    neighbourCells::VI
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

struct Boundary{S<:Symbol, VI<:AbstractArray{Int}}
    name::S
    # nodesID::Vector{I} # can be deduced from face info
    facesID::VI
    cellsID::VI
    # normal::SVector{3, F} # correctly aligned by definition
end
Adapt.@adapt_structure Boundary

struct Cell{F<:AbstractFloat,I<:Integer, SV3<:SVector{3,F},UR<:UnitRange{I}}
    centre::SV3
    volume::F
    nodes_range::UR
    # faces_map::SVector{2,I}
    faces_range::UR
end
Adapt.@adapt_structure Cell