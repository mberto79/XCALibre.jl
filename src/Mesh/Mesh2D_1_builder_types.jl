export Point, Edge, Patch, Block
export Wireframe, MeshBuilder2D

Point(x::F, y::F, z::F) where F<:AbstractFloat = Node(SVector{3, F}(x,y,z))

struct Edge{I<:Integer}
    nodesID::Vector{I}
    ncells::I
    boundary::Bool
end

struct Patch{I<:Integer}
    name::Symbol
    edgesID::Vector{I}
end

struct Block{I<:Integer}
    edgesID::SVector{4,I}
    nx::I
    ny::I
    nodesID::Matrix{I}
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

struct MeshBuilder2D{I<:Integer,F<:AbstractFloat}
    points::Vector{Node{F}}
    edges::Vector{Edge{I}}
    patches::Vector{Patch{I}}
    blocks::Vector{Block{I}}
end