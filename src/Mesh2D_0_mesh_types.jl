export UnitVectors
export Node, Face2D, Cell

struct UnitVectors
    i::SVector{3, Float64}
    j::SVector{3, Float64}
    k::SVector{3, Float64}
    UnitVectors() = new(
        SVector(1.0,0.0,0.0), SVector(0.0,1.0,0.0), SVector(0.0,0.0,1.0))
end

struct Node{F}
    coords::SVector{3, F}
end
Node(F) = begin
    zf = zero(F)
    vec_3F = SVector{3,F}(zf,zf,zf)
    Node(vec_3F)
end
Node(x::F, y::F, z::F) where F<:AbstractFloat = Node(SVector{3, F}(x,y,z))
Node(zero::F) where F<:AbstractFloat = Node(zero,zero,zero)

struct Face2D{I,F}
    nodesID::SVector{2,I}
    ownerCells::SVector{2,I}
    centre::SVector{3, F}
    normal::SVector{3, F}
    area::F
    delta::F
end
Face2D(I,F) = begin
    zi = zero(I); zf = zero(F)
    vec_2I = SVector{2,I}(zi,zi)
    vec_3F = SVector{3,F}(zf,zf,zf)
    Face2D(vec_2I, vec_2I, vec_3F, vec_3F, zf, zf)
end

struct Cell{I,F}
    nodesID::SVector{4, I}
    facesID::SVector{4, I}
    neighbours::Vector{I}
    nsign::SVector{4, I}
    centre::SVector{3, F}
    volume::F
end
Cell(I,F) = begin
    zi = zero(I); zf = zero(F)
    vec_4I = SVector{4,I}(zi,zi,zi,zi)
    vec_3F = SVector{3,F}(zf,zf,zf)
    Cell(vec_4I, vec_4I, I[], vec_4I, vec_3F, zf)
end
# Cell(zi::I, zf::F) where {I,F}= begin
#     Cell(SVector{4, I}(zi,zi,zi,zi), SVector{3, F}(zf,zf,zf))
# end

struct Mesh2{I,F}
    cells::Vector{Cell{I,F}}
    faces::Vector{Face2D{I,F}}
    nodes::Vector{Node{F}}
end