export UnitVectors
export Node, Face2D, Boundary, Cell, Mesh2

struct UnitVectors
    i::SVector{3, Float64}
    j::SVector{3, Float64}
    k::SVector{3, Float64}
    UnitVectors() = new(
        SVector(1.0,0.0,0.0), SVector(0.0,1.0,0.0), SVector(0.0,0.0,1.0))
end

struct Node{TF}
    coords::SVector{3, TF}
    neighbourCells::Vector{Int32}
end
Node(TF) = begin
    zf = zero(TF)
    vec_3F = SVector{3,TF}(zf,zf,zf)
    Node(vec_3F, Int32[])
end
Node(x::F, y::F, z::F) where F<:AbstractFloat = Node(SVector{3, F}(x,y,z), Int32[])
Node(zero::F) where F<:AbstractFloat = Node(zero, zero, zero)
Node(vector::F) where F<:AbstractVector = Node(vector, Int32[])

struct Face2D{I,F}
    nodesID::SVector{2,I}
    ownerCells::SVector{2,I}
    centre::SVector{3, F}
    normal::SVector{3, F}
    e::SVector{3, F}
    area::F
    delta::F
    weight::F
end
Face2D(I,F) = begin
    zi = zero(I); zf = zero(F)
    vec_2I = SVector{2,I}(zi,zi)
    vec_3F = SVector{3,F}(zf,zf,zf)
    Face2D(vec_2I, vec_2I, vec_3F, vec_3F, vec_3F, zf, zf, zf)
end

struct Boundary{I}
    name::Symbol
    nodesID::Vector{I}
    facesID::Vector{I}
    cellsID::Vector{I}
    # normal::SVector{3, F}
end

struct Cell{I,F}
    nodesID::Vector{I}
    facesID::Vector{I}
    neighbours::Vector{I}
    nsign::Vector{I}
    centre::SVector{3, F}
    volume::F
end
Cell(I,F) = begin
    zf = zero(F)
    vec3F = SVector{3,F}(zf,zf,zf)
    Cell(I[], I[], I[], I[], vec3F, zf)
end

struct Mesh2{I,F}
    cells::Vector{Cell{I,F}}
    faces::Vector{Face2D{I,F}}
    boundaries::Vector{Boundary{I}}
    nodes::Vector{Node{F}}
end