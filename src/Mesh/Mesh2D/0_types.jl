export Face2D, Mesh2

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

struct Mesh2{I,F} <: AbstractMesh
    cells::Vector{Cell{I,F}}
    faces::Vector{Face2D{I,F}}
    boundaries::Vector{Boundary{I}}
    nodes::Vector{Node{F}}
end