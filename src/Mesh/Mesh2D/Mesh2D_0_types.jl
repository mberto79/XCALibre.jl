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
Adapt.@adapt_structure Face2D

struct Mesh2{I,F} <: AbstractMesh
    cells::Vector{Cell{I,F}}
    faces::Vector{Face2D{I,F}}
    boundaries::Vector{Boundary{I}}
    nodes::Vector{Node{I,F}}
end
Adapt.@adapt_structure Mesh2