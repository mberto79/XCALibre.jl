export Face2D, Mesh2

struct Face2D{I,F}
    face2nodes::SVector{2,I}
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
    cell2nodes::Vector{I}
    cell2faces::Vector{I}
    cell2neighbours::Vector{I}
    cell2nsign::Vector{I}
    faces::Vector{Face2D{I,F}}
    face2nodes::Vector{I}
    boundaries::Vector{Boundary{I}}
    nodes::Vector{Node{I,F}}
end
Adapt.@adapt_structure Mesh2