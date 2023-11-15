export Face3D, Mesh3

struct Face3D{I,F}
    nodes_map::SVector{3,I}
    ownerCells::SVector{2,I}
    centre::SVector{3, F}
    normal::SVector{3, F}
    e::SVector{3, F}
    area::F
    delta::F
    weight::F
end
Adapt.@adapt_structure Face3D

struct Mesh3{I,F} <: AbstractMesh
    cells::Vector{Cell{I,F}}
    cell_nodes::Vector{I}
    cell_faces::Vector{I}
    cell_neighbours::Vector{I}
    cell_nsign::Vector{I}
    faces::Vector{Face3D{I,F}}
    face_nodes::Vector{I}
    boundaries::Vector{Boundary{I}}
    nodes::Vector{Node{I,F}}
end
Adapt.@adapt_structure Mesh3

