export Face2D, Mesh2

struct Face2D{I,F}
    nodes_range::UnitRange{I}
    ownerCells::SVector{2,I}
    centre::SVector{3, F}
    normal::SVector{3, F}
    e::SVector{3, F}
    area::F
    delta::F
    weight::F
end
Adapt.@adapt_structure Face2D

struct Mesh2{VC, VI, VF2, VB, VN} <: AbstractMesh
    cells::VC#Vector{Cell{I,F}}
    cell_nodes::VI#Vector{I}
    cell_faces::VI#Vector{I}
    cell_neighbours::VI#Vector{I}
    cell_nsign::VI#Vector{I}
    faces::VF2#Vector{Face2D{I,F}}
    face_nodes::VI#Vector{I}
    boundaries::VB#Vector{Boundary{I}}
    nodes::VN#Vector{Node{I,F}}
end
Adapt.@adapt_structure Mesh2