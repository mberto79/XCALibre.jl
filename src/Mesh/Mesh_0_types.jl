export Node, Boundary, Cell
export AbstractMesh

abstract type AbstractMesh end

struct Node{VTI, TF}
    coords::SVector{3, TF}
    neighbourCells::VTI#Vector{TI}
end
Adapt.@adapt_structure Node

struct Boundary{VI}
    name::Symbol
    # nodesID::Vector{I} # can be deduced from face info
    facesID::VI#Vector{I}
    cellsID::VI#Vector{I}
    # normal::SVector{3, F} # correctly aligned by definition
end
Adapt.@adapt_structure Boundary

struct Cell{I,F}
    centre::SVector{3, F}
    volume::F
    nodes_range::UnitRange{I}
    faces_range::UnitRange{I}
end
Adapt.@adapt_structure Cell