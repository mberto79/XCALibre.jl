export Node, Boundary, Cell
export AbstractMesh

abstract type AbstractMesh end

struct Node{TI, TF}
    coords::SVector{3, TF}
    neighbourCells::Vector{TI}
end
Adapt.@adapt_structure Node

struct Boundary{I}
    name::Symbol
    # nodesID::Vector{I} # can be deduced from face info
    facesID::Vector{I}
    cellsID::Vector{I}
    # normal::SVector{3, F} # correctly aligned by definition
end
Adapt.@adapt_structure Boundary

struct Cell{I,F}
    centre::SVector{3, F}
    volume::F
    map2nodes::SVector{2,I}
    map2faces::SVector{2,I}
    map2neighbours::SVector{2,I}
    map2nsign::SVector{2,I}
end
Adapt.@adapt_structure Cell