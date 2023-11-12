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
    # nodesID::Vector{I}
    nodesID::Vector{Vector{I}}
    facesID::Vector{I}
    cellsID::Vector{I}
    # normal::SVector{3, F}
end
Adapt.@adapt_structure Boundary

struct Cell{I,F}
    nodesID::Vector{I}
    facesID::Vector{I}
    neighbours::Vector{I}
    nsign::Vector{I}
    centre::SVector{3, F}
    volume::F
end
Adapt.@adapt_structure Cell