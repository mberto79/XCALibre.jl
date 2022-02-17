export UnitVectors

struct UnitVectors
    i::SVector{3, Float64}
    j::SVector{3, Float64}
    k::SVector{3, Float64}
    UnitVectors() = new(SVector(1.0,0.0,0.0), SVector(0.0,1.0,0.0), SVector(0.0,0.0,1.0))
end

struct Node{F}
    coords::SVector{3, F}
end

struct Element{I,F}
    nodesID::Vector{I}
    centre::SVector{3, F}
end

struct Edge{I,F}
    nodesID::Vector{I}
    centre::SVector{3, F}
end

struct Face{I,F}
    nodesID::Vector{I}
    # edgesID::Vector{I}
    ownerCells::Vector{I}
    centre::SVector{3, F}
    area::F
    normal::SVector{3, F}
    delta::F
end

struct Cell{I,F}
    nodesID::Vector{I}
    facesID::Vector{I}
    neighbours::Vector{I}
    nsign::Vector{I}
    centre::SVector{3, F}
    volume::F
end

struct Mesh{I,F}
    cells::Vector{Cell{I,F}}
    faces::Vector{Face{I,F}}
    nodes::Vector{Node{F}}
end