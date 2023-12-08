export Node, Boundary, Cell
export AbstractMesh

abstract type AbstractMesh end

struct UnitVectors
    i::SVector{3, Float64}
    j::SVector{3, Float64}
    k::SVector{3, Float64}
    UnitVectors() = new(
        SVector(1.0,0.0,0.0), SVector(0.0,1.0,0.0), SVector(0.0,0.0,1.0))
end
Adapt.@adapt_structure UnitVectors

struct Node{I<:Integer, F<:AbstractFloat, VI<:AbstractArray{I}, SV3<:SVector{3,F}}
    coords::SV3
    neighbourCells::VI
end
Adapt.@adapt_structure Node

struct Boundary{S<:Symbol, I<:Integer, VI<:AbstractArray{I}}
    name::S
    facesID::VI
    cellsID::VI
end
Adapt.@adapt_structure Boundary

struct Cell{F<:AbstractFloat,I<:Integer, SV3<:SVector{3,F},UR<:UnitRange{I}}
    centre::SV3
    volume::F
    nodes_range::UR
    faces_range::UR
end
Adapt.@adapt_structure Cell