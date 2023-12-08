export Face3D,Mesh3

struct Face3D{I<:Integer,UR<:UnitRange{I},F<:AbstractFloat,SV2<:SVector{2,I},SV3<:SVector{3,F}}
    nodes_range::UR
    ownerCells::SV2
    centre::SV3
    normal::SV3
    e::SV3
    area::F
    delta::F
    weight::F
end
Adapt.@adapt_structure Face3D

#abstract type AbstractMesh end

struct Mesh3{I<:Integer,VC<:AbstractArray{Cell},VN<:AbstractArray{Node},VF<:AbstractArray{Face3D},VB<:AbstractArray{Boundary},VI<:AbstractArray{I}} <: AbstractMesh
    cells::VC
    cell_nodes::VI
    cell_faces::VI
    cell_neighbours::VI
    cell_nsign::VI
    faces::VF
    face_nodes::VI
    boundaries::VB
    nodes::VN
end
Adapt.@adapt_structure Mesh3
