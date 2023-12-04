export Face3D,Mesh3

struct Face3D{I,F}
    nodes_range::UnitRange{I}
    ownerCells::SVector{2,I}
    centre::SVector{3, F}
    normal::SVector{3, F}
    e::SVector{3, F}
    area::F
    delta::F
    weight::F
end
Adapt.@adapt_structure Face3D
Face3D(I,F) = begin
    zi = zero(I); zf = zero(F)
    vec_2I = SVector{2,I}(zi,zi)
    vec_3I=SVector{3,I}(zi,zi,zi)
    vec_3F = SVector{3,F}(zf,zf,zf)
    Face3D(vec_3I, vec_2I, vec_3F, vec_3F, vec_3F, zf, zf, zf)
end

struct Mesh3{VC<:AbstractArray{Cell},VN<:AbstractArray{Node},VF<:AbstractArray{Face3D},VB<:AbstractArray{Boundary},VI<:AbstractArray{Int}} <: AbstractMesh
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
