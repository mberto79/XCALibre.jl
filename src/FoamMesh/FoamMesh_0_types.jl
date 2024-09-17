
mutable struct FoamMeshData{B,P,F,I}
    boundaries::B
    points::P
    faces::F
    n_cells::I
    n_faces::I
    n_ifaces::I
    n_bfaces::I
end
FoamMeshData(TI, TF) = FoamMeshData(
        Boundary{TI,Symbol}[],
        SVector{3, TF}[],
        Face{TI}[],
        zero(TI),
        zero(TI),
        zero(TI),
        zero(TI)
    )

mutable struct Boundary{I<:Integer, S<:Symbol}
    name::S
    startFace::I
    nFaces::I
end
Boundary(TI) = Boundary(:default, zero(TI), zero(TI))

mutable struct Face{I}
    nodesID::Vector{I}
    owner::I
    neighbour::I
end
Face(TI, nnodes::I) where I<:Integer = begin
    nodesIDs = zeros(TI, nnodes)
    z = zero(TI)
    Face(nodesIDs, z, z)
end