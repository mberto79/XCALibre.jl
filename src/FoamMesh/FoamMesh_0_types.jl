
mutable struct FoamMeshData{B,P,I}
    boundaries::B
    points::P
    face_nodes::Vector{I}
    face_nodes_range::Vector{UnitRange{I}}
    face_owner::Vector{I}
    face_neighbour::Vector{I}
    n_cells::I
    n_faces::I
    n_ifaces::I
    n_bfaces::I
end
FoamMeshData(TI, TF) = FoamMeshData(
        Boundary{TI,Symbol}[],
        SVector{3, TF}[],
        TI[],
        UnitRange{TI}[],
        TI[],
        TI[],
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