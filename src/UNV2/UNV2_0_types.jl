struct Point{TF<:AbstractFloat}
    xyz::SVector{3, TF}
end
Point(z::TF) where TF<:AbstractFloat = Point(SVector{3, TF}(zero(TF), zero(TF), zero(TF)))

mutable struct Element{TI<:Integer} # <: AbstractArray{Int, 1}
    index::TI
    vertexCount::TI
    vertices::Vector{TI}
end
Element(z::TI) where TI<:Integer = Element(0 , 0, TI[])

mutable struct BoundaryLoader{TI<:Integer}
    name::String
    groupNumber::TI
    elements::Vector{TI}
end
BoundaryLoader(z::TI) where TI<:Integer = BoundaryLoader("default", zero(TI), TI[])

struct Node{TI, TF}
    coords::SVector{3, TF}
    neighbourCells::Vector{TI}
end
Node(TF) = begin
    zf = zero(TF)
    vec_3F = SVector{3,TF}(zf,zf,zf)
    Node(vec_3F, Int64[])
end
Node(x::F, y::F, z::F) where F<:AbstractFloat = Node(SVector{3, F}(x,y,z), Int64[])
Node(zero::F) where F<:AbstractFloat = Node(zero, zero, zero)
Node(vector::F) where F<:AbstractVector = Node(vector, Int64[])

struct Boundary{I}
    name::Symbol
    # nodesID::Vector{I}
    nodesID::Vector{Vector{I}}
    facesID::Vector{I}
    cellsID::Vector{I}
    # normal::SVector{3, F}
end

struct Cell{I,F}
    nodesID::Vector{I}
    facesID::Vector{I}
    neighbours::Vector{I}
    nsign::Vector{I}
    centre::SVector{3, F}
    volume::F
end
Cell(I,F) = begin
    zf = zero(F)
    vec3F = SVector{3,F}(zf,zf,zf)
    Cell(I[], I[], I[], I[], vec3F, zf)
end

struct Face2D{I,F}
    nodesID::SVector{2,I}
    ownerCells::SVector{2,I}
    centre::SVector{3, F}
    normal::SVector{3, F}
    e::SVector{3, F}
    area::F
    delta::F
    weight::F
end
Face2D(I,F) = begin
    zi = zero(I); zf = zero(F)
    vec_2I = SVector{2,I}(zi,zi)
    vec_3F = SVector{3,F}(zf,zf,zf)
    Face2D(vec_2I, vec_2I, vec_3F, vec_3F, vec_3F, zf, zf, zf)
end

struct Mesh2{I,F} <: AbstractMesh2{I,F}
    cells::Vector{Cell{I,F}}
    faces::Vector{Face2D{I,F}}
    boundaries::Vector{Boundary{I}}
    nodes::Vector{Node{I,F}}
end


function get_xyz(P::Vector{Point{TF}}) where TF
    N = length(P)
    x = zeros(TF, N)
    y = zeros(TF, N)
    z = zeros(TF, N)
    for i ∈ 1:N
        x[i] = P[i].xyz[1]
        y[i] = P[i].xyz[2]
        z[i] = P[i].xyz[3]
    end
    return x, y, z
end

function get_xyz(P::Vector{Vector{TF}}) where TF<:AbstractFloat
    N = length(P)
    x = zeros(TF, N)
    y = zeros(TF, N)
    z = zeros(TF, N)
    for i ∈ 1:N
        x[i] = P[i][1]
        y[i] = P[i][2]
        z[i] = P[i][3]
    end
    return x, y, z
end

# function midPoint(a::Vector{Point})
#     sumx, sumy, sumz = 0, 0, 0 
#     # sum = Point(0,0,0)
#     N = length(a)
#     for i ∈ 1:N
#         sumx += a[i].xyz[1] # x coord
#         sumy += a[i].xyz[2] # y coord
#         sumz += a[i].xyz[3] # z coord
#     end
#     return Point([sumx/N, sumy/N, sumz/N])
# end

function midPoint(a::Vector{Vector{TF}}) where TF<:AbstractFloat
    sumx, sumy, sumz = 0, 0, 0 
    # sum = Point(0,0,0)
    N = length(a)
    for i ∈ 1:N
        sumx += a[i][1] # x coord
        sumy += a[i][2] # y coord
        sumz += a[i][3] # z coord
    end
    # return [sumx/N, sumy/N, sumz/N]
    return SVector{3, TF}(sumx/N,sumy/N,sumz/N)
end