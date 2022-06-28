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

mutable struct Boundary{TI<:Integer}
    name::String
    groupNumber::TI
    elements::Vector{TI}
end
Boundary(z::TI) where TI<:Integer = Boundary("default", zero(TI), TI[])

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