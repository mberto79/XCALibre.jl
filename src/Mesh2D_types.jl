export Point, Edge, Element, LinearBlock, Patch

struct Point{F<:AbstractFloat}
    coords::SVector{3,F}
    boundary::Bool
end
Point(x::F, y::F, z::F) where F<:AbstractFloat = Point(SVector{3, F}(x,y,z), false)
Point(zero::F) where F<:AbstractFloat = Point(zero,zero,zero)

struct Edge{I<:Integer,F<:AbstractFloat}
    p1::Point{F}
    p2::Point{F}
    n::I
    points::Vector{Point{F}}
    boundary::Bool
    nodesID::Vector{I}
end
Edge(p1::Point{F}, p2::Point{F}, n::I) where {I,F} = begin
    points  = fill(Point(zero(F)), n-1)
    nodesID = fill(zero(I), n+1)
    Edge(
        p1, p2, n, points, false, nodesID
    )
end

struct Element{I<:Integer,F<:AbstractFloat}
    nodesID::SVector{4,I}
    centre::SVector{3,F}
end

struct LinearBlock{I<:Integer,F<:AbstractFloat}
    ex1::Edge{I,F}
    ex2::Edge{I,F} 
    ey1::Edge{I,F} 
    ey2::Edge{I,F} 
    nx::I
    ny::I
    nodesID::Matrix{I} # use matrix here instead
end
LinearBlock(
    ex1::Edge{I,F},ex2::Edge{I,F},ey1::Edge{I,F},ey2::Edge{I,F},nx::I,ny::I
    ) where {I,F} = begin
    # if nx != ex1.n || ny != ey1.n
    #     println("Edge and block elements must be equal")
    #     return
    # end
    LinearBlock(
        ex1, ex2, ey1, ey2, nx, ny, 
        fill(zero(I), (nx+1), (ny+1)) # use matrix here instead
        )
end

struct Patch{I<:Integer}
    name::Symbol
    ID::I
    edgesID::Vector{I}
end
