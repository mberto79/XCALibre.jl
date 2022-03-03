export Point, Edge, Element, Block

struct Point{F<:AbstractFloat}
    coords::SVector{3,F}
end
Point(x::F, y::F, z::F) where F<:AbstractFloat = Point(SVector{3, F}(x,y,z))

struct Edge{F<:AbstractFloat}
    p1::Point{F}
    p2::Point{F}
end

struct Element{I<:Integer,F<:AbstractFloat}
    nodesID::SVector{4,I}
    centre::SVector{3,F}
end

struct Block{I<:Integer,F<:AbstractFloat}
    p1::Point{F}
    p2::Point{F} 
    p3::Point{F} 
    p4::Point{F} 
    nx::I
    ny::I
    nodes::Vector{Point{F}}
    elements::Vector{Element{I,F}}
end
Block(p1::Point{F},p2::Point{F},p3::Point{F},p4::Point{F},nx::I,ny::I) where {I,F} = begin
    Block(
        p1, p2, p3, p4, nx, ny, 
        fill(Point(0.0, 0.0, 0.0), (nx+1)*(ny+1)),
        fill(Element(SVector{4, I}(0,0,0,0), SVector{3, F}(0.0,0.0,0.0)), (nx)*(ny)))
end