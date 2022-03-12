using Plots
using StaticArrays
using LinearAlgebra

using FVM_1D.Mesh2D
using FVM_1D.Plotting

n_vertical      = 3
n_horizontal1   = 5
n_horizontal2   = 4

p1 = Point(0.0,0.0,0.0)
p2 = Point(1.0,0.0,0.0)
p3 = Point(1.5,0.0,0.0)
p4 = Point(0.0,1.0,0.0)
p5 = Point(0.8,0.8,0.0)
p6 = Point(1.5,0.7,0.0)
points = [p1,p2,p3,p4,p5,p6]

# Edges in x-direction
# e1 = Edge(1,2,n_horizontal1)
e1 = line!(points,1,2,n_horizontal1)
e2 = line!(points,2,3,n_horizontal2)
e3 = line!(points,4,5,n_horizontal1)
e4 = line!(points,5,6,n_horizontal2)

# Edges in y-direction
e5 = line!(points,1,4,n_vertical)
e6 = line!(points,2,5,n_vertical)
e7 = line!(points,3,6,n_vertical)
edges = [e1,e2,e3,e4,e5,e6,e7]

b1 = quad(edges, [1,3,5,6])
b2 = quad(edges, [2,4,6,7])
blocks = [b1,b2]

patch1 = Patch(:inlet,  [5])
patch2 = Patch(:outlet, [7])
patch3 = Patch(:bottom, [1,2])
patch4 = Patch(:top,    [3,4])
patches = [patch1, patch2, patch3, patch4]

@time mesh_info = Wireframe(points, edges, patches, blocks)
@time builder = preallocate_mesh(mesh_info)
@time generate_inner_points!(builder, mesh_info)
@time generate_elements!(builder, mesh_info)

fig = plot(builder.points)
plot!(fig, centre.(builder.elements), colour=:red)

function centre(element::Element{I,F}) where {I,F}
    Point(element.centre)
end

function generate_elements!(
    builder::MeshBuilder2D{I,F}, mesh_info::Wireframe{I,F}
    ) where {I,F}
    blocks = mesh_info.blocks
    points = builder.points
    elements = builder.elements
    elementi = zero(I)
    for block ∈ blocks
        nx, ny = block.nx, block.ny
        nodesID = block.nodesID
        for yi ∈ 1:ny
            for xi ∈ 1:nx
                elementi += 1
                n1 = nodesID[xi,yi]
                n2 = nodesID[xi+1,yi]
                n3 = nodesID[xi+1,yi+1]
                n4 = nodesID[xi,yi+1]
                nodeList = SVector{4, I}(n1,n2,n3,n4)
                p1 = points[n1]
                p2 = points[n2]
                p3 = points[n3]
                p4 = points[n4]
                centre = geometric_centre((p1,p2,p3,p4))
                elements[elementi] = Element(nodeList, centre)
            end
        end
    end
end

function geometric_centre(points::NTuple{N, Point{F}}) where {N,F}
    sum = SVector{3, F}(0.0,0.0,0.0)
        for point ∈ points
            sum += point.coords
        end
    return sum/(length(points))
end

function generate_inner_points!(
    builder::MeshBuilder2D{I,F}, mesh_info::Wireframe{I,F}
    ) where {I,F}
    points_count = total_edge_points(mesh_info)
    points = mesh_info.points
    edges = mesh_info.edges
    blocks = mesh_info.blocks
    for block ∈ blocks
        edgeID1 = block.edgesID[3] 
        edgeID2 = block.edgesID[4] 
        edge_y1 = edges[edgeID1] # "y-dir" edge 1
        edge_y2 = edges[edgeID2] # "y-dir" edge 2
        for yi ∈ 2:block.ny
            pID1 = edge_y1.pointsID[yi]
            pID2 = edge_y2.pointsID[yi]
            p1 = points[pID1]
            p2 = points[pID2]
            spacing, normal = linear_distribution(p1, p2, block.nx)
            for xi ∈ 2:block.nx
                points_count += 1
                builder.points[points_count] = Point(p1.coords + spacing*normal*(xi-1))
                # push!(pts, Point(p1.coords + spacing*normal*(xi-1)))
                # pointsID[j] = length(pts)
                block.nodesID[xi, yi] = points_count
            end
        end
    end
end

function preallocate_mesh(mesh_info::Wireframe{I,F}) where {I,F}
    points = fill(Point(zero(F)), total_points(mesh_info))
    elements = fill(Element(zero(I), zero(F)), total_elements(mesh_info))
    # Copy existing edge points to new points vector
    for i ∈ eachindex(mesh_info.points)
        points[i] = mesh_info.points[i]
    end
    MeshBuilder2D(points, elements)
end

function total_edge_points(mesh_info::Wireframe{I,F}) where {I,F}
    length(mesh_info.points)
end

function total_points(mesh_info::Wireframe{I,F}) where {I,F}
    edge_points = total_edge_points(mesh_info)
    inner_points = zero(I)
    for block ∈ mesh_info.blocks
        inner_points += block.inner_points
    end
    edge_points + inner_points
end

function total_elements(mesh_info::Wireframe{I,F}) where {I,F}
    total_elements = zero(I)
    for block ∈ mesh_info.blocks
        total_elements += block.nx*block.ny
    end
    total_elements
end

function line!(pts::Vector{Point{F}}, p1_index::I, p2_index::I, ncells::I) where {I,F}
    pointsID = fill(zero(I), ncells+1)
    pointsID[1] = p1_index
    pointsID[end] = p2_index

    p1 = pts[p1_index]
    p2 = pts[p2_index]
    spacing, normal = linear_distribution(p1, p2, ncells)
    for j ∈ 2:ncells
        push!(pts, Point(p1.coords + spacing*normal*(j-1)))
        pointsID[j] = length(pts)
    end
    return Edge(pointsID, ncells, false)
end

function linear_distribution(p1::Point{F}, p2::Point{F}, ncells::I) where {I,F}
    d = p2.coords - p1.coords
    d_mag = norm(d)
    normal = d/d_mag
    spacing = d_mag/ncells
    spacing, normal
end

function quad(edges::Vector{Edge{I}}, edgesID::Vector{I}) where {I,F}
    IDs = SVector{4,I}(edgesID)
    nx = edges[IDs[1]].ncells
    ny = edges[IDs[3]].ncells
    ID_matrix =  zeros(I, nx+1, ny+1)
    ID_matrix[:,1]      = edges[IDs[1]].pointsID
    ID_matrix[:,end]    = edges[IDs[2]].pointsID
    ID_matrix[1,:]      = edges[IDs[3]].pointsID
    ID_matrix[end,:]    = edges[IDs[4]].pointsID
    inner_points = (nx+1-2)*(ny+1-2)
    Block(IDs, nx, ny, ID_matrix, inner_points, true)
end

# function curve(pts::Vector{Point{F}}, p1_index::I, p2_index::I, ncells::I) where {I,F}
#     pointsID = fill(zero(I), ncells+1)
#     # points = fill(Point(zero(F)), ncells-1)
#     pointsID[1] = p1_index
#     pointsID[end] = p2_index

#     p1 = pts[p1_index]
#     p2 = pts[p2_index]

#     # points[1] = p1; points[end] = p2

#     d = p2.coords - p1.coords
#     d_mag = norm(d)
#     e1 = d/d_mag
#     spacing = d_mag/ncells
#     for j ∈ 2:ncells
#     # for j ∈ eachindex(points)
#         # points[j] = Point(spacing*e1*j + p1.coords)
#         push!(pts, Point(p1.coords +
#             [spacing*e1*(j-1),
#             # spacing*e1[2]*(j-1),
#             spacing*e1[2]*(j-1) + 0.25*sin(pi*spacing*e1[1]*(j-1)),
#             spacing*e1[3]*(j-1)]))
#         pointsID[j] = length(pts)

#     end
#     return Edge(pointsID, ncells, false)
# end

# MeshDefinition(points, edges, patches, blocks)   

# domain = define_mesh(n_vertical, n_horizontal1, n_horizontal2)
# tag_boundaries!(domain)
# multiblock = build_multiblock(domain)
# counter = generate_boundary_nodes!(multiblock)
# counter = generate_internal_edge_nodes!(multiblock, counter)
# generate_internal_nodes!(multiblock, counter)
# build_elements!(multiblock)

# multiblock = nothing
# @time multiblock = generate(n_vertical, n_horizontal1, n_horizontal2)
# c = centres(multiblock.elements)

# fig = plot(multiblock.nodes)
# plot!(fig, c; colour=:red)
# plot!(fig, multiblock.definition.points, colour=:red)