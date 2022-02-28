export generate_mesh_1D

function define_element_1D(nodes::Vector{Node{F}}, nodeList::Vector{I}) where {I <: Integer, F}
    nNodes = length(nodeList)
    sum = SVector{3, F}(0., 0., 0.)
    for nID ∈ nodeList
        sum += nodes[nID].coords
    end
    centre = sum/nNodes
    return Element(nodeList, centre)
end

function Edge(nodes::Vector{Node{F}}, nodesID::Vector{I}) where {I <: Integer, F}
    centre = (nodes[nodesID[1]].coords .+ nodes[nodesID[2]].coords)./2
    return Edge(nodesID, centre)
end


function Face_1D(nodes::Vector{Node{F}}, edges::Vector{Edge{I,F}}, 
    elements::Vector{Element{I,F}}, edgeID, ownerCells) where {I <: Integer, F}
    nodesID = edges[edgeID].nodesID
    nEgdes = length(edges)
    tangentVector = nodes[nodesID[2]].coords - nodes[nodesID[1]].coords
    area = norm(tangentVector) 
    normal = (tangentVector × UnitVectors().k)./area
    
    if edgeID == 1
        ownerCells .= [1, 1]
        delta = norm(elements[ownerCells[1]].centre - edges[edgeID].centre)
        return Face(nodesID, ownerCells, edges[edgeID].centre, area, normal, delta)
    elseif edgeID == nEgdes-1
        ownerCells .= [nEgdes-2, nEgdes-1]
    elseif edgeID == nEgdes
        ownerCells .= [nEgdes-1, nEgdes-1]
        delta = norm(elements[ownerCells[1]].centre - edges[edgeID].centre)
        return Face(nodesID, ownerCells, edges[edgeID].centre, area, normal, delta)
    end

    delta = norm(elements[ownerCells[1]].centre - elements[ownerCells[2]].centre)
    return Face(nodesID, ownerCells, edges[edgeID].centre, area, normal, delta)
end

function Cell_1D(
    faces::Vector{Face{I,F}}, elements::Vector{Element{I,F}}, i) where {I <: Integer, F}
    element = elements[i]
    facesID = [i,i+1]
    neighbours = I[]
    for fID ∈ facesID
        push!(neighbours, (faces[fID].ownerCells .!= i) ⋅ faces[fID].ownerCells)
    end
    if i == 1
        nsign = [-1, -1]
    elseif i == length(elements)
        nsign = [-1, -1]
    else
        nsign = [-1, 1]
    end
    cellLength = norm(faces[facesID[2]].centre - faces[facesID[1]].centre)
    volume = faces[facesID[1]].area * cellLength
    return Cell(
        element.nodesID,
        facesID,
        neighbours,
        nsign, #[-1, 1], # nsign
        element.centre,
        volume
    )
end

function mesh1D_nodes(x0, xL, h, nCells::Int)
    dx = xL/(nCells)
    dy = h/2
    n = nCells+1
    nodes = Node{Float64}[]
    for i ∈ 1:n
        x = x0 + (i-1)*dx
        push!(nodes, Node(SVector{3, Float64}([x, -dy, 0.0])))
        push!(nodes, Node(SVector{3, Float64}([x, dy, 0.0])))
    end
    return nodes
end

function mesh1D_elements(nodes::Vector{Node{F}}, nCells::I) where {I <: Integer, F}
    elements = Element{I,F}[]
    for i ∈ 1:nCells
        δ = (i-1)*2
        nodeList = [1+δ, 2+δ, 4+δ, 3+δ]
        push!(elements, define_element_1D(nodes, nodeList))
    end
    return elements
end

function mesh1D_edges(nodes::Vector{Node{F}}) where F
    edges = Edge{Int,Float64}[]
    for i ∈ 1:2:length(nodes)
        push!(edges, Edge(nodes, [i, i+1]))
    end
    return edges
end

function mesh1D_faces(nodes::Vector{Node{F}}, edges::Vector{Edge{I,F}}, elements::Vector{Element{I,F}}) where {I <: Integer, F}
    nEdges = length(edges)
    faces = Face{Int,Float64}[]
    for i ∈ 1:nEdges # avoid boundary faces
        push!(faces, Face_1D(nodes, edges, elements, i, [i-1, i]))
    end
    faces[1].ownerCells .= [1, 1]
    faces[nEdges].ownerCells .= [nEdges-1, nEdges-1]
    return faces
end

function mesh1D_cells(elements::Vector{Element{I,F}}, faces::Vector{Face{I,F}}) where {I <: Integer, F}
    nCells = length(elements)
    cells = Cell{I,F}[]
    for i ∈ 1:nCells
        push!(cells, Cell_1D(faces, elements, i))
    end
    return cells
end

function generate_mesh_1D(x0, xL, h, nCells::Int)
    nodes       = mesh1D_nodes(x0, xL, h, nCells)
    elements    = mesh1D_elements(nodes, nCells)
    edges       = mesh1D_edges(nodes)
    faces       = mesh1D_faces(nodes, edges, elements)
    cells       = mesh1D_cells(elements, faces)
    elements    = 0
    return Mesh(cells, faces, nodes)
end