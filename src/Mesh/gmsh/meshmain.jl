################################################################################
## Anemoi Developemnt Code                                                    ##
## Mesh Construction                                                          ##
## Author: Christopher D. Ellis                                               ##
## Date: 04/04/2020 20:15                                                     ##
################################################################################

using StatsBase

struct Cell
    nodes::Array{Int64, 1}
    coord::Array{Float64, 1}
    vol::Float64
end

struct Face
    coord::Array{Float64, 1}
    n::Array{Float64, 1}
    S::Float64
    lambda::Float64
    delta::Float64
    cellcon::Array{Int64, 1}
    AdrdPN::Float64
    At::Array{Float64, 1}
    f1f::Array{Float64, 1}
end

struct Boundary
    id::Int64
    type::Int64
    uniformval::Bool
    U::Union{Float64, Array{Float64, 1}}
    V::Union{Float64, Array{Float64, 1}}
    W::Union{Float64, Array{Float64, 1}}
    P::Union{Float64, Array{Float64, 1}}
    cellcon::Array{Int64, 1}
    facecon::Array{Int64, 1}
end

struct MeshNew
    cells::Array{Cell, 1}
    faces::Array{Face, 1}
    nodes::Array{Float64, 2}
    boundaries::Array{Boundary, 1}
end


function buildmesh3d(setupfilename, gmesh)
    println("Building mesh from .gmsh connectivity...")

    numcells = gmesh.vconnect[1].numElementsInBlock
    println("Num. Cells: ", numcells)

    cellvol = calcvol(gmesh.nodes, gmesh.vconnect[1].connectivity, gmesh.vconnect[1].elementType)

    # Cell centres is next
    cellcentre = calccent(gmesh.nodes, gmesh.vconnect[1].connectivity)

    cells = Array{Cell, 1}(undef, numcells)
    for i = 1:numcells
        cells[i] = Cell(gmesh.vconnect[1].connectivity[i][:], cellcentre[i,:], cellvol[i])
    end

    numnodes = size(gmesh.nodes, 1)
    println("Num. Nodes: ", numnodes)

    nodecell = [Vector{Int}() for _ in 1:numnodes]

    for i = 1:length(gmesh.vconnect[1].connectivity)
        for j = 1:length(gmesh.vconnect[1].connectivity[i])
            nodeid = gmesh.vconnect[1].connectivity[i][j]
            push!(nodecell[nodeid], i)
        end
    end

    println("Arranging and calculating face properties...")
    facenode = Vector[]
    facecell = Vector[]
    faceS = Vector[]

    for i = 1:length(gmesh.vconnect[1].connectivity)
        if length(gmesh.vconnect[1].connectivity[i]) == 8  # Hexahedral
            localid = [[1, 8, 5, 4],[2, 4, 1, 3],[6, 3, 2, 7],[5, 7, 6, 8],[8, 3, 7, 4],[2, 5, 6, 1]]
        elseif length(gmesh.vconnect[1].connectivity[i]) == 6  # Prism
            localid = [[1, 2, 3],[6, 5, 4],[1, 5, 2, 4],[3, 4, 1, 6],[2, 6, 3, 5]]
        elseif length(gmesh.vconnect[1].connectivity[i]) == 5  # Pyramid
            localid = [[1, 5, 2],[2, 5, 3],[3, 5, 4],[4, 5, 1],[1, 3, 4, 2]]
        elseif length(gmesh.vconnect[1].connectivity[i]) == 4  # Tetrahedral
            localid = [[0, 2, 3],[1, 2, 0],[1, 2, 3],[0, 3, 1]]
        end
        for j = 1:length(localid)
            nodeids = gmesh.vconnect[1].connectivity[i][localid[j][1:3]]
            cellsposs = modes(vcat(nodecell[nodeids,:]...))
            if length(cellsposs) == 2
                id = findall(x->x!=i, cellsposs)
                if !([cellsposs[id[1]], i] in facecell)
                    push!(facecell, [i, cellsposs[id[1]]])
                    push!(facenode, gmesh.vconnect[1].connectivity[i][localid[j]])
                end
            else
                push!(facecell, [i])
                push!(facenode, gmesh.vconnect[1].connectivity[i][localid[j]])
            end

        end
    end
    numfaces = length(facecell)
    println("Num. Faces: ", numfaces)

    faceS, facen = calcfacenormals(gmesh.nodes, facenode, numfaces)

    facedelta, facelambda, facecoords, AdrdPN, At, f1f = calcdeltalambda(cellcentre, gmesh.nodes, facenode, facecell, numfaces, faceS, facen)

    faces = Array{Face, 1}(undef, numfaces)
    for i = 1:numfaces
        faces[i] = Face(facecoords[i, :], facen[i, :], faceS[i], facelambda[i], facedelta[i], facecell[i], AdrdPN[i], At[i,:], f1f[i,:])
    end

    nodeface = [Vector{Int}() for _ in 1:numfaces]

    for i = 1:numfaces
        for j = 1:length(facenode[i])
            nodeid = facenode[i][j]
            push!(nodeface[nodeid], i)
        end
    end

    # Boundary Stuff
    println("Sorting the boundary patches...")

    println("Num. Boundary: ", length(gmesh.sconnect))

    boundaries = Array{Boundary, 1}(undef, length(gmesh.sconnect))

    this_setup = readsetup(setupfilename)
    allsurf = this_setup["boundaryinfo"]
    boundarytype = Dict{Integer,Integer}()
    boundaryuniform = Dict{Integer,Integer}()
    boundaryub = Dict{Integer,Integer}()
    boundaryvb = Dict{Integer,Integer}()
    boundarywb = Dict{Integer,Integer}()
    boundarypb = Dict{Integer,Integer}()
    for i = 1:length(allsurf)
        here = allsurf[i]
        enttag = here["entitytag"]
        tyid = here["typeid"]
        uniform = here["uniform"]
        Ub = here["Ub"]
        Vb = here["Vb"]
        Wb = here["Wb"]
        Pb = here["Pb"]
        boundarytype[enttag] = tyid
        boundaryuniform[enttag] = uniform
        boundaryub[enttag] = Ub
        boundaryvb[enttag] = Vb
        boundarywb[enttag] = Wb
        boundarypb[enttag] = Pb
    end

    for i = 1:length(gmesh.sconnect)
        boundid = gmesh.sconnect[i].entityTag
        type = boundarytype[boundid]
        numelements = gmesh.sconnect[i].numElementsInBlock
        facecon = Array{Int}(undef, numelements)
        cellcon = Array{Int}(undef, numelements)
        uniformval = boundaryuniform[boundid]
        U = Float64(boundaryub[boundid])
        V = Float64(boundaryvb[boundid])
        W = Float64(boundarywb[boundid])
        P = Float64(boundarypb[boundid])
        for j = 1:numelements
            nodeids = gmesh.sconnect[i].connectivity[j][1:3]
            posscells = modes(vcat(nodecell[nodeids,:]...))
            cellcon[j] = posscells[1]
            possfaces = modes(vcat(nodeface[nodeids,:]...))
            facecon[j] = possfaces[1]
        end
        boundaries[i] = Boundary(boundid, type, uniformval, U, V, W, P, cellcon, facecon)
    end

    mesh = MeshNew(cells, faces, gmesh.nodes, boundaries)

    return mesh
end


function calcvol(nodes, connectivity, elementType)
    volume = Array{Float32,1}(undef,size(connectivity))
    for i = 1:length(connectivity)
        if length(connectivity[i]) == 8  # Hexahedral
            id = [0 7 3 2;7 2 6 5;7 2 0 5;7 4 0 5;2 1 5 0].+1
            volume[i] = 0
            for j = 1:5
                temp = ones(4, 4)
                temp[1, 1:3] = nodes[connectivity[i][id[j, 1]],:]
                temp[2, 1:3] = nodes[connectivity[i][id[j, 2]],:]
                temp[3, 1:3] = nodes[connectivity[i][id[j, 3]],:]
                temp[4, 1:3] = nodes[connectivity[i][id[j, 4]],:]
                volume[i] += abs(det(temp))
            end
            volume[i] /= 6
        elseif length(connectivity[i]) == 6  # Prism
            id = [0 1 2 3;1 2 3 5;1 3 4 5].+1
            volume[i] = 0
            for j = 1:3
                temp = ones(4, 4)
                temp[1, 1:3] = nodes[connectivity[i][id[j, 1]],:]
                temp[2, 1:3] = nodes[connectivity[i][id[j, 2]],:]
                temp[3, 1:3] = nodes[connectivity[i][id[j, 3]],:]
                temp[4, 1:3] = nodes[connectivity[i][id[j, 4]],:]
                volume[i] += abs(det(temp))
            end
            volume[i] /= 6
        elseif length(connectivity[i]) == 5  # Pyramid
            id = [0 1 2 4;0 2 3 4].+1
            volume[i] = 0
            for j = 1:2
                temp = ones(4, 4)
                temp[1, 1:3] = nodes[connectivity[i][id[j, 1]],:]
                temp[2, 1:3] = nodes[connectivity[i][id[j, 2]],:]
                temp[3, 1:3] = nodes[connectivity[i][id[j, 3]],:]
                temp[4, 1:3] = nodes[connectivity[i][id[j, 4]],:]
                volume[i] += abs(det(temp))
            end
            volume[i] /= 6
        elseif length(connectivity[i]) == 4  # Tetrahedral
            id = [0 1 2 3].+1
            volume[i] = 0
            for j = 1:1
                temp = ones(4, 4)
                temp[1, 1:3] = nodes[connectivity[i][id[j, 1]],:]
                temp[2, 1:3] = nodes[connectivity[i][id[j, 2]],:]
                temp[3, 1:3] = nodes[connectivity[i][id[j, 3]],:]
                temp[4, 1:3] = nodes[connectivity[i][id[j, 4]],:]
                volume[i] += abs(det(temp))
            end
            volume[i] /= 6
        end
    end
    return volume
end

function calccent(nodes, connectivity)
    cellcentre = Array{Float32, 2}(undef, length(connectivity), 3)
    for i = 1:length(connectivity)
        cellcentre[i, :] = mean(nodes[connectivity[i], :], dims=1)
    end
    return cellcentre
end

function calcfacenormals(nodes, facenode, numfaces)
    faceS = Array{Float64}(undef, numfaces)
    facen = Array{Float64, 2}(undef, numfaces, 3)
    for i = 1:numfaces
        if length(facenode[i]) == 4  # Quadrilateral faces
            temp1 = nodes[facenode[i][2], :] .- nodes[facenode[i][1], :]
            temp2 = nodes[facenode[i][4], :] .- nodes[facenode[i][3], :]
            Sn = 0.5*cross(temp1, temp2)
            faceS[i] = norm(Sn)
            facen[i, :] = Sn./norm(Sn)
        elseif length(facenode[i]) == 3  # Triangle faces
            temp1 = nodes[facenode[i][1], :] .- nodes[facenode[i][2], :]
            temp2 = nodes[facenode[i][3], :] .- nodes[facenode[i][2], :]
            Sn = 0.5*cross(temp1, temp2)
            faceS[i] = norm(Sn)
            facen[i, :] = Sn./norm(Sn)
        end
    end

    return faceS, facen
end


function calcdeltalambda(cellcentre, nodes, facenode, facecell, numfaces, S, n)
    facedelta = Array{Float64}(undef, numfaces)
    facelambda = Array{Float64}(undef, numfaces)
    facecoords = Array{Float64, 2}(undef, numfaces, 3)

    AdrdPN = Array{Float64}(undef, numfaces)
    At = Array{Float64, 2}(undef, numfaces, 3)
    f1f = Array{Float64, 2}(undef, numfaces, 3)

    for i = 1:numfaces
        xf = mean(nodes[facenode[i], 1], dims=1)
        yf = mean(nodes[facenode[i], 2], dims=1)
        zf = mean(nodes[facenode[i], 3], dims=1)
        facecoords[i, :] = [xf[1], yf[1], zf[1]]
        if length(facecell[i]) == 1  # Boundary
            facelambda[i] = -1
            facedelta[i] = norm(cellcentre[facecell[i][1],:]-facecoords[i,:])
            AdrdPN[i] = dot(S[i].*n[i, :], S[i].*n[i, :])/dot(S[i].*n[i, :], facecoords[i,:].-cellcentre[facecell[i][1],:])
            At[i, :] = S[i].*n[i, :] .- AdrdPN[i] .* facedelta[i] .* (facecoords[i,:].-cellcentre[facecell[i][1],:])

            f1f[i, :] = [0, 0, 0]
        else  # Internal Face
            temp = norm(cellcentre[facecell[i][1],:]-facecoords[i,:])
            facedelta[i] = norm(cellcentre[facecell[i][1],:]-cellcentre[facecell[i][2],:])
            facelambda[i] = 1-(temp/facedelta[i])
            AdrdPN[i] = dot(S[i].*n[i, :], S[i].*n[i, :])/dot(S[i].*n[i, :], cellcentre[facecell[i][2],:].-cellcentre[facecell[i][1],:])
            n_PN = (cellcentre[facecell[i][2],:].-cellcentre[facecell[i][1],:])./norm(cellcentre[facecell[i][2],:].-cellcentre[facecell[i][1],:])
            Ad = S[i]/dot(n_PN ,n[i, :])
            At[i, :] = S[i].*n[i, :] .- Ad .* n_PN
            Pf = facecoords[i,:].-cellcentre[facecell[i][1],:]
            f1f[i, :] = Pf .- (dot(Pf, n[i,:])/dot(cellcentre[facecell[i][2],:].-cellcentre[facecell[i][1],:], n[i,:])) .* (cellcentre[facecell[i][2],:].-cellcentre[facecell[i][1],:])
            PN = cellcentre[facecell[i][2],:].-cellcentre[facecell[i][1],:]
            facelambda[i] = 1 - (dot(Pf, n_PN)./sqrt(dot(PN, PN)))
        end
    end

    return facedelta, facelambda, facecoords, AdrdPN, At, f1f
end
