using Plots
using FVM_1D
using Krylov
using StaticArrays
using CUDA

mesh2_from_UNV(mesh; integer=Int64, float=Float64) = begin
    boundaries = Vector{Boundary{Vector{integer}}}(undef, length(mesh.boundaries))
    cells = Vector{Cell{integer,float}}(undef, length(mesh.cells))
    faces = Vector{Face2D{integer,float}}(undef, length(mesh.faces))
    nodes = Vector{Node{Vector{integer},float}}(undef, length(mesh.nodes))

    for (i, b) ∈ enumerate(mesh.boundaries)
        boundaries[i] = Boundary{Vector{integer}}(b.name, b.facesID, b.cellsID)
    end

    for (i, n) ∈ enumerate(mesh.nodes)
        nodes[i] = Node(n.coords, n.neighbourCells)
    end

    # PROCESSING CELLS
    # Calculate array size needed for nodes and faces
    nnodes = zero(integer)
    nfaces = zero(integer)
    for cell ∈ mesh.cells
        (; nodesID, facesID) = cell 
        nnodes += length(nodesID)
        nfaces += length(facesID)
    end

    # Initialise arrays 
    cell_nodes = Vector{integer}(undef, nnodes)
    cell_faces = Vector{integer}(undef, nfaces)
    cell_neighbours = Vector{integer}(undef, nfaces)
    cell_nsign = Vector{integer}(undef, nfaces)

    ni = 1 # node counter
    fi = 1 # face counter
    for (i, cell) ∈ enumerate(mesh.cells)
        (;nodesID, facesID, neighbours, nsign) = cell
        # node array loop
        nodes_range = ni:(ni + length(nodesID) - 1) #SVector{2,integer}(length(nodesID), ni)
        for nodeID ∈ nodesID
            cell_nodes[ni] = nodeID 
            ni += 1
        end
        # cell array loop
        # faces_range = SVector{2,integer}(length(facesID), fi)
        faces_range = fi:(fi + length(facesID) - 1) 
        for j ∈ eachindex(facesID)
            cell_faces[fi] = facesID[j]
            cell_neighbours[fi] = neighbours[j]
            cell_nsign[fi] = nsign[j]
            fi += 1
        end

        # cell assignment
        cells[i] = Cell{integer,float}(
            cell.centre,
            cell.volume,
            nodes_range,
            faces_range
        ) |> cu
    end

    # PROCESSING FACES
    # Calculate array size needed for all face nodes
    nnodes = zero(integer)
    for face ∈ mesh.faces
        (; nodesID) = face 
        nnodes += length(nodesID)
    end

    # Initialise arrays 
    face_nodes = Vector{integer}(undef, nnodes)

    ni = 1 # node counter
    for (i, face) ∈ enumerate(mesh.faces)
        (;nodesID) = face
        # node array loop
        nodes_range = ni:(ni + length(nodesID) - 1) #SVector{2,integer}(length(nodesID), ni)
        for nodeID ∈ nodesID
            face_nodes[ni] = nodeID 
            ni += 1
        end

        # face assignment
        faces[i] = Face2D{integer,float}(
            nodes_range,
            face.ownerCells,
            face.centre,
            face.normal,
            face.e,
            face.area,
            face.delta,
            face.weight
        ) |> cu
    end

    Mesh2{Vector{Cell{integer,float}}, Vector{integer}, Vector{Face2D{integer,float}}, Vector{Boundary{Vector{integer}}}, Vector{Node{Vector{integer},float}}}(
        cells,
        cell_nodes,
        cell_faces,
        cell_neighbours,
        cell_nsign,
        faces,
        face_nodes,
        boundaries,
        nodes,
    ) |> cu
end

function test_kernel_cells!(mesh)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x

    (; cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign) = mesh
    
    @inbounds if i <= length(cells) && i > 0

        (; centre, volume, nodes_range, faces_range) = cells[i]
        
        for j in nodes_range

        cell_nodes[j] = cell_nodes[j] + cell_nodes[j]

        end

        for k in faces_range

            cell_faces[k] = cell_faces[k]+cell_faces[k]
            cell_neighbours[k] = cell_neighbours[k] + cell_neighbours[k]
            cell_nsign[k] = cell_nsign[k] + cell_nsign[k]

        end

        centre = centre + centre
        volume = volume + volume

        cells[i] = Cell{Int64, Float64}(
            centre,
            volume,
            nodes_range,
            faces_range
        )

    end

    mesh = Mesh2(
        cells,
        cell_nodes,
        cell_faces,
        cell_neighbours,
        cell_nsign,
        mesh.faces,
        mesh.face_nodes,
        mesh.boundaries,
        mesh.nodes,
    )

    return nothing
end 

function test_kernel_faces!(mesh)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x

    (; faces, face_nodes) = mesh

    @inbounds if i <= length(faces) && i > 0

        (; nodes_range, ownerCells, centre, normal, e, area, delta, weight) = faces[i]
       
        ownerCells = ownerCells + ownerCells
        centre = centre + centre
        normal = normal + normal
        e = e + e
        area = area + area
        delta = delta + delta
        weight = weight + weight

        for j in nodes_range
            face_nodes[j] = face_nodes[j] + face_nodes[j]
        end

        faces[i] = Face2D{Int64,Float64}(
        nodes_range,
        ownerCells,
        centre,
        normal,
        e,
        area,
        delta,
        weight
        ) 

    end

    mesh = Mesh2(
        mesh.cells,
        mesh.cell_nodes,
        mesh.cell_faces,
        mesh.cell_neighbours,
        mesh.cell_nsign,
        faces,
        face_nodes,
        mesh.boundaries,
        mesh.nodes,
    )

    return nothing
end

function test_kernel_boundaries!(mesh)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x

    (; boundaries) = mesh

    for j in 1:length(boundaries)

        (; name, facesID, cellsID) = boundaries[j]

        # @cuprint(name)

        @inbounds if i <= length(facesID) && i > 0

            facesID[i] = facesID[i] + facesID[i]
            cellsID[j] = cellsID[j] + cellsID[j]

        end

        boundaries[j] = Boundary(
            name,
            facesID,
            cellsID
        )

    end

    mesh = Mesh2(
        mesh.cells,
        mesh.cell_nodes,
        mesh.cell_faces,
        mesh.cell_neighbours,
        mesh.cell_nsign,
        mesh.faces,
        mesh.face_nodes,
        boundaries,
        mesh.nodes,
    )

    return nothing

end

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
unv_mesh = build_mesh(mesh_file, scale=0.001)

mesh = mesh2_from_UNV(unv_mesh)

# CELLS
mesh.cell_nodes
mesh.cell_faces
mesh.cells[1].centre

@cuda threads = 1024 blocks = cld(length(mesh.cells),1024) test_kernel_cells!(mesh)

mesh.cell_nodes
mesh.cell_faces
mesh.cells[1].centre



# FACES
mesh.faces[1].ownerCells
mesh.face_nodes

@cuda threads = 1024 blocks = cld(length(mesh.faces),1024) test_kernel_faces!(mesh)

mesh.faces[1].ownerCells
mesh.face_nodes


# boundaries

mesh.boundaries[1].facesID

@cuda threads = 1024 test_kernel_boundaries!(mesh)

