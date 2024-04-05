using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK_3D/VTU.jl")


unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/5_cell_new_boundaries.unv"


points, edges, bfaces, volumes, boundaryElements = load_3D(unv_mesh,scale=1, integer=Int64, float=Float64)

points
edges
bfaces
volumes
boundaryElements

@time mesh = build_mesh3D(unv_mesh)

cell_nodes, cell_nodes_range = FVM_1D.UNV_3D.generate_cell_nodes(volumes) # Should be Hybrid compatible, tested for hexa. Using push instead of allocating vector.
node_cells, node_cells_range = FVM_1D.UNV_3D.generate_node_cells(points, volumes)
bface_nodes, bface_nodes_range, bface_owners_cells, boundary_cellsID = FVM_1D.UNV_3D.generate_boundary_faces(boundaryElements, bfaces, node_cells, node_cells_range, volumes)
iface_nodes, iface_nodes_range, iface_owners_cells = FVM_1D.UNV_3D.generate_internal_faces(volumes, bfaces, nodes, node_cells)

get_data(array, range, index) = @view array[range[index]]
get_data(array, range) =  array[range] #@view array[range] # 
nodeIDs = get_data
faceIDs = get_data
cellIDs = get_data

function generate_internal_faces(volumes, bfaces, nodes, node_cells)
    # determine total number of faces based on cell type (including duplicates)
    total_faces = 0
    for volume ∈ volumes
        # add faces for tets
        if volume.volumeCount == 4
            total_faces += 4
        end
        # add conditions to add faces for other cell types
        # add faces for Hexa
        if volume.volumeCount == 8 
            total_faces += 6
        end
        #add faces for Wedge/Penta
        if volume.volumeCount == 6
            total_faces += 5
        end
    end

    # Face nodeIDs for each cell is a vector of vectors of vectors :-)
    cells_faces_nodeIDs = Vector{Vector{Int64}}[Vector{Int64}[] for _ ∈ 1:length(volumes)] 

    # Generate all faces for each cell/element/volume
    for (cellID, volume) ∈ enumerate(volumes)
        # Generate faces for tet elements
        if volume.volumeCount == 4
            nodesID = volume.volumes
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[3]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[3], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[2], nodesID[3], nodesID[4]])
        end
        # add conditions for other cell types
        # Generate faces for hexa elements using UNV structure method
        if volume.volumeCount == 8
            nodesID = volume.volumes
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[3], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[5], nodesID[6], nodesID[7], nodesID[8]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[5], nodesID[6]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[3], nodesID[4], nodesID[7], nodesID[8]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[2], nodesID[3], nodesID[6], nodesID[7]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[4], nodesID[5], nodesID[8]])
        end
        # Pattern for faces needs to be found for wedge elements
    end

    # Sort nodesIDs for each face based on ID (need to correct order later to be physical)
    # this allows to find duplicates later more easily (query on ordered ids is faster)
    for face_nodesID ∈ cells_faces_nodeIDs
        for nodesID ∈ face_nodesID
            sort!(nodesID)
        end
    end

    # Find owner cells for each face
    owners_cellIDs = Vector{Int64}[zeros(Int64, 2) for _ ∈ 1:total_faces]
    facei = 0 # faceID counter (will be reduced to internal faces later)
    for (cellID, faces_nodeIDs) ∈ enumerate(cells_faces_nodeIDs) # loop over cells
        for facei_nodeIDs ∈ faces_nodeIDs # loop over vectors of nodeIDs for each face
            facei += 1 # face counter
            owners_cellIDs[facei][1] = cellID # ID of first cell containing the face
            for nodeID ∈ facei_nodeIDs # loop over ID of each node in the face
                cells_range = nodes[nodeID].cells_range
                node_cellIDs = @view node_cells[cells_range] # find cells that use this node
                for nodei_cellID ∈ node_cellIDs # loop over cells that share the face node
                    if nodei_cellID !== cellID # ensure cellID is not same as current cell 
                        for face ∈ cells_faces_nodeIDs[nodei_cellID]
                            if face == facei_nodeIDs
                                owners_cellIDs[facei][2] = nodei_cellID # set owner cell ID 
                                break
                            end
                        end
                    end
                end
            end
        end
    end

    # Sort all face owner vectors
    sort!.(owners_cellIDs) # in-place sorting

    # Extract nodesIDs for each face from all cells into a vector of vectors
    face_nodes = Vector{Int64}[Int64[] for _ ∈ 1:total_faces] # nodesID for all faces
    fID = 0 # counter to keep track of faceID
    for celli_faces_nodeIDs ∈ cells_faces_nodeIDs
        for nodesID ∈ celli_faces_nodeIDs
            fID += 1
            face_nodes[fID] = nodesID
        end
    end

    # Remove duplicates
    unique_indices = unique(i -> face_nodes[i], eachindex(face_nodes))
    unique!(face_nodes)
    keepat!(owners_cellIDs, unique_indices)

    # Remove boundary faces

    total_bfaces = 0 # count boundary faces
    for owners ∈ owners_cellIDs
        if owners[1] == 0
            total_bfaces += 1
        end
    end

    bfaces_indices = zeros(Int64, total_bfaces) # preallocate memory
    counter = 0
    for (i, owners) ∈ enumerate(owners_cellIDs)
        if owners[1] == 0
            counter += 1
            bfaces_indices[counter] = i # contains indices of faces to remove
        end
    end

    deleteat!(owners_cellIDs, bfaces_indices)
    deleteat!(face_nodes, bfaces_indices)

    println("Removing ", total_bfaces, " (from ", length(bfaces), ") boundary faces")

    # Generate face_nodes_range
    face_nodes_range = Vector{UnitRange{Int64}}(undef, length(face_nodes))
    start = 1
    for (fID, nodesID) ∈ enumerate(face_nodes)
        nnodes = length(nodesID)
        face_nodes_range[fID] = UnitRange{Int64}(start:(start + nnodes - 1))
        start += nnodes
    end

    # Flatten array i.e. go from Vector{Vector{Int}} to Vector{Int}
    face_nodes = vcat(face_nodes...) 

    return face_nodes, face_nodes_range, owners_cellIDs
end