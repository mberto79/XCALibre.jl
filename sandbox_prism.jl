using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK_3D/VTU.jl")


unv_mesh="src/UNV_3D/TET_PRISM_HM.unv"
# unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/HEXA_HM.unv"
# unv_mesh="src/UNV_3D/TET_HM.unv"

@time mesh = build_mesh3D(unv_mesh)
mesh.faces
mesh.cells
mesh.boundaries

name="tet_prism"

write_vtk(name, mesh::Mesh3)

points, edges, efaces, volumes, boundaryElements = load_3D(unv_mesh,scale=1, integer=Int64, float=Float64)

points
edges
efaces
volumes
boundaryElements

cell_nodes, cell_nodes_range = FVM_1D.UNV_3D.generate_cell_nodes(volumes) # Should be Hybrid compatible, tested for hexa. Using push instead of allocating vector.
node_cells, node_cells_range = FVM_1D.UNV_3D.generate_node_cells(points, volumes)  # Should be Hybrid compatible, tested for hexa.
nodes = FVM_1D.UNV_3D.build_nodes(points, node_cells_range) # Hyrbid compatible, works for Tet and Hexa
boundaries = FVM_1D.UNV_3D.build_boundaries(boundaryElements) # Hybrid compatible

nbfaces = sum(length.(getproperty.(boundaries, :IDs_range))) # total boundary faces

bface_nodes, bface_nodes_range, bface_owners_cells, boundary_cellsID = 
begin
    FVM_1D.UNV_3D.generate_boundary_faces(boundaryElements, efaces, nbfaces, node_cells, node_cells_range, volumes) # Hybrid compatible, tested with hexa
end

iface_nodes, iface_nodes_range, iface_owners_cells = 
begin 
    FVM_1D.UNV_3D.generate_internal_faces(volumes, nbfaces, nodes, node_cells) # Hybrid compatible, tested with hexa.
end

# NOTE: A function will be needed here to reorder the nodes IDs of "faces" to be geometrically sound! (not needed for tet cells though)
bface_nodes,iface_nodes=FVM_1D.UNV_3D.order_face_nodes(bface_nodes_range,iface_nodes_range,bface_nodes,iface_nodes,nodes)
#2 methods, using old as new function produced negative volumes?

# Shift range of nodes_range for internal faces (since it will be appended)
iface_nodes_range .= [
    iface_nodes_range[i] .+ length(bface_nodes) for i ∈ eachindex(iface_nodes_range)
    ]

# Concatenate boundary and internal faces
face_nodes = vcat(bface_nodes, iface_nodes)
face_nodes_range = vcat(bface_nodes_range, iface_nodes_range)
face_owner_cells = vcat(bface_owners_cells, iface_owners_cells)

# Sort out cell to face connectivity
cell_faces, cell_nsign, cell_faces_range, cell_neighbours = begin
    FVM_1D.UNV_3D.generate_cell_face_connectivity(volumes, nbfaces, face_owner_cells) # Hybrid compatible. Hexa and tet tested.
end

# Build mesh (without calculation of geometry/properties)
cells = FVM_1D.UNV_3D.build_cells(cell_nodes_range, cell_faces_range) # Hybrid compatible. Hexa tested.
faces = FVM_1D.UNV_3D.build_faces(face_nodes_range, face_owner_cells) # Hybrid compatible. Hexa tested.

all_cell_faces = Vector{Int64}[Int64[] for _ ∈ eachindex(cells)]
    for fID ∈ eachindex(faces)
        owners = faces[fID].ownerCells
        owner1 = owners[1]
        owner2 = owners[2]
        #if faces_cpu[fID].ownerCells[1]==cID || faces_cpu[fID].ownerCells[2]==cID
            push!(all_cell_faces[owner1],fID)
            if owner1 !== owner2 #avoid duplication of cells for boundary faces
                push!(all_cell_faces[owner2],fID)
            end
        #end
    end

    all_cell_faces

    #function generate_internal_faces(volumes, nbfaces, nodes, node_cells)
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
        total_faces
    
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
            # Generate faces for prism elements, using pattern in UNV file.
            if volume.volumeCount == 6
                nodesID = volume.volumes
                push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[3]]) # Triangle 1
                #push!(cells_faces_nodeIDs[cellID], Int64[nodesID[3], nodesID[4], nodesID[5]]) # Triangle 2
                push!(cells_faces_nodeIDs[cellID], Int64[nodesID[4], nodesID[5], nodesID[6]]) # Triangle 2
                push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[4], nodesID[5]]) # Rectangle 1
                push!(cells_faces_nodeIDs[cellID], Int64[nodesID[2], nodesID[3], nodesID[5], nodesID[6]]) # Rectangle 2
                push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[3], nodesID[4], nodesID[6]]) # Rectangle 3
            end
        end
        cells_faces_nodeIDs
    
        # Sort nodesIDs for each face based on ID (need to correct order later to be physical)
        # this allows to find duplicates later more easily (query on ordered ids is faster)
        for face_nodesID ∈ cells_faces_nodeIDs
            for nodesID ∈ face_nodesID
                sort!(nodesID)
            end
        end
        cells_faces_nodeIDs
    
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
        owners_cellIDs
    
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
        face_nodes
    
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
        total_bfaces
    
        bfaces_indices = zeros(Int64, total_bfaces) # preallocate memory
        counter = 0
        for (i, owners) ∈ enumerate(owners_cellIDs)
            if owners[1] == 0
                counter += 1
                bfaces_indices[counter] = i # contains indices of faces to remove
            end
        end
        bfaces_indices
    
        deleteat!(owners_cellIDs, bfaces_indices)
        deleteat!(face_nodes, bfaces_indices)
    
        println("Removing ", total_bfaces, " (from ", nbfaces, ") boundary faces")
    
        # Generate face_nodes_range
        face_nodes_range = Vector{UnitRange{Int64}}(undef, length(face_nodes))
        start = 1
        for (fID, nodesID) ∈ enumerate(face_nodes)
            nnodes = length(nodesID)
            face_nodes_range[fID] = UnitRange{Int64}(start:(start + nnodes - 1))
            start += nnodes
        end
        face_nodes_range
    
        # Flatten array i.e. go from Vector{Vector{Int}} to Vector{Int}
        face_nodes = vcat(face_nodes...) 
    
        return face_nodes, face_nodes_range, owners_cellIDs
    