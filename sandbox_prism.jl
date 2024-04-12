using FVM_1D
using StaticArrays
using Statistics
using LinearAlgebra
#include("src/VTK_3D/VTU.jl")


unv_mesh="src/UNV_3D/TET_PRISM_HM.unv"
unv_mesh="src/UNV_3D/Quad_cell_new_boundaries.unv"
unv_mesh="src/UNV_3D/HEXA_HM.unv"
unv_mesh="src/UNV_3D/TET_HM.unv"
unv_mesh="src/UNV_3D/3D_cylinder_HEX_PRISM.unv"

@time mesh = build_mesh3D(unv_mesh)
mesh.faces
mesh.cells
mesh.boundaries

name="tet_prism"

write_vtk(name, mesh::Mesh3)

points, efaces, volumes, boundaryElements = load_3D(unv_mesh,scale=1, integer=Int64, float=Float64)

points
efaces
volumes
boundaryElements

cell_nodes, cell_nodes_range = FVM_1D.UNV_3D.generate_cell_nodes(volumes) # Should be Hybrid compatible, tested for hexa. Using push instead of allocating vector.
node_cells, node_cells_range = FVM_1D.UNV_3D.generate_node_cells(points, volumes)  # Should be Hybrid compatible, tested for hexa.
nodes = FVM_1D.UNV_3D.build_nodes(points, node_cells_range) # Hyrbid compatible, works for Tet and Hexa
boundaries = FVM_1D.UNV_3D.build_boundaries(boundaryElements) # Hybrid compatible

nbfaces = sum(length.(getproperty.(boundaries, :IDs_range)))


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
        # Old method needs clean up

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
        faces = FVM_1D.UNV_3D.build_faces(face_nodes_range, face_owner_cells)

        store=[]
        for i=1:length(faces)
            push!(store,faces[i].ownerCells)
        end
        store
        #unique(store)

        unique_indices = unique(i -> store[i], eachindex(store))
    #unique!(face_nodes)
    keepat!(faces, unique_indices)

    #function generate_boundary_faces(
        # boundaryElements, efaces, nbfaces, node_cells, node_cells_range, volumes
        # )
        bface_nodes = Vector{Vector{Int64}}(undef, nbfaces)
        bface_nodes_range = Vector{UnitRange{Int64}}(undef, nbfaces)
        bowners_cells = Vector{Int64}[Int64[0,0] for _ ∈ 1:nbfaces]
        boundary_cells = Vector{Int64}(undef, nbfaces)
    
        fID = 0 # faceID index of output array (reordered)
        start = 1
        for boundary ∈ boundaryElements
            elements = boundary.elements
                for bfaceID ∈ elements
                    fID += 1
                    nnodes = length(efaces[bfaceID].faces)
                    nodeIDs = efaces[bfaceID].faces # Actually nodesIDs
                    bface_nodes[fID] = nodeIDs
                    bface_nodes_range[fID] = UnitRange{Int64}(start:(start + nnodes - 1))
                    start += nnodes
    
                    # Find owner cells (same as boundary cells)
                    assigned = false
                    for nodeID ∈ nodeIDs
                        cIDs = cellIDs(node_cells, node_cells_range, nodeID)
                        for cID ∈ cIDs
                            if intersect(nodeIDs, volumes[cID].volumes) == nodeIDs
                                bowners_cells[fID] .= cID
                                boundary_cells[fID] = cID
                                assigned = true
                                break
                            end
                        end
                        if assigned 
                            break
                        end
                    end
                end
        end
    
        bface_nodes = vcat(bface_nodes...) # Flatten - from Vector{Vector{Int}} to Vector{Int}
        # Check: Length of bface_nodes = no. of bfaces x no. of nodes of each bface
    
        bface_nodes
        bface_nodes_range
        return bface_nodes, bface_nodes_range, bowners_cells, boundary_cells
    #end