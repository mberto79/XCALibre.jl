export UNV3D_mesh

"""
    UNV3D_mesh(unv_mesh; scale=1, integer_type=Int64, float_type=Float64)

Read and convert 3D UNV mesh file into XCALibre.jl.
"""
function UNV3D_mesh(unv_mesh; scale=1.0, integer_type=Int64, float_type=Float64)
    itype = integer_type
    ftype = float_type
    stats = @timed begin
        println("Loading UNV File...")
        @time points, efaces, cells_UNV, boundaryElements = read_UNV3( 
            unv_mesh; scale=scale, integer=itype, float=ftype)
        println("File Read Successfully")
        println("Generating Mesh...")

        cell_nodes, cell_nodes_range = generate_cell_nodes(cells_UNV, itype, ftype) 
        node_cells, node_cells_range = generate_node_cells(points, cells_UNV, itype, ftype)  
        nodes = build_nodes(points, node_cells_range, itype, ftype) 
        boundaries = build_boundaries(boundaryElements, itype, ftype) 

        nbfaces = sum(length.(getproperty.(boundaries, :IDs_range))) 

        bface_nodes, bface_nodes_range, bface_owners_cells, boundary_cellsID = 
        begin
            generate_boundary_faces(boundaryElements, efaces, nbfaces, node_cells, node_cells_range, cells_UNV, itype, ftype) 
        end

        iface_nodes, iface_nodes_range, iface_owners_cells = 
        begin 
            generate_internal_faces(cells_UNV, nbfaces, nodes, node_cells, itype, ftype) 
        end

        # REORDER NODES: Critical for area/volume accuracy.
        # Fixes bow-tie polygons caused by identification sorting.
        bface_nodes, iface_nodes = order_face_nodes(
            bface_nodes_range, iface_nodes_range, bface_nodes, iface_nodes, nodes, itype, ftype)

        iface_nodes_range .= [
            iface_nodes_range[i] .+ length(bface_nodes) for i ∈ eachindex(iface_nodes_range)
            ]

        face_nodes = vcat(bface_nodes, iface_nodes)
        face_nodes_range = vcat(bface_nodes_range, iface_nodes_range)
        face_owner_cells = vcat(bface_owners_cells, iface_owners_cells)

        cell_faces, cell_nsign, cell_faces_range, cell_neighbours = begin
            generate_cell_face_connectivity(
                cells_UNV, nbfaces, face_owner_cells, itype, ftype) 
        end

        cells = build_cells(cell_nodes_range, cell_faces_range, itype, ftype) 
        faces = build_faces(face_nodes_range, face_owner_cells, itype, ftype) 

        mesh = Mesh3(
            cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign, 
            faces, face_nodes, boundaries, 
            nodes, node_cells,
            SVector{3, ftype}(0.0, 0.0, 0.0), UnitRange{itype}(0, 0), boundary_cellsID
        ) 

        # --- GEOMETRY PIPELINE ---
        calculate_centres!(mesh, itype, ftype)
        calculate_area_and_volume!(mesh, itype, ftype) 
        calculate_face_properties!(mesh, itype, ftype) 

        return mesh
    end
end

# Helpers
get_data(array, range, index) = @view array[range[index]]
get_data(array, range) =  array[range] 
nodeIDs = get_data
faceIDs = get_data
cellIDs = get_data

function alpha_sort(v1, v2, normal)
    m1, m2 = norm(v1), norm(v2)
    if m1 < 1e-15 || m2 < 1e-15; return 0.0; end
    u1, u2 = v1/m1, v2/m2
    ang = acosd(clamp(u1 ⋅ u2, -1.0, 1.0))
    return dot(cross(u1, u2), normal) < 0 ? 360.0 - ang : ang
end

# BUILD Functions
build_cells(cell_nodes_range, cell_faces_range, itype, ftype) = begin
    cells = [Cell(Int64, ftype) for _ ∈ eachindex(cell_faces_range)]
    for cID ∈ eachindex(cell_nodes_range)
        cell = cells[cID]
        @reset cell.nodes_range = cell_nodes_range[cID]
        @reset cell.faces_range = cell_faces_range[cID]
        cells[cID] = cell
    end
    return cells
end

build_faces(face_nodes_range, face_owner_cells, itype, ftype) = begin
    faces = [Face3D(Int64, ftype) for _ ∈ eachindex(face_nodes_range)]
    for fID ∈ eachindex(face_nodes_range)
        face = faces[fID]
        @reset face.nodes_range = face_nodes_range[fID]
        @reset face.ownerCells = SVector{2,Int64}(face_owner_cells[fID])
        faces[fID] = face 
    end
    return faces
end

function build_nodes(points, node_cells_range, itype, ftype) 
    nodes = [Node(SVector{3, ftype}(0.0,0.0,0.0), 1:1) for _ ∈ eachindex(points)]
    @inbounds for i ∈ eachindex(points)
        nodes[i] =  Node(points[i].xyz, node_cells_range[i])
    end
    return nodes
end

function build_boundaries(boundaryElements, itype, ftype)
    bfaces_start = 1
    boundaries = Vector{Boundary{Symbol,UnitRange{Int64}}}(undef,length(boundaryElements))
    for (i, boundaryElement) ∈ enumerate(boundaryElements)
        bfaces = length(boundaryElement.facesID)
        bfaces_range = UnitRange{Int64}(bfaces_start:(bfaces_start + bfaces - 1))
        boundaries[i] = Boundary(Symbol(boundaryElement.name), bfaces_range)
        bfaces_start += bfaces
    end
    return boundaries
end

# GENERATE Functions
function generate_cell_nodes(cells_UNV, itype, ftype)
    cell_nodes = Int64[] 
    for n = eachindex(cells_UNV)
        for i = 1:cells_UNV[n].nodeCount
            push!(cell_nodes,cells_UNV[n].nodesID[i])
        end
    end
    cell_nodes_range = Vector{UnitRange{Int64}}(undef, length(cells_UNV)) 
    x = 0
    for i = eachindex(cells_UNV)
        cell_nodes_range[i] = UnitRange(x + 1, x + length(cells_UNV[i].nodesID))
        x = x + length(cells_UNV[i].nodesID)
    end
    return cell_nodes, cell_nodes_range
end

function generate_node_cells(points, cells_UNV, itype, ftype)
    temp_node_cells = [Int64[] for _ ∈ eachindex(points)] 
    for (cellID, cell) ∈ enumerate(cells_UNV)
        for nodeID ∈ cell.nodesID
            push!(temp_node_cells[nodeID], cellID)
        end
    end 
    node_cells_size = sum(length.(temp_node_cells)) 
    node_cells_index = 0 
    node_cells = zeros(Int64, node_cells_size)
    node_cells_range = [UnitRange{Int64}(1, 1) for _ ∈ eachindex(points)]
    for (nodeID, cellsID) ∈ enumerate(temp_node_cells)
        for cellID ∈ cellsID
            node_cells_index += 1
            node_cells[node_cells_index] = cellID
        end
        node_cells_range[nodeID] = UnitRange{Int64}(node_cells_index - length(cellsID) + 1, node_cells_index)
    end
    return node_cells, node_cells_range
end 

function generate_boundary_faces(boundaryElements, efaces, nbfaces, node_cells, node_cells_range, cells_UNV, itype, ftype)
    bface_nodes = Vector{Vector{Int64}}(undef, nbfaces)
    bface_nodes_range = Vector{UnitRange{Int64}}(undef, nbfaces)
    bowners_cells = Vector{Int64}[Int64[0,0] for _ ∈ 1:nbfaces]
    boundary_cells = Vector{Int64}(undef, nbfaces)
    fID = 0 
    start = 1
    for boundary ∈ boundaryElements
        elements = boundary.facesID
            for bfaceID ∈ elements
                fID += 1
                nnodes = length(efaces[bfaceID].nodesID)
                nodeIDs = efaces[bfaceID].nodesID 
                bface_nodes[fID] = nodeIDs
                bface_nodes_range[fID] = UnitRange{Int64}(start:(start + nnodes - 1))
                start += nnodes
                assigned = false
                for nodeID ∈ nodeIDs
                    cIDs = cellIDs(node_cells, node_cells_range, nodeID)
                    for cID ∈ cIDs
                        if intersect(nodeIDs, cells_UNV[cID].nodesID) == nodeIDs
                            bowners_cells[fID] .= cID
                            boundary_cells[fID] = cID
                            assigned = true
                            break
                        end
                    end
                    if assigned; break; end
                end
            end
    end
    bface_nodes = vcat(bface_nodes...) 
    return bface_nodes, bface_nodes_range, bowners_cells, boundary_cells
end

function generate_internal_faces(cells_UNV, nbfaces, nodes, node_cells, itype, ftype)
    total_faces = 0
    for cell ∈ cells_UNV
        if cell.nodeCount == 4; total_faces += 4; end
        if cell.nodeCount == 8; total_faces += 6; end
        if cell.nodeCount == 6; total_faces += 5; end
    end
    cells_faces_nodeIDs = Vector{Vector{Int64}}[Vector{Int64}[] for _ ∈ 1:length(cells_UNV)] 
    for (cellID, cell) ∈ enumerate(cells_UNV)
        # Sequence nodes correctly for standard UNV elements
        if cell.nodeCount == 4
            nodesID = cell.nodesID
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[3]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[3], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[2], nodesID[3], nodesID[4]])
        end
        if cell.nodeCount == 8
            nodesID = cell.nodesID
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[3], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[5], nodesID[6], nodesID[7], nodesID[8]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[6], nodesID[5]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[2], nodesID[3], nodesID[7], nodesID[6]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[3], nodesID[4], nodesID[8], nodesID[7]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[4], nodesID[1], nodesID[5], nodesID[8]])
        end
        if cell.nodeCount == 6
            nodesID = cell.nodesID
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[3]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[4], nodesID[5], nodesID[6]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[1], nodesID[2], nodesID[5], nodesID[4]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[2], nodesID[3], nodesID[6], nodesID[5]])
            push!(cells_faces_nodeIDs[cellID], Int64[nodesID[3], nodesID[1], nodesID[4], nodesID[6]])
        end
    end
    for face_nodesID ∈ cells_faces_nodeIDs
        for nodesID ∈ face_nodesID
            sort!(nodesID) # For unique identification
        end
    end
    owners_cellIDs = Vector{Int64}[zeros(Int64, 2) for _ ∈ 1:total_faces]
    facei = 0 
    for (cellID, faces_nodeIDs) ∈ enumerate(cells_faces_nodeIDs) 
        for facei_nodeIDs ∈ faces_nodeIDs 
            facei += 1 
            owners_cellIDs[facei][1] = cellID 
            for nodeID ∈ facei_nodeIDs 
                cells_range = nodes[nodeID].cells_range
                node_cellIDs = @view node_cells[cells_range] 
                for nodei_cellID ∈ node_cellIDs 
                    if nodei_cellID !== cellID 
                        for face ∈ cells_faces_nodeIDs[nodei_cellID]
                            if face == facei_nodeIDs
                                owners_cellIDs[facei][2] = nodei_cellID 
                                break
                            end
                        end
                    end
                end
            end
        end
    end
    sort!.(owners_cellIDs) 
    face_nodes = Vector{Int64}[Int64[] for _ ∈ 1:total_faces] 
    fID = 0 
    for celli_faces_nodeIDs ∈ cells_faces_nodeIDs
        for nodesID ∈ celli_faces_nodeIDs
            fID += 1
            face_nodes[fID] = nodesID
        end
    end
    unique_indices = unique(i -> face_nodes[i], eachindex(face_nodes))
    unique!(face_nodes)
    keepat!(owners_cellIDs, unique_indices)
    total_bfaces = 0 
    for owners ∈ owners_cellIDs
        if owners[1] == 0; total_bfaces += 1; end
    end
    bfaces_indices = zeros(Int64, total_bfaces) 
    counter = 0
    for (i, owners) ∈ enumerate(owners_cellIDs)
        if owners[1] == 0
            counter += 1
            bfaces_indices[counter] = i 
        end
    end
    deleteat!(owners_cellIDs, bfaces_indices)
    deleteat!(face_nodes, bfaces_indices)
    face_nodes_range = Vector{UnitRange{Int64}}(undef, length(face_nodes))
    start = 1
    for (fID, nodesID) ∈ enumerate(face_nodes)
        nnodes = length(nodesID)
        face_nodes_range[fID] = UnitRange{Int64}(start:(start + nnodes - 1))
        start += nnodes
    end
    face_nodes = vcat(face_nodes...) 
    return face_nodes, face_nodes_range, owners_cellIDs
end

function order_face_nodes(bface_nodes_range, iface_nodes_range, bface_nodes, iface_nodes, nodes, itype, ftype)
    # Generic re-ordering for polygons >= 3 nodes using angular coordinates
    for (range, f_nodes) in [(bface_nodes_range, bface_nodes), (iface_nodes_range, iface_nodes)]
        for fID in eachindex(range)
            nIDs = nodeIDs(f_nodes, range[fID])
            if length(nIDs) < 3; continue; end
            
            pts = getproperty.(nodes[nIDs], :coords)
            c_f = sum(pts) / length(pts)
            
            # Robust normal identification independent of node order
            v1 = pts[1] - c_f
            max_cross = -1.0
            best_normal = SVector{3, ftype}(0, 0, 1)
            for i in 2:length(pts)
                n_curr = cross(v1, pts[i] - c_f)
                mag = norm(n_curr)
                if mag > max_cross
                    max_cross = mag
                    best_normal = n_curr / mag
                end
            end
            
            angles = [alpha_sort(v1, p - c_f, best_normal) for p in pts]
            sorted_indices = sortperm(angles)
            
            for (i, idx) in enumerate(range[fID])
                f_nodes[idx] = nIDs[sorted_indices[i]]
            end
        end
    end
    return bface_nodes, iface_nodes
end

function generate_cell_face_connectivity(cells_UNV, nbfaces, face_owner_cells, itype, ftype)
    cell_faces = Vector{Int64}[Int64[] for _ ∈ eachindex(cells_UNV)] 
    cell_nsign = Vector{Int64}[Int64[] for _ ∈ eachindex(cells_UNV)] 
    cell_neighbours = Vector{Int64}[Int64[] for _ ∈ eachindex(cells_UNV)] 
    cell_faces_range = UnitRange{Int64}[UnitRange{Int64}(0,0) for _ ∈ eachindex(cells_UNV)] 
    first_internal_face = nbfaces + 1
    total_faces = length(face_owner_cells)
    for fID ∈ first_internal_face:total_faces
        owners = face_owner_cells[fID] 
        owner1 = owners[1]
        owner2 = owners[2]
        push!(cell_faces[owner1], fID)     
        push!(cell_faces[owner2], fID)
        push!(cell_nsign[owner1], 1)      
        push!(cell_nsign[owner2], -1)   
        push!(cell_neighbours[owner1], owner2)     
        push!(cell_neighbours[owner2], owner1)     
    end
    start = 1
    for (cID, faces) ∈ enumerate(cell_faces)
        nfaces = length(faces)
        cell_faces_range[cID] = UnitRange{Int64}(start:(start + nfaces - 1))
        start += nfaces
    end
    cell_faces = vcat(cell_faces...)
    cell_nsign = vcat(cell_nsign...)
    cell_neighbours = vcat(cell_neighbours...) 
    return cell_faces, cell_nsign, cell_faces_range, cell_neighbours
end

# CALCULATION functions

calculate_centres!(mesh, itype, ftype) = begin
    (; nodes, cells, faces, cell_nodes, face_nodes) = mesh
    for cID ∈ eachindex(cells)
        cell = cells[cID]
        n_ids = nodeIDs(cell_nodes, cell.nodes_range)
        @reset cell.centre = sum(getproperty.(nodes[n_ids], :coords)) / length(n_ids)
        cells[cID] = cell
    end
    for fID ∈ eachindex(faces)
        face = faces[fID]
        n_ids = nodeIDs(face_nodes, face.nodes_range)
        @reset face.centre = sum(getproperty.(nodes[n_ids], :coords)) / length(n_ids)
        faces[fID] = face
    end        
end

calculate_face_properties!(mesh, itype, ftype) = begin
    (; nodes, cells, faces, face_nodes, boundary_cellsID) = mesh
    n_bfaces = length(boundary_cellsID)
    n_faces = length(mesh.faces)

    for fID ∈ 1:n_faces
        face = faces[fID]
        nIDs = nodeIDs(face_nodes, face.nodes_range)
        (; ownerCells) = face
        F1 = face.centre
        C1 = cells[ownerCells[1]].centre 
        normal = face.normal 
        
        # Consistent normal orientation: away from owner
        if (F1 - C1) ⋅ normal < 0
            normal *= -one(ftype)
            face_nodes[face.nodes_range] .= reverse(nIDs)
            @reset face.normal = normal
        end

        C1F1 = F1 - C1
        if fID <= n_bfaces
            weight, delta, e = Mesh.weight_delta_e(C1F1, normal)
        else
            C2 = cells[ownerCells[2]].centre 
            C2F1 = F1 - C2
            C1C2 = C2 - C1
            weight, delta, e = Mesh.weight_delta_e(C1F1, C2F1, C1C2, normal)
        end
        
        @reset face.delta = delta
        @reset face.e = e
        @reset face.weight = weight
        faces[fID] = face
    end
end

calculate_area_and_volume!(mesh, itype, ftype) = begin
    (; nodes, faces, face_nodes, cells, cell_nodes, boundary_cellsID) = mesh
    
    # 1. Robust Area vector integration
    for fID ∈ eachindex(faces)
        face = faces[fID]
        nIDs = nodeIDs(face_nodes, face.nodes_range)
        pts = getproperty.(nodes[nIDs], :coords)
        area_vec = SVector{3, ftype}(0.0, 0.0, 0.0)
        fc = face.centre
        for i in 1:length(pts)
            p1 = pts[i] - fc
            p2 = pts[mod1(i+1, length(pts))] - fc
            area_vec += 0.5 * cross(p1, p2)
        end
        area = norm(area_vec)
        @reset face.area = area
        @reset face.normal = area_vec / (area + eps(ftype))
        faces[fID] = face
    end

    # 2. Map all faces (Boundary + Internal) per cell
    all_cell_faces = [Int64[] for _ ∈ eachindex(cells)]
    for (fID, cID) in enumerate(boundary_cellsID)
        push!(all_cell_faces[cID], fID)
    end
    for fID in (length(boundary_cellsID)+1):length(faces)
        occ = faces[fID].ownerCells
        push!(all_cell_faces[occ[1]], fID)
        push!(all_cell_faces[occ[2]], fID)
    end

    # 3. Robust Volume via Divergence Theorem (Pyramid decomposition)
    for cID ∈ eachindex(cells)
        cell = cells[cID]
        apex = cell.centre 
        total_vol = zero(ftype)
        moment = SVector{3, ftype}(0.0, 0.0, 0.0)

        for fID in all_cell_faces[cID]
            face = faces[fID]
            # Outward normal relative to this cell
            Sf = face.normal * face.area
            if dot((face.centre - apex), Sf) < 0
                Sf *= -1.0
            end
            v_pyr = (1/3) * dot((face.centre - apex), Sf)
            c_pyr = 0.75 * face.centre + 0.25 * apex
            total_vol += v_pyr
            moment += v_pyr * c_pyr
        end
        
        @reset cell.volume = abs(total_vol)
        @reset cell.centre = moment / (abs(total_vol) + eps(ftype))
        cells[cID] = cell
    end
end