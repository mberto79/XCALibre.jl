
function compute_geometry!(mesh)
    mesh = calculate_cell_centres!(mesh)
    mesh = calculate_face_centres!(mesh)
    mesh = calculate_face_properties!(mesh)
    mesh = calculate_cell_volumes!(mesh)
    return mesh
end

function calculate_cell_centres!(mesh)
    (; nodes, cells, cell_nodes) = mesh
    TF = eltype(nodes[1].coords)
    for (cID, cell) ∈ enumerate(cells)
        sum = SVector{3, TF}(0, 0, 0)
        nodesID = @view cell_nodes[cell.nodes_range]
        for nID ∈ nodesID
            sum += nodes[nID].coords
        end
        @reset cell.centre = sum/length(nodesID)
        cells[cID] = cell
    end
    return mesh
end

function calculate_face_centres!(mesh)
    (; nodes, faces, face_nodes) = mesh
    TF = eltype(nodes[1].coords)
    for (fID, face) ∈ enumerate(faces)
        sum = SVector{3, TF}(0, 0, 0)
        nodesID = @view face_nodes[face.nodes_range]
        for nID ∈ nodesID
            sum += nodes[nID].coords
        end
        @reset face.centre = sum/length(nodesID)
        faces[fID] = face
    end
    return mesh
end

# Sub-triangle (fan) face geometry: returns unit normal, area and area-weighted centroid.
# area_vec = Σ (pᵢ-apex)×(pᵢ₊₁-apex)/2 is independent of apex for a closed loop, so the simple
# node-average apex is fine. normal*area == area_vec keeps cell volumes exact via the divergence
# theorem. Polyhedral faces need not be planar; degenerate sub-triangles are skipped to avoid NaNs.
function _face_geometry(nodes, nIDs, apex::SVector{3, TF}) where {TF}
    area_vec = SVector{3, TF}(0, 0, 0)
    centre_sum = SVector{3, TF}(0, 0, 0)
    sum_area = zero(TF)
    n = length(nIDs)
    @inbounds for i ∈ 1:n
        j = i == n ? 1 : i + 1
        p1 = nodes[nIDs[i]].coords
        p2 = nodes[nIDs[j]].coords
        tri_vec = ((p1 - apex) × (p2 - apex))/2
        tri_area = norm(tri_vec)
        tri_area > zero(TF) || continue
        area_vec += tri_vec
        centre_sum += tri_area*(apex + p1 + p2)/3
        sum_area += tri_area
    end
    mag = norm(area_vec)
    normal = mag > zero(TF) ? area_vec/mag : SVector{3, TF}(0, 0, 0)
    centre = sum_area > zero(TF) ? centre_sum/sum_area : apex
    return normal, mag, centre
end

function calculate_face_properties!(mesh)
    (; nodes, cells, faces, face_nodes, boundary_cellsID) = mesh
    n_bfaces = length(boundary_cellsID)
    n_faces = length(faces)

    # boundary faces (ownerCells[1] == ownerCells[2]); delta measured to the cell centroid
    for fID ∈ 1:n_bfaces
        face = faces[fID]
        nIDs = @view face_nodes[face.nodes_range]
        normal, area, centre = _face_geometry(nodes, nIDs, face.centre)
        C1 = cells[face.ownerCells[1]].centre
        C1F1 = centre - C1
        C1F1 ⋅ normal < 0 && (normal = -normal) # orient out of the domain
        weight, delta, e = Mesh.weight_delta_e(C1F1, normal)
        @reset face.centre = centre
        @reset face.normal = normal
        @reset face.area = area
        @reset face.delta = delta
        @reset face.e = e
        @reset face.weight = weight
        faces[fID] = face
    end

    # internal faces
    for fID ∈ (n_bfaces + 1):n_faces
        face = faces[fID]
        nIDs = @view face_nodes[face.nodes_range]
        normal, area, centre = _face_geometry(nodes, nIDs, face.centre)
        C1 = cells[face.ownerCells[1]].centre
        C2 = cells[face.ownerCells[2]].centre
        C1F1 = centre - C1
        C2F1 = centre - C2
        C1C2 = C2 - C1
        C1C2 ⋅ normal < 0 && (normal = -normal) # orient owner1(low ID) -> owner2(high ID), keeps n·e ≥ 0
        weight, delta, e = Mesh.weight_delta_e(C1F1, C2F1, C1C2, normal)
        @reset face.centre = centre
        @reset face.normal = normal
        @reset face.area = area
        @reset face.delta = delta
        @reset face.e = e
        @reset face.weight = weight
        faces[fID] = face
    end
    return mesh
end

function calculate_cell_volumes!(mesh)
    (; faces, cells, cell_faces, cell_nsign, boundaries, boundary_cellsID) = mesh
    TF = _get_float(mesh)
    oneThird = TF(1/3)

    volumes = zeros(TF, length(cells))
    max_areas = zeros(TF, length(cells)) # largest face area per cell, to estimate degenerate volumes

    # internal faces (nsign orients the stored normal outward for the cell)
    for (cID, cell) ∈ enumerate(cells)
        fIDs = @view cell_faces[cell.faces_range]
        nsigns = @view cell_nsign[cell.faces_range]
        for (fID, nsign) ∈ zip(fIDs, nsigns)
            face = faces[fID]
            xf = face.centre - cell.centre
            volumes[cID] += oneThird*(xf⋅face.normal)*nsign*face.area
            max_areas[cID] = max(max_areas[cID], face.area)
        end
    end

    # boundary faces (normals outward by definition)
    for boundary ∈ boundaries
        cIDs = @view boundary_cellsID[boundary.IDs_range]
        for (cID, fID) ∈ zip(cIDs, boundary.IDs_range)
            face = faces[fID]
            xf = face.centre - cells[cID].centre
            volumes[cID] += oneThird*(xf⋅face.normal)*face.area
            max_areas[cID] = max(max_areas[cID], face.area)
        end
    end

    # write volumes; recover degenerate/sliver cells with a positive estimate
    fixed = 0
    for (cID, cell) ∈ enumerate(cells)
        V = volumes[cID]
        if !(isfinite(V) && V > zero(V))
            estimate = max_areas[cID]^TF(1.5)*TF(1e-3) # tiny fraction of a face-sized cube
            V = max(isfinite(V) ? abs(V) : zero(V), estimate)
            fixed += 1
        end
        @reset cell.volume = V
        cells[cID] = cell
    end
    fixed > 0 && @warn "compute_geometry!: $fixed cell(s) had non-positive volume (degenerate/sliver cells); replaced with positive estimates."

    return mesh
end