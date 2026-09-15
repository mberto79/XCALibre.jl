# src/UNV3/UNV3_2_builder.jl

using LinearAlgebra

export UNV3D_mesh

# ==============================================================================
# TOP-LEVEL API
# ==============================================================================

"""
    UNV3D_mesh(unv_mesh::String; scale::Real=1.0, integer_type::Type=Int64, float_type::Type=Float64) -> Mesh3

Constructs a `Mesh3` object from a Universal (UNV) file format.

# Arguments
- `unv_mesh::String`: The file path to the UNV mesh file.

# Keyword Arguments
- `scale::Real=1.0`: A scaling factor applied to the nodal coordinates.
- `integer_type::Type=Int64`: The integer type used for topological indices and connectivity arrays.
- `float_type::Type=Float64`: The floating-point type used for coordinates and geometric properties.

# Returns
- `Mesh3`: A fully constructed 3D mesh object containing nodes, cells, faces, boundaries, and populated geometric properties.
"""
function UNV3D_mesh(unv_mesh; scale=1.0, integer_type=Int64, float_type=Float64)
    return _UNV3D_mesh(unv_mesh, scale, integer_type, float_type)
end

# Type-parameter barrier keeps the build type-stable for non-default integer/float types.
function _UNV3D_mesh(unv_mesh, scale, ::Type{TI}, ::Type{TF}) where {TI<:Integer, TF<:AbstractFloat}
    local points, efaces, cells_UNV, boundaryElements
    # geometry is always built in Float64; conversion to float_type happens after a cheap check
    t_parse = @elapsed begin
        points, efaces, cells_UNV, boundaryElements = read_UNV3(
            unv_mesh; scale=scale, integer=TI, float=Float64)
    end
    @info "UNV file parsed in $(round(t_parse, digits=3)) seconds."

    @info "Generating mesh connectivity and geometry..."
    local mesh

    t_build = @elapsed begin
        mesh = _build_UNV3D_mesh_core(points, efaces, cells_UNV, boundaryElements, TI, Float64)
    end
    mesh = Mesh.convert_mesh_float(mesh, TF)

    @info "Mesh constructed in $(round(t_build, digits=3)) seconds."
    @info "Mesh entities: $(length(mesh.nodes)) nodes | $(length(mesh.faces)) faces | $(length(mesh.cells)) cells."
    
    return mesh
end

# ==============================================================================
# CORE WORKFLOW ORCHESTRATOR
# ==============================================================================

function _build_UNV3D_mesh_core(points, efaces, cells_UNV, boundaryElements, ::Type{I}, ::Type{F}) where {I, F}
    n_cells = length(cells_UNV)
    n_points = length(points)

    node_cells, node_cells_range = _build_node_to_cell_map(n_points, n_cells, cells_UNV, I)

    total_faces, n_bfaces, all_owners, face_nodes, face_nodes_range, boundary_cellsID = 
        _identify_and_concatenate_faces(cells_UNV, boundaryElements, efaces, node_cells, node_cells_range, n_cells, I)

    cell_faces_vec, cell_faces_range, cell_neigh_vec, cell_nsign_vec = 
        _build_cell_to_face_map(n_cells, total_faces, all_owners, n_bfaces, I)

    mesh = _construct_mesh_entities(
        points, cells_UNV, boundaryElements, 
        node_cells, node_cells_range, face_nodes, face_nodes_range, all_owners,
        cell_faces_vec, cell_faces_range, cell_neigh_vec, cell_nsign_vec, boundary_cellsID, 
        total_faces, n_points, n_cells, I, F
    )

    Mesh.compute_3d_geometry!(mesh)

    return mesh
end

# ==============================================================================
# CONNECTIVITY BUILDERS
# ==============================================================================

function _build_node_to_cell_map(n_points, n_cells, cells_UNV, ::Type{I}) where I
    node_cell_counts = zeros(I, n_points)
    for cID in 1:n_cells
        c_nodes = cells_UNV[cID].nodesID
        @inbounds for j in 1:length(c_nodes)
            node_cell_counts[c_nodes[j]] += 1
        end
    end
    
    node_cells_range, cursors_pts, total_nc = _compute_flat_offsets(node_cell_counts)
    node_cells = Vector{I}(undef, total_nc)
    
    for cID in 1:n_cells
        c_nodes = cells_UNV[cID].nodesID
        @inbounds for j in 1:length(c_nodes)
            nID = c_nodes[j]
            node_cells[cursors_pts[nID]] = I(cID)
            cursors_pts[nID] += 1
        end
    end
    
    return node_cells, node_cells_range
end

function _identify_and_concatenate_faces(cells_UNV, boundaryElements, efaces, node_cells, node_cells_range, n_cells, ::Type{I}) where I
    n_bfaces = sum(b -> length(b.facesID), boundaryElements)
    bface_nodes_list = Vector{Vector{I}}(undef, n_bfaces)
    boundary_cellsID = Vector{I}(undef, n_bfaces)
    bface_keys = Dict{NTuple{4, I}, I}()
    sizehint!(bface_keys, n_bfaces)

    f_idx = 1
    for boundary in boundaryElements
        for b_idx in boundary.facesID
            nIDs = efaces[b_idx].nodesID
            owner = I(0)
            
            rng = node_cells_range[nIDs[1]]
            @inbounds for ptr in rng[1]:rng[end]
                cID = node_cells[ptr]
                match = true
                c_nodes = cells_UNV[cID].nodesID
                for id in nIDs
                    if !(id in c_nodes)
                        match = false; break
                    end
                end
                if match; owner = I(cID); break; end
            end
            boundary_cellsID[f_idx] = owner
            
            # The key is sorted ONLY for hashing. 
            if length(nIDs) == 3
                bface_keys[make_key(I(nIDs[1]), I(nIDs[2]), I(nIDs[3]))] = f_idx
            else
                bface_keys[make_key(I(nIDs[1]), I(nIDs[2]), I(nIDs[3]), I(nIDs[4]))] = f_idx
            end
            
            # IMPLEMENTATION NOTE: The UNV strict ordered sequence is preserved here!
            bface_nodes_list[f_idx] = I.(nIDs)
            f_idx += 1
        end
    end

    maps_tet   = ((1,2,3), (1,2,4), (1,3,4), (2,3,4))
    maps_hex   = ((1,2,3,4), (5,6,7,8), (1,2,6,5), (2,3,7,6), (3,4,8,7), (4,1,5,8))
    maps_pri_3 = ((1,2,3), (4,5,6))
    maps_pri_4 = ((1,2,5,4), (2,3,6,5), (3,1,4,6))

    max_expected_ifaces = n_cells * 4
    iface_nodes_list = Vector{NTuple{4, I}}(undef, max_expected_ifaces)
    iface_sizes = Vector{I}(undef, max_expected_ifaces)
    iface_owners = Vector{Tuple{I, I}}(undef, max_expected_ifaces)
    iface_count = 0 
    
    iface_keys = Dict{NTuple{4, I}, I}()
    sizehint!(iface_keys, n_cells * 3)

    for cID in 1:n_cells
        cell = cells_UNV[cID]
        nc = cell.nodeCount
        
        # IMPLEMENTATION NOTE: cell.nodesID is extracted sequentially. 
        # The true 3D spatial orientation of the face is natively retained.
        if nc == 4
            for m in maps_tet
                @inbounds n1, n2, n3 = I(cell.nodesID[m[1]]), I(cell.nodesID[m[2]]), I(cell.nodesID[m[3]])
                key = make_key(n1, n2, n3)
                iface_count = _add_internal_face!(iface_nodes_list, iface_sizes, iface_owners, iface_keys, bface_keys, key, (n1, n2, n3, I(0)), I(3), I(cID), iface_count, I)
            end
        elseif nc == 6
            for m in maps_pri_3
                @inbounds n1, n2, n3 = I(cell.nodesID[m[1]]), I(cell.nodesID[m[2]]), I(cell.nodesID[m[3]])
                key = make_key(n1, n2, n3)
                iface_count = _add_internal_face!(iface_nodes_list, iface_sizes, iface_owners, iface_keys, bface_keys, key, (n1, n2, n3, I(0)), I(3), I(cID), iface_count, I)
            end
            for m in maps_pri_4
                @inbounds n1, n2, n3, n4 = I(cell.nodesID[m[1]]), I(cell.nodesID[m[2]]), I(cell.nodesID[m[3]]), I(cell.nodesID[m[4]])
                key = make_key(n1, n2, n3, n4)
                iface_count = _add_internal_face!(iface_nodes_list, iface_sizes, iface_owners, iface_keys, bface_keys, key, (n1, n2, n3, n4), I(4), I(cID), iface_count, I)
            end
        elseif nc == 8
            for m in maps_hex
                @inbounds n1, n2, n3, n4 = I(cell.nodesID[m[1]]), I(cell.nodesID[m[2]]), I(cell.nodesID[m[3]]), I(cell.nodesID[m[4]])
                key = make_key(n1, n2, n3, n4)
                iface_count = _add_internal_face!(iface_nodes_list, iface_sizes, iface_owners, iface_keys, bface_keys, key, (n1, n2, n3, n4), I(4), I(cID), iface_count, I)
            end
        end
    end

    total_faces = n_bfaces + iface_count
    all_owners = Vector{SVector{2, I}}(undef, total_faces)
    
    @inbounds for i in 1:n_bfaces
        all_owners[i] = SVector{2, I}(boundary_cellsID[i], boundary_cellsID[i])
    end
    @inbounds for i in 1:iface_count
        all_owners[n_bfaces + i] = SVector{2, I}(iface_owners[i][1], iface_owners[i][2])
    end
    
    total_fnodes = sum(@view iface_sizes[1:iface_count])
    for i in 1:n_bfaces; total_fnodes += length(bface_nodes_list[i]); end
    
    face_nodes = Vector{I}(undef, total_fnodes)
    face_nodes_range = Vector{UnitRange{I}}(undef, total_faces)
    
    let cursor = I(1)
        for i in 1:n_bfaces
            nodes_arr = bface_nodes_list[i]
            len = length(nodes_arr)
            for j in 1:len
                @inbounds face_nodes[cursor + j - 1] = nodes_arr[j]
            end
            face_nodes_range[i] = cursor:(cursor + I(len) - I(1))
            cursor += I(len)
        end
        
        for i in 1:iface_count
            t = iface_nodes_list[i]
            len = iface_sizes[i]
            @inbounds face_nodes[cursor] = t[1]
            @inbounds face_nodes[cursor+1] = t[2]
            @inbounds face_nodes[cursor+2] = t[3]
            if len == 4; @inbounds face_nodes[cursor+3] = t[4]; end
            face_nodes_range[n_bfaces + i] = cursor:(cursor + len - I(1))
            cursor += len
        end
    end

    return total_faces, n_bfaces, all_owners, face_nodes, face_nodes_range, boundary_cellsID
end

function _build_cell_to_face_map(n_cells, total_faces, all_owners, n_bfaces, ::Type{I}) where I
    cell_f_counts = zeros(I, n_cells)
    for fID in (n_bfaces + 1):total_faces
        o = all_owners[fID]
        @inbounds cell_f_counts[o[1]] += 1
        @inbounds cell_f_counts[o[2]] += 1
    end
    
    cell_faces_range, cursors_cf, total_cf = _compute_flat_offsets(cell_f_counts)
    
    cell_faces_vec = Vector{I}(undef, total_cf)
    cell_nsign_vec = Vector{I}(undef, total_cf)
    cell_neigh_vec = Vector{I}(undef, total_cf)
    
    for fID in (n_bfaces + 1):total_faces
        o = all_owners[fID]
        @inbounds begin
            p1, p2 = cursors_cf[o[1]], cursors_cf[o[2]]
            cell_faces_vec[p1] = fID; cell_nsign_vec[p1] = I(1);  cell_neigh_vec[p1] = o[2]
            cell_faces_vec[p2] = fID; cell_nsign_vec[p2] = I(-1); cell_neigh_vec[p2] = o[1]
            cursors_cf[o[1]] += 1; cursors_cf[o[2]] += 1
        end
    end
    
    return cell_faces_vec, cell_faces_range, cell_neigh_vec, cell_nsign_vec
end

function _construct_mesh_entities(points, cells_UNV, boundaryElements, 
                                  node_cells, node_cells_range, face_nodes, face_nodes_range, all_owners,
                                  cell_faces_vec, cell_faces_range, cell_neigh_vec, cell_nsign_vec, boundary_cellsID,
                                  total_faces, n_points, n_cells, ::Type{I}, ::Type{F}) where {I, F}

    boundaries = Vector{Boundary{Symbol, UnitRange{I}}}(undef, length(boundaryElements))
    let cursor = I(1)
        for i in eachindex(boundaryElements)
            len = length(boundaryElements[i].facesID)
            boundaries[i] = Boundary(Symbol(boundaryElements[i].name), cursor:(cursor + I(len) - I(1)))
            cursor += I(len)
        end
    end

    proto_node = Node(SVector{3, F}(zero(F),zero(F),zero(F)), I(1):I(0))
    nodes = Vector{typeof(proto_node)}(undef, n_points)
    
    for i in 1:n_points
        p_xyz = points[i].xyz
        svec_coords = SVector{3, F}(F(p_xyz[1]), F(p_xyz[2]), F(p_xyz[3]))
        @inbounds nodes[i] = Node(svec_coords, node_cells_range[i])
    end

    tot_c_nodes = sum(c -> c.nodeCount, cells_UNV)
    all_cell_nodes = Vector{I}(undef, tot_c_nodes)
    
    proto_cell = Cell(SVector{3, F}(zero(F),zero(F),zero(F)), zero(F), I(1):I(0), I(1):I(0))
    cells = Vector{typeof(proto_cell)}(undef, n_cells)
    
    let cursor = I(1)
        for i in 1:n_cells
            c_nodes = cells_UNV[i].nodesID
            len = cells_UNV[i].nodeCount
            for j in 1:len
                @inbounds all_cell_nodes[cursor + j - 1] = I(c_nodes[j])
            end
            cells[i] = Cell(SVector{3, F}(zero(F),zero(F),zero(F)), zero(F), cursor:(cursor + I(len) - I(1)), cell_faces_range[i])
            cursor += I(len)
        end
    end

    proto_face = Face3D(I(1):I(0), SVector{2, I}(I(0),I(0)), 
             SVector{3, F}(zero(F),zero(F),zero(F)), 
             SVector{3, F}(zero(F),zero(F),zero(F)), 
             SVector{3, F}(zero(F),zero(F),zero(F)),
             zero(F), zero(F), zero(F))
    faces = Vector{typeof(proto_face)}(undef, total_faces)
    
    for i in 1:total_faces
        @inbounds faces[i] = Face3D(face_nodes_range[i], all_owners[i], 
             SVector{3, F}(zero(F),zero(F),zero(F)), 
             SVector{3, F}(zero(F),zero(F),zero(F)), 
             SVector{3, F}(zero(F),zero(F),zero(F)),
             zero(F), zero(F), zero(F))
    end

    return Mesh3(
        cells, all_cell_nodes, cell_faces_vec, cell_neigh_vec, cell_nsign_vec,
        faces, face_nodes, boundaries, nodes, node_cells,
        SVector{3, F}(zero(F),zero(F),zero(F)), I(0):I(0), boundary_cellsID
    )
end

# ==============================================================================
# UTILITY AND AUXILIARY FUNCTIONS
# ==============================================================================

function _compute_flat_offsets(counts::Vector{I}) where I
    n = length(counts)
    ranges = Vector{UnitRange{I}}(undef, n)
    cursors = zeros(I, n)
    cursor = I(1)
    @inbounds for i in 1:n
        c = counts[i]
        ranges[i] = cursor:(cursor + c - I(1))
        cursors[i] = cursor
        cursor += c
    end
    return ranges, cursors, cursor - I(1)
end

@inline function _add_internal_face!(iface_nodes_list, iface_sizes, iface_owners, iface_keys, bface_keys, key, nodes_tuple, face_size::I, cID::I, count::Int, ::Type{I}) where I
    if haskey(bface_keys, key); return count; end
    
    idx = get(iface_keys, key, I(0))
    if idx > 0
        @inbounds iface_owners[idx] = (iface_owners[idx][1], cID)
        return count
    else
        count += 1
        if count > length(iface_nodes_list)
            resize!(iface_nodes_list, count * 2)
            resize!(iface_sizes, count * 2)
            resize!(iface_owners, count * 2)
        end
        
        # IMPLEMENTATION NOTE: The ordered `nodes_tuple` is stored directly!
        @inbounds iface_nodes_list[count] = nodes_tuple
        @inbounds iface_sizes[count] = face_size
        @inbounds iface_owners[count] = (cID, I(0))
        iface_keys[key] = count
        return count
    end
end

@inline function make_key(a::T, b::T, c::T) where T
    if a > b; a, b = b, a; end
    if b > c; b, c = c, b; end
    if a > b; a, b = b, a; end
    return (a, b, c, zero(T))
end

@inline function make_key(a::T, b::T, c::T, d::T) where T
    if a > b; a, b = b, a; end
    if c > d; c, d = d, c; end
    if a > c; a, c = c, a; end
    if b > d; b, d = d, b; end
    if b > c; b, c = c, b; end
    return (a, b, c, d)
end
